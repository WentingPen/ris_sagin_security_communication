import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from algorithm.ppo.main import _build_environment
from algorithm.traditional_AO.main import EPISODE_FIELDS, PPO_UPDATE_FIELDS, main, run_experiment
from algorithm.traditional_AO.solver import AOConfig, AOSolver, FrozenSlotEvaluator, GroupAction, project_beam, to_env_action
from system_model.params import get_system_params


ROOT = Path(__file__).resolve().parents[1]


def small_params(*extra):
    return get_system_params([
        "--num_ris_elements", "16", "--num_ris_groups", "4",
        "--ris_array_rows", "4", "--risk_grid_size", "4",
        "--max_episode_steps", "2", *extra,
    ])


class AOTests(unittest.TestCase):
    def make_env(self, *extra):
        env = _build_environment(small_params(*extra))
        env.reset(seed=2026)
        self.addCleanup(env.close)
        return env

    def test_group_expansion_and_power_ball(self):
        env = self.make_env()
        g = GroupAction(np.ones(32), np.linspace(-1, 1, 4), np.array([0, 1, 0, 1]))
        action = to_env_action(env, g)
        self.assertLessEqual(np.linalg.norm(np.r_[action['sat_beam_real'], action['sat_beam_imag']]), 1.000001)
        base = env.get_ris_phase_reference(np.r_[action['sat_beam_real'], action['sat_beam_imag']])
        phase = np.angle(action['ris_phase_real'] + 1j * action['ris_phase_imag'])
        residual = np.angle(np.exp(1j * (phase - base)))
        expected = np.radians(env.params.ris_phase_residual_max_deg) * g.phase[env.uav_ris.element_to_group]
        np.testing.assert_allclose(residual, expected, atol=2e-7)
        for index in range(4):
            np.testing.assert_array_equal(action['ris_element_mode'][env.uav_ris.element_to_group == index], g.mode[index])

    def test_frozen_candidate_order_and_environment_isolation(self):
        env = self.make_env()
        original = pickle.dumps(env)
        evaluator = FrozenSlotEvaluator(env)
        beam, _ = env.get_phase_only_reference()
        a = GroupAction(beam, np.zeros(4), np.zeros(4, dtype=int))
        b = GroupAction(beam, np.ones(4), np.ones(4, dtype=int))
        first_a, first_b = evaluator.evaluate(a), evaluator.evaluate(b)
        second_a, second_b = evaluator.evaluate(a), evaluator.evaluate(b)
        np.testing.assert_allclose(first_a, second_a, rtol=0, atol=0)
        np.testing.assert_allclose(first_b, second_b, rtol=0, atol=0)
        self.assertEqual(original, pickle.dumps(env))

    def test_monotonic_objective_replay_and_real_power_constraints(self):
        for jammer in ('0', '0.1'):
            with self.subTest(jammer=jammer):
                env = self.make_env('--jammer_power', jammer)
                original = pickle.dumps(env)
                result = AOSolver(AOConfig(max_iterations=2)).solve(env)
                self.assertEqual(original, pickle.dumps(env))
                self.assertGreaterEqual(result.objective + 1e-12, result.initial_objective)
                for entry in result.trace:
                    self.assertGreaterEqual(entry['improvement'], -1e-12)
                    for key in ('beam_gain', 'phase_gain', 'mode_gain'):
                        self.assertGreaterEqual(entry[key], -1e-12)
                _, reward, _, _, info = env.step(result.action)
                self.assertAlmostEqual(reward, result.predicted_reward, places=10)
                self.assertAlmostEqual(info['system_margin'], result.objective, places=10)
                self.assertLessEqual(info['satellite_power'], env.params.sat_max_power * (1 + 1e-10))
                self.assertLessEqual(info['ris_power'], env.params.ris_max_power * (1 + 1e-10))

    def test_infeasible_active_modes_use_existing_passive_repair(self):
        env = self.make_env('--ris_max_power', '1e-25')
        evaluator = FrozenSlotEvaluator(env)
        beam, _ = env.get_phase_only_reference()
        evaluator.evaluate(GroupAction(beam, np.zeros(4), np.ones(4, dtype=int)))
        self.assertTrue(evaluator.env.ris_action_repaired)
        self.assertEqual(evaluator.env.ris_power, 0)
        self.assertEqual(evaluator.env.uav_ris.element_mode.sum(), 0)

    def test_all_three_blocks_improve_a_known_objective(self):
        env = self.make_env()
        reference, _ = env.get_phase_only_reference()

        class QuadraticEvaluator:
            def __init__(self, env):
                self.env, self.evaluations = env, 0

            def evaluate(self, groups):
                self.evaluations += 1
                value = -float(
                    np.sum((project_beam(groups.beam) - 0.25 * reference) ** 2)
                    + np.sum((np.clip(groups.phase, -1, 1) - np.array([0.4, -0.3, 0.6, -0.2])) ** 2)
                    + np.sum(groups.mode != np.array([1, 0, 1, 0]))
                )
                return value, value

        with patch('algorithm.traditional_AO.solver.FrozenSlotEvaluator', QuadraticEvaluator):
            result = AOSolver(AOConfig(max_iterations=3)).solve(env)
        for key in ('beam_gain', 'phase_gain', 'mode_gain'):
            self.assertGreater(sum(entry[key] for entry in result.trace), 0)
        np.testing.assert_array_equal(result.groups.mode, [1, 0, 1, 0])

    def test_output_matches_mappo_columns_and_environment_statistics(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            run_dir = Path(directory) / 'run'
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_experiment(small_params(), AOConfig(max_iterations=1), 3, 2, run_dir)
            self.assertEqual(summary['completed_episodes'], 3)
            self.assertEqual(summary['total_env_steps'], 6)
            self.assertEqual(summary['completed_batches'], 2)
            with (run_dir/'episode_metrics.csv').open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f); rows = list(reader)
                self.assertEqual(reader.fieldnames, EPISODE_FIELDS)
            with (run_dir/'step_metrics.csv').open(newline='', encoding='utf-8') as f:
                steps = list(csv.DictReader(f))
            for row in rows:
                data = [s for s in steps if s['episode'] == row['episode']]
                self.assertAlmostEqual(float(row['return']), sum(float(s['reward']) for s in data))
                self.assertAlmostEqual(float(row['max_bob_capacity']), np.mean([float(s['bob_capacity']) for s in data]))
                self.assertAlmostEqual(float(row['outage_rate']), np.mean([int(s['outage']) for s in data]))
            with (run_dir/'update_metrics.csv').open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f); updates = list(reader)
                self.assertEqual(reader.fieldnames[:len(PPO_UPDATE_FIELDS)], PPO_UPDATE_FIELDS)
            self.assertEqual(len(updates), 2)
            self.assertEqual(updates[0]['actor_loss'], '')
            self.assertEqual(updates[0]['critic_loss'], '')
            self.assertGreater((run_dir/'training_curves.png').stat().st_size, 10000)
            self.assertEqual(json.loads((run_dir/'checkpoint.json').read_text(encoding='utf-8'))['completed_episodes'], 3)

    def test_metric_column_names_match_current_mappo_entrypoint(self):
        tree = ast.parse((ROOT.parent/'ppo/main.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('episode_fields', 'update_fields'):
                        expected = EPISODE_FIELDS if target.id == 'episode_fields' else PPO_UPDATE_FIELDS
                        self.assertEqual(ast.literal_eval(node.value), expected)

    def test_invalid_configuration_and_cli_fail_before_experiment(self):
        for kwargs in ({'max_iterations': 0}, {'gradient_step': float('nan')}, {'finite_difference': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AOConfig(**kwargs)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--episodes', '0'])


if __name__ == '__main__':
    unittest.main()
