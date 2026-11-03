import ast
from dataclasses import asdict, replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from algorithm.IPPO.buffer import RolloutBuffer
from algorithm.IPPO.Controller import Controller, IPPOConfig
from algorithm.IPPO.main import (
    EPISODE_FIELDS, UPDATE_FIELDS, LiveTrainingPlotter, MODEL_NAMES,
    _build_environment, _load_resume_checkpoint, _restore_controller,
    _save_checkpoint,
)
from algorithm.ppo.Controller import PPOConfig
from system_model.params import get_system_params


torch.set_num_threads(1)
IPPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = IPPO_ROOT / "result" / "_verification"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def small_environment(*extra):
    return _build_environment(get_system_params([
        "--risk_grid_size", "4", "--max_episode_steps", "3", *extra,
    ]))


def small_config(**overrides):
    return replace(IPPOConfig(
        episodes_per_update=2, batch_size=6, ppo_epochs=1,
        hidden_dim=32, risk_hidden_dim=8, risk_feature_dim=8,
    ), **overrides)


class PublicInfoEnvironment:
    """Make any access to the private critic observation fail immediately."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @staticmethod
    def strip_private(info):
        return {
            key: value for key, value in info.items()
            if key not in ("critic_observation", "critic_eve_nodes")
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self.strip_private(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self.strip_private(info)


class IPPOTests(unittest.TestCase):
    def test_default_hyperparameters_match_mappo(self):
        self.assertEqual(asdict(IPPOConfig()), asdict(PPOConfig()))

    def test_two_agent_gae_terminal_and_truncation(self):
        buffer = RolloutBuffer(4, 1, (4, 3), 2, 2, 1, 1)
        values = [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0]]
        next_values = [[0.2, 2.0], [9.0, 90.0], [0.4, 4.0], [7.0, 70.0]]
        for index in range(4):
            sample = {
                key: torch.zeros(tensor.shape[1:], dtype=tensor.dtype)
                for key, tensor in buffer.data.items()
                if key not in ("advantage", "returns")
            }
            sample.update(
                value=torch.tensor(values[index]),
                next_value=torch.tensor(next_values[index]),
                reward=float(index + 1),
                terminated=index == 1,
                truncated=index == 3,
            )
            buffer.add(**sample)
        buffer.compute_advantages(gamma=0.9, gae_lambda=0.8, normalize=False)
        expected = torch.tensor([
            [2.376, 1.8], [1.8, 0.0], [10.188, 48.96], [9.9, 63.0],
        ])
        torch.testing.assert_close(buffer.data["advantage"], expected)
        returns = expected + torch.tensor(values)
        torch.testing.assert_close(buffer.data["returns"], returns)
        buffer.compute_advantages(gamma=0.9, gae_lambda=0.8, normalize=True)
        torch.testing.assert_close(buffer.data["returns"], returns)
        torch.testing.assert_close(
            buffer.data["advantage"].mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0,
        )
        torch.testing.assert_close(
            buffer.data["advantage"].std(dim=0, unbiased=False), torch.ones(2),
        )

    def test_critics_are_independent_and_training_needs_no_private_info(self):
        env = PublicInfoEnvironment(small_environment())
        controller = Controller(env, small_config(), seed=2026, device="cpu")
        networks = [getattr(controller, name) for name in MODEL_NAMES]
        parameter_ids = [{id(p) for p in network.parameters()} for network in networks]
        for index, ids in enumerate(parameter_ids):
            for other in parameter_ids[index + 1:]:
                self.assertTrue(ids.isdisjoint(other))
        before = [
            [p.detach().clone() for p in network.parameters()]
            for network in networks
        ]
        metrics = controller.update()
        for network, previous in zip(networks, before):
            self.assertTrue(any(
                not torch.equal(old, new) for old, new in zip(previous, network.parameters())
            ))
        self.assertEqual(metrics["total_episodes"], 2)
        self.assertEqual(metrics["total_env_steps"], 6)
        self.assertNotIn("critic_vector", controller.rollout.buffer.data)
        for critic in controller.critics:
            self.assertEqual(critic.actor_dim, env.actor_vector_dim)
        for field in UPDATE_FIELDS:
            self.assertTrue(np.isfinite(metrics[field]), field)
        self.assertAlmostEqual(
            metrics["critic_loss"],
            0.5 * (metrics["satellite_critic_loss"] + metrics["haps_ris_critic_loss"]),
        )

    def test_each_actor_receives_its_own_advantage(self):
        controller = Controller(
            small_environment(), small_config(), seed=2027, device="cpu",
        )
        original_batches = controller.rollout.buffer.batches
        original_loss = controller._ppo_actor_loss
        received = []

        def batches(batch_size, device):
            for batch in original_batches(batch_size, device):
                batch["advantage"][:, 0] = 1.0
                batch["advantage"][:, 1] = -1.0
                yield batch

        def loss(new_log_prob, old_log_prob, advantage, entropy, entropy_coef):
            received.append(advantage.detach().clone())
            return original_loss(new_log_prob, old_log_prob, advantage, entropy, entropy_coef)

        with (
            patch.object(controller.rollout.buffer, "batches", side_effect=batches),
            patch.object(controller, "_ppo_actor_loss", side_effect=loss),
        ):
            controller.update()
        self.assertEqual(len(received), 2)
        torch.testing.assert_close(received[0], torch.ones(6))
        torch.testing.assert_close(received[1], -torch.ones(6))

    def test_rollout_replays_group_log_prob_and_satellite_condition(self):
        controller = Controller(
            small_environment(), small_config(), seed=2028, device="cpu",
        )
        controller.rollout.collect()
        data = controller.rollout.buffer.data
        with torch.no_grad():
            sat_log_prob, _ = controller.satellite_actor.evaluate(
                data["actor_vector"], data["risk_nodes"], data["satellite_pre_tanh"],
            )
            ris_log_prob, _ = controller.haps_ris_actor.evaluate(
                data["actor_vector"], data["risk_nodes"], data["satellite_condition"],
                data["haps_ris_pre_tanh"], data["ris_mode"],
            )
            condition = controller.haps_ris_actor.satellite_beam_condition(
                torch.tanh(data["satellite_pre_tanh"])
            )
        torch.testing.assert_close(sat_log_prob, data["old_satellite_log_prob"])
        torch.testing.assert_close(ris_log_prob, data["old_haps_ris_log_prob"])
        torch.testing.assert_close(condition, data["satellite_condition"])
        self.assertFalse(torch.equal(data["value"][:, 0], data["value"][:, 1]))

    def test_64_and_256_element_group_actions_and_power_constraints(self):
        cases = [
            ["--num_ris_elements", "64", "--num_ris_groups", "64",
             "--ris_group_rows", "1", "--ris_group_cols", "1"],
            ["--num_ris_elements", "256", "--num_ris_groups", "64",
             "--ris_group_rows", "2", "--ris_group_cols", "2"],
        ]
        for options in cases:
            with self.subTest(elements=options[1]):
                env = small_environment(*options)
                controller = Controller(env, small_config(), seed=2029, device="cpu")
                obs, _ = controller.rollout._pending_state
                with torch.no_grad():
                    sat, _, _ = controller.satellite_actor.sample(obs["vector"], obs["risk_nodes"])
                    base = env.get_ris_phase_reference(sat[0].numpy())
                    condition = controller.haps_ris_actor.satellite_beam_condition(sat)
                    residual, mode, _, _ = controller.haps_ris_actor.sample(
                        obs["vector"], obs["risk_nodes"], condition,
                    )
                action = controller.satellite_actor.to_env_action(sat[0])
                action.update(controller.haps_ris_actor.to_env_action(
                    residual[0], mode[0], base_phase=base,
                ))
                self.assertTrue(env.action_space.contains(action))
                mapping = env.uav_ris.element_to_group
                np.testing.assert_array_equal(action["ris_element_mode"], mode[0].numpy()[mapping])
                expected_phase = base + np.radians(30.0) * residual[0].numpy()[mapping]
                actual_phase = np.angle(action["ris_phase_real"] + 1j * action["ris_phase_imag"])
                np.testing.assert_allclose(
                    np.angle(np.exp(1j * (actual_phase - expected_phase))), 0, atol=2e-6,
                )
                _, reward, _, _, _ = env.step(action)
                self.assertTrue(np.isfinite(reward))
                self.assertLessEqual(env.ris_power, env.params.ris_max_power + 1e-10)
                self.assertLessEqual(
                    np.sum(np.abs(env.satellite.W) ** 2), env.params.sat_max_power + 1e-10,
                )

    def test_current_default_risk_graph_forward_on_cpu(self):
        env = _build_environment(get_system_params([]))
        controller = Controller(
            env, replace(IPPOConfig(), episodes_per_update=1),
            seed=2030, device="cpu",
        )
        obs, _ = controller.rollout._pending_state
        with torch.no_grad():
            values = controller.rollout._values(obs)
            sat, _, _ = controller.satellite_actor.sample(obs["vector"], obs["risk_nodes"])
            condition = controller.haps_ris_actor.satellite_beam_condition(sat)
            residual, mode, _, _ = controller.haps_ris_actor.sample(
                obs["vector"], obs["risk_nodes"], condition,
            )
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(tuple(residual.shape), (1, env.params.num_ris_groups))
        self.assertEqual(tuple(mode.shape), tuple(residual.shape))
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(controller.device.type, "cpu")

    def test_phase_only_freezes_satellite_and_modes(self):
        env = small_environment()
        config = small_config(
            phase_only_training=True, fixed_csi_seed=123, gamma=0.0, gae_lambda=0.0,
            phase_log_std_init=-2.0, haps_ris_actor_lr=3e-5,
        )
        controller = Controller(env, config, seed=2031, device="cpu")
        sat_before = [p.detach().clone() for p in controller.satellite_actor.parameters()]
        mode_before = [p.detach().clone() for p in controller.haps_ris_actor.mode_head.parameters()]
        metrics = controller.update()
        self.assertEqual(metrics["satellite_actor_loss"], 0.0)
        self.assertTrue(torch.all(controller.rollout.buffer.data["ris_mode"] == 0))
        for old, new in zip(sat_before, controller.satellite_actor.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        for old, new in zip(mode_before, controller.haps_ris_actor.mode_head.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_checkpoint_restores_all_four_networks_and_optimizers(self):
        controller = Controller(
            small_environment(), small_config(), seed=2032, device="cpu",
        )
        controller.update()
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            path = Path(directory) / "checkpoint.pt"
            _save_checkpoint(path, controller, controller.env.params)
            _, checkpoint, params, config = _load_resume_checkpoint(path)
            restored = Controller(_build_environment(params), config, seed=2033, device="cpu")
            _restore_controller(restored, checkpoint)
            for name in MODEL_NAMES:
                for old, new in zip(
                    getattr(controller, name).parameters(), getattr(restored, name).parameters(),
                ):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                old_state = getattr(controller, name + "_optimizer").state_dict()["state"]
                new_state = getattr(restored, name + "_optimizer").state_dict()["state"]
                self.assertEqual(old_state.keys(), new_state.keys())
                self.assertGreater(len(old_state), 0)
                for key in old_state:
                    for field in old_state[key]:
                        torch.testing.assert_close(
                            old_state[key][field], new_state[key][field], rtol=0, atol=0,
                        )
            self.assertEqual(restored.update_index, 1)
            result = restored.update()
            self.assertEqual(result["update"], 2)
            self.assertEqual(result["total_episodes"], 4)
            self.assertEqual(result["total_env_steps"], 12)

    def test_mappo_and_wrong_observation_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"algorithm": "MAPPO"}, path)
            with self.assertRaisesRegex(ValueError, "only IPPO"):
                _load_resume_checkpoint(path)
            torch.save({
                "algorithm": "IPPO",
                "action_schema": "satellite_conditioned_ris_group_residual_v3",
                "critic_observation_schema": "privileged",
            }, path)
            with self.assertRaisesRegex(ValueError, "public-observation"):
                _load_resume_checkpoint(path)

    def test_csv_columns_and_six_panel_plot_match_mappo(self):
        source = ast.parse((IPPO_ROOT.parent / "ppo" / "main.py").read_text(encoding="utf-8"))
        fields = {}
        for node in ast.walk(source):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("episode_fields", "update_fields"):
                        fields[target.id] = ast.literal_eval(node.value)
        self.assertEqual(EPISODE_FIELDS, fields["episode_fields"])
        self.assertEqual(UPDATE_FIELDS[:-2], fields["update_fields"])
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            plotter = LiveTrainingPlotter(
                output_path=Path(directory) / "training_curves.png", interactive=False,
            )
            plotter.on_episode({
                "episode": 1, "outage_rate": 0.5, "return": -1,
                "max_bob_capacity": 0.2, "max_eve_capacity": 0.1,
            })
            plotter.on_update({
                "total_episodes": 1, "satellite_actor_loss": 0.01,
                "haps_ris_actor_loss": 0.02, "critic_loss": 0.3,
                "satellite_critic_loss": 0.4, "haps_ris_critic_loss": 0.2,
            })
            self.assertEqual(len(plotter.figure.axes), 6)
            self.assertEqual(len(plotter.critic_axis.lines), 3)
            self.assertEqual(plotter.satellite_critic_losses, [0.4])
            self.assertEqual(plotter.haps_ris_critic_losses, [0.2])
            self.assertGreater(plotter.output_path.stat().st_size, 10000)
            plotter.finish()


if __name__ == "__main__":
    unittest.main()
