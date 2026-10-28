import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Bernoulli, Normal

from algorithm.ppo.Controller import Controller, PPOConfig
from algorithm.ppo.actor import HAPSRISActor
from algorithm.ppo.main import _build_environment
from system_model.params import get_system_params, thermal_noise_power


torch.set_num_threads(1)


def _small_setup(phase_only_training=False):
    params = get_system_params([
        "--seed", "123",
        "--risk_grid_size", "2",
        "--max_episode_steps", "3",
    ])
    env = _build_environment(params)
    observation, _ = env.reset(seed=params.seed)
    torch.manual_seed(params.seed)
    actor = HAPSRISActor(
        actor_dim=env.actor_vector_dim,
        risk_shape=observation["risk_nodes"].shape,
        edge_index=observation["edge_index"],
        risk_grid_size=params.risk_grid_size,
        num_ris_elements=params.num_ris_elements,
        num_ris_groups=params.num_ris_groups,
        element_to_group=env.uav_ris.element_to_group,
        num_sat_antennas=params.num_sat_antennas,
        hidden_dim=32,
        risk_hidden_dim=8,
        risk_feature_dim=8,
        phase_only_training=phase_only_training,
        phase_residual_max_deg=30.0,
    )
    return env, observation, actor


class GroupedRISActionTests(unittest.TestCase):
    def test_defaults_and_derived_thermal_noise(self):
        params = get_system_params([])
        self.assertEqual(params.num_ris_elements, 256)
        self.assertEqual(params.num_ris_groups, 64)
        self.assertEqual(params.ris_array_rows, 16)
        self.assertEqual((params.ris_group_rows, params.ris_group_cols), (2, 2))
        self.assertEqual(params.sat_max_power, 25.0)
        self.assertEqual(params.ris_max_amplitude, 8.0)
        self.assertEqual(params.bob_noise_figure_db, 5.0)
        self.assertEqual(params.eve_noise_figure_db, 5.0)
        self.assertEqual(params.secrecy_rate_threshold, [0.1])
        for receiver in ("bob", "eve", "ris"):
            self.assertEqual(
                getattr(params, receiver + "_noise_power"),
                thermal_noise_power(
                    params.bandwidth_hz,
                    getattr(params, receiver + "_noise_figure_db"),
                    params.noise_temperature_k,
                ),
            )
        wider = get_system_params(["--bandwidth_hz", str(2 * params.bandwidth_hz)])
        self.assertEqual(wider.bob_noise_power, 2 * params.bob_noise_power)
        self.assertEqual(wider.eve_noise_power, 2 * params.eve_noise_power)
        self.assertEqual(wider.ris_noise_power, 2 * params.ris_noise_power)

    def test_group_map_tiles_physical_two_by_two_blocks(self):
        env, _, _ = _small_setup()
        mapping = env.uav_ris.element_to_group
        expected = np.arange(64).reshape(8, 8).repeat(2, axis=0).repeat(2, axis=1)
        np.testing.assert_array_equal(mapping.reshape(16, 16), expected)
        np.testing.assert_array_equal(np.flatnonzero(mapping == 0), [0, 1, 16, 17])
        np.testing.assert_array_equal(np.flatnonzero(mapping == 8), [32, 33, 48, 49])
        np.testing.assert_array_equal(np.bincount(mapping), np.full(64, 4))

    def test_group_modes_and_residuals_expand_over_distinct_element_bases(self):
        env, _, actor = _small_setup()
        groups, elements = env.params.num_ris_groups, env.params.num_ris_elements
        mapping = env.uav_ris.element_to_group
        continuous = torch.linspace(-0.9, 0.9, groups)
        mode = (torch.arange(groups) % 2).float()
        base_phase = np.linspace(-2.7, 2.7, elements)

        action = actor.to_env_action(continuous, mode, base_phase=base_phase)

        self.assertEqual(actor.continuous_dim, groups)
        self.assertEqual(actor.mode_dim, groups)
        self.assertEqual(actor.phase_head.out_features, groups)
        self.assertEqual(actor.mode_head.out_features, groups)
        self.assertEqual(tuple(actor.phase_log_std.shape), (groups,))
        self.assertEqual(set(actor.env_action_shapes), set(action))
        for key, value in action.items():
            self.assertEqual(value.shape, (elements,))
            self.assertEqual(actor.env_action_shapes[key], (elements,))
        np.testing.assert_array_equal(action["ris_element_mode"], mode.numpy()[mapping])
        unit_phase = action["ris_phase_real"] + 1j * action["ris_phase_imag"]
        expected_residual = np.radians(30.0) * continuous.numpy()[mapping]
        np.testing.assert_allclose(
            unit_phase,
            np.exp(1j * (base_phase + expected_residual)),
            rtol=1e-6,
            atol=1e-7,
        )
        residual = np.angle(unit_phase * np.exp(-1j * base_phase))
        first_group = np.flatnonzero(mapping == 0)
        np.testing.assert_allclose(
            residual[first_group], np.full(4, expected_residual[0]), atol=1e-7
        )
        self.assertFalse(np.isclose(unit_phase[first_group[0]], unit_phase[first_group[1]]))

    def test_base_phase_is_required_in_both_training_modes(self):
        for phase_only_training in (False, True):
            with self.subTest(phase_only_training=phase_only_training):
                env, _, actor = _small_setup(phase_only_training)
                group_zeros = torch.zeros(env.params.num_ris_groups)
                with self.assertRaises((TypeError, ValueError)):
                    actor.to_env_action(group_zeros, group_zeros)

    def test_sample_and_evaluate_count_each_group_probability_once(self):
        for phase_only_training in (False, True):
            with self.subTest(phase_only_training=phase_only_training):
                env, observation, actor = _small_setup(phase_only_training)
                satellite_continuous, _ = env.get_phase_only_reference()
                arguments = (
                    observation["vector"],
                    observation["risk_nodes"],
                    satellite_continuous,
                )
                continuous, mode, pre_tanh, sampled_log_prob = actor.sample(*arguments)
                evaluated_log_prob, _ = actor.evaluate(*arguments, pre_tanh, mode)
                mode_distribution, phase_distribution = actor._distributions(*arguments)
                z = pre_tanh.double()
                manual_phase = (
                    Normal(
                        phase_distribution.mean.double(),
                        phase_distribution.stddev.double(),
                    ).log_prob(z)
                    - torch.log(1.0 - torch.tanh(z).square())
                ).sum(dim=-1)
                manual_log_prob = manual_phase
                if not phase_only_training:
                    manual_log_prob = manual_log_prob + Bernoulli(
                        logits=mode_distribution.logits.double()
                    ).log_prob(mode.double()).sum(dim=-1)
                else:
                    self.assertTrue(torch.equal(mode, torch.zeros_like(mode)))
                self.assertEqual(tuple(continuous.shape), (1, env.params.num_ris_groups))
                self.assertEqual(tuple(mode.shape), (1, env.params.num_ris_groups))
                torch.testing.assert_close(sampled_log_prob, evaluated_log_prob)
                torch.testing.assert_close(
                    sampled_log_prob.double(), manual_log_prob, rtol=1e-5, atol=1e-4
                )
                self.assertFalse(torch.allclose(sampled_log_prob.double(), 4 * manual_log_prob))

    def test_joint_zero_residual_is_coherent_for_actual_projected_satellite_beam(self):
        env, observation, actor = _small_setup()
        rng = np.random.default_rng(51)
        for scale in (0.03, 0.9):
            with self.subTest(scale=scale):
                satellite_continuous = rng.uniform(
                    -scale, scale, 2 * env.params.num_sat_antennas
                ).astype(np.float32)
                beam_before = env.satellite.W.copy()
                base_phase = env.get_ris_phase_reference(satellite_continuous)
                np.testing.assert_array_equal(env.satellite.W, beam_before)
                continuous, mode, _, _ = actor.sample(
                    observation["vector"],
                    observation["risk_nodes"],
                    satellite_continuous,
                    deterministic=True,
                )
                self.assertTrue(torch.equal(continuous, torch.zeros_like(continuous)))
                action = actor.to_env_action(continuous[0], mode[0], base_phase=base_phase)
                antennas = env.params.num_sat_antennas
                action.update(
                    sat_beam_real=satellite_continuous[:antennas],
                    sat_beam_imag=satellite_continuous[antennas:],
                )
                env._apply_action(action)
                expected_beam = np.sqrt(env.params.sat_max_power) * (
                    satellite_continuous[:antennas].astype(np.float64)
                    + 1j * satellite_continuous[antennas:].astype(np.float64)
                ) / max(np.linalg.norm(satellite_continuous.astype(np.float64)), 1.0)
                np.testing.assert_allclose(env.satellite.W[:, 0], expected_beam, atol=1e-12)
                direct_signal = env.bobs[0].direct_channel @ env.satellite.W[:, 0]
                terms = (
                    env.bobs[0].ris_channel
                    * (env.h_sat_ris @ env.satellite.W[:, 0])
                    * env.uav_ris.ris_coeff
                )
                nonzero = np.abs(terms) > 1e-30
                self.assertLess(
                    np.max(np.abs(np.angle(terms[nonzero] * np.conj(direct_signal)))),
                    1e-6,
                )
                efficiency = abs(terms.sum()) ** 2 / np.sum(np.abs(terms)) ** 2
                self.assertAlmostEqual(efficiency, 1.0, places=12)

    def test_joint_rollout_applies_current_beam_base_and_sampled_group_residuals(self):
        env, _, _ = _small_setup()
        controller = Controller(
            env,
            config=PPOConfig(
                episodes_per_update=1,
                hidden_dim=32,
                risk_hidden_dim=8,
                risk_feature_dim=8,
            ),
            seed=env.params.seed,
            device="cpu",
        )
        actor = controller.haps_ris_actor
        original_sample = actor.sample
        original_step = env.step
        sampled = []
        checked = []

        def sample(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            sampled.append((result[0].detach().numpy()[0], result[1].detach().numpy()[0]))
            return result

        def step(action):
            continuous, mode = sampled[-1]
            current_satellite_action = np.r_[
                action["sat_beam_real"], action["sat_beam_imag"]
            ]
            base = env.get_ris_phase_reference(current_satellite_action)
            mapping = env.uav_ris.element_to_group
            actual_phase = action["ris_phase_real"] + 1j * action["ris_phase_imag"]
            expected_phase = np.exp(
                1j * (base + actor.phase_residual_max * continuous[mapping])
            )
            np.testing.assert_allclose(actual_phase, expected_phase, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(action["ris_element_mode"], mode[mapping])
            checked.append(True)
            return original_step(action)

        with patch.object(actor, "sample", side_effect=sample), patch.object(
            env, "step", side_effect=step
        ):
            controller.rollout.collect()
        self.assertEqual(len(checked), env.horizon)


if __name__ == "__main__":
    unittest.main()
