import unittest

import numpy as np
import torch

from algorithm.ppo.Controller import Controller, PPOConfig
from algorithm.ppo.main import _build_environment, _parse_training_args
from system_model.params import get_system_params


torch.set_num_threads(1)


def _small_params(seed=123):
    return get_system_params([
        "--seed",
        str(seed),
        "--risk_grid_size",
        "2",
        "--max_episode_steps",
        "3",
    ])


def _phase_only_config(seed=123):
    return PPOConfig(
        episodes_per_update=1,
        batch_size=3,
        ppo_epochs=1,
        gamma=0.0,
        gae_lambda=0.0,
        clip_coef=0.1,
        entropy_coef=1e-4,
        entropy_coef_final=0.0,
        entropy_anneal_updates=2,
        actor_lr=1e-5,
        haps_ris_actor_lr=3e-5,
        critic_lr=1e-4,
        hidden_dim=32,
        risk_hidden_dim=8,
        risk_feature_dim=8,
        phase_only_training=True,
        fixed_csi_seed=seed,
        phase_log_std_init=-2.0,
        phase_residual_max_deg=30.0,
    )


def _reference_action(env):
    satellite_continuous, base_phase = env.get_phase_only_reference()
    num_antennas = env.satellite.num_sat_antennas
    return {
        "sat_beam_real": satellite_continuous[:num_antennas],
        "sat_beam_imag": satellite_continuous[num_antennas:],
        "ris_element_mode": np.zeros(env.uav_ris.num_elements, dtype=np.int8),
        "ris_phase_real": np.cos(base_phase).astype(np.float32),
        "ris_phase_imag": np.sin(base_phase).astype(np.float32),
    }


class PhaseOnlyTrainingTests(unittest.TestCase):
    def test_phase_only_cli_switch_and_seed(self):
        training_args, system_argv = _parse_training_args([
            "--phase_only_training",
            "--phase_only_csi_seed",
            "77",
            "--seed",
            "123",
        ])
        self.assertTrue(training_args.phase_only_training)
        self.assertEqual(training_args.phase_only_csi_seed, 77)
        self.assertEqual(system_argv, ["--seed", "123"])

    def test_reference_is_mrt_and_coherently_aligned(self):
        params = _small_params()
        env = _build_environment(params)
        env.reset(seed=params.seed)

        satellite_continuous, base_phase = env.get_phase_only_reference()
        num_antennas = params.num_sat_antennas
        unit_beam = (
            satellite_continuous[:num_antennas]
            + 1j * satellite_continuous[num_antennas:]
        )
        direct_channel = env.bobs[0].direct_channel
        expected = np.conj(direct_channel) / np.linalg.norm(direct_channel)
        np.testing.assert_allclose(unit_beam, expected, rtol=1e-6, atol=1e-7)

        beam = np.sqrt(params.sat_max_power) * unit_beam
        direct_signal = direct_channel @ beam
        reflected_terms = env.bobs[0].ris_channel * (env.h_sat_ris @ beam)
        nonzero = np.abs(reflected_terms) > 1e-30
        aligned = (
            reflected_terms[nonzero]
            * np.exp(1j * base_phase[nonzero])
            * np.conj(direct_signal)
        )
        self.assertLess(np.max(np.abs(np.angle(aligned))), 1e-7)

    def test_fixed_seed_replays_a_varying_slot_trajectory(self):
        params = _small_params()
        env = _build_environment(params)

        trajectories = []
        for _ in range(2):
            env.reset(seed=params.seed)
            trajectory = []
            for _ in range(2):
                trajectory.append((
                    env.h_sat_ris.copy(),
                    env.bobs[0].ris_channel,
                    env.active_ris_rx_channel.copy(),
                    env.eve_positions.copy(),
                ))
                env.step(_reference_action(env))
            trajectories.append(trajectory)

        for first, second in zip(trajectories[0], trajectories[1]):
            for first_value, second_value in zip(first, second):
                np.testing.assert_array_equal(first_value, second_value)
        self.assertFalse(
            np.array_equal(
                trajectories[0][0][1],
                trajectories[0][1][1],
            )
        )

    def test_phase_only_policy_probability_excludes_mode(self):
        params = _small_params()
        env = _build_environment(params)
        controller = Controller(
            env,
            config=_phase_only_config(params.seed),
            seed=params.seed,
            device="cpu",
        )
        observation, _ = controller.rollout._pending_state
        satellite_continuous, _ = env.get_phase_only_reference()
        satellite_condition = (
            controller.haps_ris_actor.satellite_beam_condition(
                satellite_continuous
            )
        )
        continuous, mode, pre_tanh, sampled_log_prob = (
            controller.haps_ris_actor.sample(
                observation["vector"],
                observation["risk_nodes"],
                satellite_condition,
            )
        )
        evaluated_log_prob, _ = controller.haps_ris_actor.evaluate(
            observation["vector"],
            observation["risk_nodes"],
            satellite_condition,
            pre_tanh,
            mode,
        )
        self.assertEqual(tuple(continuous.shape), (1, params.num_ris_groups))
        self.assertEqual(tuple(mode.shape), (1, params.num_ris_groups))
        self.assertTrue(torch.equal(mode, torch.zeros_like(mode)))
        torch.testing.assert_close(sampled_log_prob, evaluated_log_prob)
        torch.testing.assert_close(
            controller.haps_ris_actor.phase_log_std,
            torch.full_like(controller.haps_ris_actor.phase_log_std, -2.0),
        )

        _, base_phase = env.get_phase_only_reference()
        action = controller.haps_ris_actor.to_env_action(
            continuous[0], mode[0], base_phase=base_phase
        )
        for value in action.values():
            self.assertEqual(value.shape, (params.num_ris_elements,))
        applied_phase = np.angle(
            action["ris_phase_real"] + 1j * action["ris_phase_imag"]
        )
        residual = np.angle(np.exp(1j * (applied_phase - base_phase)))
        self.assertLessEqual(
            np.max(np.abs(residual)),
            np.radians(30.0) + 1e-7,
        )
        self.assertTrue(np.all(action["ris_element_mode"] == 0))

    def test_one_update_keeps_satellite_and_mode_frozen(self):
        params = _small_params()
        env = _build_environment(params)
        controller = Controller(
            env,
            config=_phase_only_config(params.seed),
            seed=params.seed,
            device="cpu",
        )
        satellite_before = {
            name: value.detach().clone()
            for name, value in controller.satellite_actor.state_dict().items()
        }
        mode_before = {
            name: value.detach().clone()
            for name, value in controller.haps_ris_actor.mode_head.state_dict().items()
        }

        metrics = controller.update()

        self.assertEqual(
            tuple(controller.rollout.buffer.haps_ris_pre_tanh.shape),
            (3, params.num_ris_groups),
        )
        self.assertEqual(
            tuple(controller.rollout.buffer.ris_mode.shape),
            (3, params.num_ris_groups),
        )
        for name, value in controller.satellite_actor.state_dict().items():
            torch.testing.assert_close(value, satellite_before[name], rtol=0, atol=0)
        for name, value in controller.haps_ris_actor.mode_head.state_dict().items():
            torch.testing.assert_close(value, mode_before[name], rtol=0, atol=0)
        self.assertEqual(metrics["satellite_actor_loss"], 0.0)
        self.assertEqual(metrics["satellite_entropy"], 0.0)
        self.assertEqual(metrics["satellite_approx_kl"], 0.0)
        self.assertTrue(np.isfinite(metrics["haps_ris_actor_loss"]))
        torch.testing.assert_close(
            controller.rollout.buffer.returns[: controller.rollout.buffer.pos],
            controller.rollout.buffer.reward[: controller.rollout.buffer.pos],
        )
        self.assertTrue(np.all(controller.env.uav_ris.element_mode == 0))
        self.assertEqual(controller.env.ris_power, 0.0)
        expected_mrt = (
            np.sqrt(params.sat_max_power)
            * np.conj(controller.env.bobs[0].direct_channel)
            / np.linalg.norm(controller.env.bobs[0].direct_channel)
        )
        np.testing.assert_allclose(
            controller.env.satellite.W[:, 0],
            expected_mrt,
            rtol=1e-6,
            atol=1e-12,
        )

    def test_default_joint_mode_still_runs(self):
        params = _small_params(seed=321)
        env = _build_environment(params)
        config = PPOConfig(
            episodes_per_update=1,
            batch_size=3,
            ppo_epochs=1,
            entropy_anneal_updates=2,
            hidden_dim=32,
            risk_hidden_dim=8,
            risk_feature_dim=8,
        )
        controller = Controller(env, config=config, seed=params.seed, device="cpu")
        metrics = controller.update()
        self.assertTrue(np.isfinite(metrics["satellite_actor_loss"]))
        self.assertTrue(np.isfinite(metrics["haps_ris_actor_loss"]))
        self.assertEqual(controller.rollout.buffer.pos, 3)
        self.assertEqual(
            tuple(controller.rollout.buffer.haps_ris_pre_tanh.shape),
            (3, params.num_ris_groups),
        )
        self.assertEqual(
            tuple(controller.rollout.buffer.ris_mode.shape),
            (3, params.num_ris_groups),
        )


if __name__ == "__main__":
    unittest.main()
