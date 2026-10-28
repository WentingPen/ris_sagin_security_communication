from collections.abc import Mapping

import torch

from algorithm.ppo.buffer import RolloutBuffer


class Rollout:
    def __init__(
        self,
        env,
        satellite_actor,
        haps_ris_actor,
        critic,
        episodes_per_update=8,
        initial_state=None,
        phase_only_training=False,
        fixed_csi_seed=None,
    ):
        self.env = env
        self.satellite_actor = satellite_actor
        self.haps_ris_actor = haps_ris_actor
        self.critic = critic
        self.episodes_per_update = int(episodes_per_update)
        if self.episodes_per_update <= 0:
            raise ValueError("episodes_per_update must be positive")
        self._pending_state = initial_state
        self.phase_only_training = bool(phase_only_training)
        self.fixed_csi_seed = (
            None if fixed_csi_seed is None else int(fixed_csi_seed)
        )
        if self.fixed_csi_seed is not None and self.fixed_csi_seed < 0:
            raise ValueError("fixed_csi_seed must be non-negative")

        self.buffer = RolloutBuffer(
            capacity=self.episodes_per_update * env.horizon,
            actor_dim=env.actor_vector_dim,
            critic_dim=env.critic_vector_dim,
            risk_shape=env.observation_space["risk_nodes"].shape,
            satellite_continuous_dim=satellite_actor.continuous_dim,
            satellite_condition_dim=haps_ris_actor.satellite_condition_dim,
            haps_ris_continuous_dim=haps_ris_actor.continuous_dim,
            ris_mode_dim=haps_ris_actor.mode_dim,
        )

    @staticmethod
    def _require_shape(value, expected_shape, name):
        actual_shape = tuple(torch.as_tensor(value).shape)
        expected_shape = tuple(expected_shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_shape}"
            )

    @staticmethod
    def _mean(value):
        return float(torch.as_tensor(value, dtype=torch.float64).mean().item())

    @staticmethod
    def _max(value):
        return float(torch.as_tensor(value, dtype=torch.float64).max().item())

    def _actor_inputs(self, observation):
        if not isinstance(observation, Mapping):
            raise TypeError("actor observation must be a mapping")
        if "vector" not in observation or "risk_nodes" not in observation:
            raise KeyError("actor observation must contain vector and risk_nodes")

        vector = observation["vector"]
        risk_nodes = observation["risk_nodes"]
        self._require_shape(
            vector, (self.env.actor_vector_dim,), "actor observation vector"
        )
        self._require_shape(
            risk_nodes,
            self.env.observation_space["risk_nodes"].shape,
            "actor risk_nodes",
        )
        return vector, risk_nodes

    def _critic_inputs(self, info):
        if not isinstance(info, Mapping) or "critic_observation" not in info:
            raise KeyError("info must contain critic_observation")
        observation = info["critic_observation"]
        if not isinstance(observation, Mapping):
            raise TypeError("critic_observation must be a mapping")
        if "vector" not in observation or "risk_nodes" not in observation:
            raise KeyError("critic_observation must contain vector and risk_nodes")

        vector = observation["vector"]
        risk_nodes = observation["risk_nodes"]
        self._require_shape(
            vector, (self.env.critic_vector_dim,), "critic observation vector"
        )
        self._require_shape(
            risk_nodes,
            self.env.critic_observation_space["risk_nodes"].shape,
            "critic risk_nodes",
        )
        return vector, risk_nodes

    @staticmethod
    def _require_shared_risk_nodes(actor_risk_nodes, critic_risk_nodes):
        actor_tensor = torch.as_tensor(
            actor_risk_nodes, dtype=torch.float32, device="cpu"
        )
        critic_tensor = torch.as_tensor(
            critic_risk_nodes, dtype=torch.float32, device="cpu"
        )
        if not torch.equal(actor_tensor, critic_tensor):
            raise ValueError(
                "actor and critic risk_nodes must be identical because the "
                "rollout buffer stores one shared risk tensor"
            )

    def _reset_episode(self):
        if self._pending_state is not None:
            state = self._pending_state
            self._pending_state = None
            return state
        if self.phase_only_training:
            return self.env.reset(seed=self.fixed_csi_seed)
        return self.env.reset()

    def _phase_only_satellite_action(self):
        satellite_continuous, base_phase = (
            self.env.get_phase_only_reference()
        )
        device = next(self.haps_ris_actor.parameters()).device
        satellite_continuous = torch.as_tensor(
            satellite_continuous,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        satellite_pre_tanh = torch.zeros_like(satellite_continuous)
        satellite_log_prob = torch.zeros(1, dtype=torch.float32, device=device)
        return (
            satellite_continuous,
            satellite_pre_tanh,
            satellite_log_prob,
            base_phase,
        )

    @torch.no_grad()
    def collect(self, gamma=0.99, gae_lambda=0.95, episode_callback=None):
        self.buffer.clear()
        self.satellite_actor.eval()
        self.haps_ris_actor.eval()
        self.critic.eval()
        episode_stats = []

        for _ in range(self.episodes_per_update):
            observation, info = self._reset_episode()
            episode_return = 0.0
            bob_capacity_sum = 0.0
            max_eve_capacity_sum = 0.0
            step_max_bob_capacity_sum = 0.0
            step_max_eve_capacity_sum = 0.0
            jammer_inr_sum = 0.0
            ris_repair_count = 0

            for step in range(self.env.horizon):
                actor_vector, actor_risk_nodes = self._actor_inputs(observation)
                critic_vector, critic_risk_nodes = self._critic_inputs(info)
                self._require_shared_risk_nodes(
                    actor_risk_nodes, critic_risk_nodes
                )

                base_phase = None
                if self.phase_only_training:
                    (
                        satellite_continuous,
                        satellite_pre_tanh,
                        satellite_log_prob,
                        base_phase,
                    ) = self._phase_only_satellite_action()
                else:
                    (
                        satellite_continuous,
                        satellite_pre_tanh,
                        satellite_log_prob,
                    ) = self.satellite_actor.sample(
                        actor_vector, actor_risk_nodes
                    )
                    base_phase = self.env.get_ris_phase_reference(
                        satellite_continuous[0].detach().cpu().numpy()
                    )
                satellite_condition = (
                    self.haps_ris_actor.satellite_beam_condition(
                        satellite_continuous
                    )
                )
                (
                    haps_ris_continuous,
                    ris_mode,
                    haps_ris_pre_tanh,
                    haps_ris_log_prob,
                ) = self.haps_ris_actor.sample(
                    actor_vector,
                    actor_risk_nodes,
                    satellite_condition,
                )
                self._require_shape(
                    satellite_continuous,
                    (1, self.satellite_actor.continuous_dim),
                    "sampled satellite action",
                )
                self._require_shape(
                    satellite_pre_tanh,
                    (1, self.satellite_actor.continuous_dim),
                    "sampled satellite pre_tanh action",
                )
                self._require_shape(
                    satellite_log_prob,
                    (1,),
                    "sampled satellite log_prob",
                )
                self._require_shape(
                    satellite_condition,
                    (1, self.haps_ris_actor.satellite_condition_dim),
                    "sampled satellite condition",
                )
                self._require_shape(
                    haps_ris_continuous,
                    (1, self.haps_ris_actor.continuous_dim),
                    "sampled HAPS-RIS action",
                )
                self._require_shape(
                    ris_mode,
                    (1, self.haps_ris_actor.mode_dim),
                    "sampled RIS mode",
                )
                self._require_shape(
                    haps_ris_pre_tanh,
                    (1, self.haps_ris_actor.continuous_dim),
                    "sampled HAPS-RIS pre_tanh action",
                )
                self._require_shape(
                    haps_ris_log_prob,
                    (1,),
                    "sampled HAPS-RIS log_prob",
                )

                value_batch = self.critic(critic_vector, critic_risk_nodes)
                self._require_shape(value_batch, (1,), "critic value")
                value = value_batch[0]
                action = self.satellite_actor.to_env_action(
                    satellite_continuous[0]
                )
                action.update(
                    self.haps_ris_actor.to_env_action(
                        haps_ris_continuous[0],
                        ris_mode[0],
                        base_phase=base_phase,
                    )
                )
                next_observation, reward, terminated, truncated, next_info = (
                    self.env.step(action)
                )
                done = bool(terminated or truncated)

                step_bob_capacity = self._max(next_info["bob_capacity"])
                step_max_eve_capacity = self._max(
                    next_info["max_eve_capacity"]
                )
                bob_capacity_sum += self._mean(next_info["bob_capacity"])
                max_eve_capacity_sum += self._mean(
                    next_info["max_eve_capacity"]
                )
                step_max_bob_capacity_sum += step_bob_capacity
                step_max_eve_capacity_sum += step_max_eve_capacity
                jammer_inr_sum += self._mean(next_info["bob_jammer_inr"])
                ris_repair_count += int(next_info["ris_action_repaired"])

                if terminated:
                    next_value = 0.0
                else:
                    next_critic_vector, next_critic_risk_nodes = (
                        self._critic_inputs(next_info)
                    )
                    next_value_batch = self.critic(
                        next_critic_vector, next_critic_risk_nodes
                    )
                    self._require_shape(
                        next_value_batch, (1,), "next critic value"
                    )
                    next_value = next_value_batch[0]

                self.buffer.add(
                    actor_vector=actor_vector,
                    risk_nodes=actor_risk_nodes,
                    critic_vector=critic_vector,
                    satellite_pre_tanh=satellite_pre_tanh[0],
                    satellite_condition=satellite_condition[0],
                    haps_ris_pre_tanh=haps_ris_pre_tanh[0],
                    ris_mode=ris_mode[0],
                    satellite_log_prob=satellite_log_prob[0],
                    haps_ris_log_prob=haps_ris_log_prob[0],
                    value=value,
                    reward=reward,
                    next_value=next_value,
                    terminated=terminated,
                    truncated=truncated,
                )

                episode_return += float(reward)
                observation, info = next_observation, next_info
                if done:
                    break

            episode_length = step + 1
            mean_step_max_bob_capacity = (
                step_max_bob_capacity_sum / episode_length
            )
            mean_step_max_eve_capacity = (
                step_max_eve_capacity_sum / episode_length
            )
            stats = {
                "return": episode_return,
                "mean_reward": episode_return / episode_length,
                "length": episode_length,
                "outage_rate": float(info["episode_outage_rate"]),
                "num_eves": 1 + int(info["num_passive_eves"]),
                "mean_raw_security_gap": (
                    mean_step_max_bob_capacity
                    - mean_step_max_eve_capacity
                ),
                "mean_bob_capacity": bob_capacity_sum / episode_length,
                "mean_max_eve_capacity": (
                    max_eve_capacity_sum / episode_length
                ),
                "max_bob_capacity": mean_step_max_bob_capacity,
                "max_eve_capacity": mean_step_max_eve_capacity,
                "mean_bob_jammer_inr": jammer_inr_sum / episode_length,
                "ris_repair_rate": ris_repair_count / episode_length,
            }
            episode_stats.append(stats)
            if episode_callback is not None:
                episode_callback(dict(stats))

        self.buffer.compute_advantages(
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize=True,
        )
        return episode_stats
