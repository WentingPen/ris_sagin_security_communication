import torch
from collections.abc import Mapping

from algorithm.ppo.buffer import RolloutBuffer


class Rollout:
    def __init__(self, env, actor, critic, episodes_per_update=8, initial_state=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.episodes_per_update = int(episodes_per_update)
        if self.episodes_per_update <= 0:
            raise ValueError("episodes_per_update must be positive")
        self._pending_state = initial_state

        self.buffer = RolloutBuffer(
            capacity=self.episodes_per_update * env.horizon,
            actor_dim=env.actor_vector_dim,
            critic_dim=env.critic_vector_dim,
            risk_shape=env.observation_space["risk_nodes"].shape,
            continuous_dim=actor.continuous_dim,
            mode_dim=actor.G,
        )

    @staticmethod
    def _require_shape(value, expected_shape, name):
        actual_shape = tuple(torch.as_tensor(value).shape)
        expected_shape = tuple(expected_shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_shape}"
            )

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

        # Do not pass the same seed again: each reset must sample a new HCPP realization.
        return self.env.reset()

    @torch.no_grad()
    def collect(self, gamma=0.99, gae_lambda=0.95, episode_callback=None):
        # Clear once per PPO update, not once per episode.
        self.buffer.clear()
        self.actor.eval()
        self.critic.eval()
        episode_stats = []

        for _ in range(self.episodes_per_update):
            observation, info = self._reset_episode()
            episode_return = 0.0
            raw_gap_sum = 0.0
            bob_capacity_sum = 0.0
            max_eve_capacity_sum = 0.0
            jammer_inr_sum = 0.0
            ris_repair_count = 0

            for step in range(self.env.horizon):
                actor_vector, actor_risk_nodes = self._actor_inputs(observation)
                critic_vector, critic_risk_nodes = self._critic_inputs(info)
                self._require_shared_risk_nodes(
                    actor_risk_nodes, critic_risk_nodes
                )

                continuous, mode, pre_tanh, log_prob = self.actor.sample(
                    actor_vector, actor_risk_nodes
                )
                self._require_shape(
                    continuous,
                    (1, self.actor.continuous_dim),
                    "sampled continuous action",
                )
                self._require_shape(
                    mode, (1, self.actor.G), "sampled RIS mode"
                )
                self._require_shape(
                    pre_tanh,
                    (1, self.actor.continuous_dim),
                    "sampled pre_tanh action",
                )
                self._require_shape(log_prob, (1,), "sampled log_prob")

                value_batch = self.critic(critic_vector, critic_risk_nodes)
                self._require_shape(value_batch, (1,), "critic value")
                value = value_batch[0]
                action = self.actor.to_env_action(continuous[0], mode[0])
                next_observation, reward, terminated, truncated, next_info = (
                    self.env.step(action)
                )
                done = bool(terminated or truncated)
                raw_gap_sum += float(next_info["raw_security_gap"].min())
                bob_capacity_sum += float(next_info["bob_capacity"].mean())
                max_eve_capacity_sum += float(
                    next_info["max_eve_capacity"].mean()
                )
                jammer_inr_sum += float(next_info["bob_jammer_inr"].mean())
                ris_repair_count += int(next_info["ris_action_repaired"])

                # The episode boundary ends this HCPP realization, so do not bootstrap.
                if done:
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
                    pre_tanh=pre_tanh[0],
                    mode=mode[0],
                    log_prob=log_prob[0],
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
            stats = {
                "return": episode_return,
                "mean_reward": episode_return / episode_length,
                "length": episode_length,
                "outage_rate": float(info["episode_outage_rate"]),
                "num_eves": 1 + int(info["num_passive_eves"]),
                "mean_raw_security_gap": raw_gap_sum / episode_length,
                "mean_bob_capacity": bob_capacity_sum / episode_length,
                "mean_max_eve_capacity": max_eve_capacity_sum / episode_length,
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
