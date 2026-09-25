from dataclasses import dataclass
from collections.abc import Mapping
import math

import torch

from algorithm.ppo.actor import HybridActor
from algorithm.ppo.critic import PrivilegeCritic
from algorithm.ppo.Rollout import Rollout


@dataclass(frozen=True)
class PPOConfig:
    episodes_per_update: int = 8
    batch_size: int = 256
    ppo_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    risk_hidden_dim: int = 64
    risk_feature_dim: int = 64


class Controller:
    def __init__(self, env, config=None, seed=None, device=None):
        self.env = env
        self.config = PPOConfig() if config is None else config
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        base_seed = int(env.params.seed if seed is None else seed)
        torch.manual_seed(base_seed)
        initial_state = env.reset(seed=base_seed)
        observation, info = initial_state
        params = env.params
        risk_shape = observation["risk_nodes"].shape

        self.actor = HybridActor(
            actor_dim=env.actor_vector_dim,
            risk_shape=risk_shape,
            edge_index=observation["edge_index"],
            risk_grid_size=params.risk_grid_size,
            num_ris_group=params.num_ris_groups,
            hidden_dim=self.config.hidden_dim,
            risk_hidden_dim=self.config.risk_hidden_dim,
            risk_feature_dim=self.config.risk_feature_dim,
        ).to(self.device)
        self.critic = PrivilegeCritic(
            critic_dim=env.critic_vector_dim,
            risk_shape=risk_shape,
            edge_index=observation["edge_index"],
            risk_grid_size=params.risk_grid_size,
            hidden_dim=self.config.hidden_dim,
            risk_hidden_dim=self.config.risk_hidden_dim,
            risk_feature_dim=self.config.risk_feature_dim,
        ).to(self.device)

        self._validate_interfaces(observation, info)

        self.rollout = Rollout(
            env=env,
            actor=self.actor,
            critic=self.critic,
            episodes_per_update=self.config.episodes_per_update,
            initial_state=initial_state,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

        self.update_index = 0
        self.total_episodes = 0
        self.total_env_steps = 0

    @staticmethod
    def _require_shape(value, expected_shape, name):
        actual_shape = tuple(torch.as_tensor(value).shape)
        expected_shape = tuple(expected_shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_shape}"
            )

    def _validate_interfaces(self, observation, info):
        if self.actor.actor_dim != int(self.env.actor_vector_dim):
            raise ValueError("actor input dimension does not match the environment")
        if self.critic.critic_dim != int(self.env.critic_vector_dim):
            raise ValueError("critic input dimension does not match the environment")

        if not isinstance(observation, Mapping):
            raise TypeError("environment observation must be a mapping")
        if "vector" not in observation or "risk_nodes" not in observation:
            raise KeyError("environment observation must contain vector and risk_nodes")
        self._require_shape(
            observation["vector"],
            (self.env.actor_vector_dim,),
            "actor observation vector",
        )
        self._require_shape(
            observation["risk_nodes"],
            self.env.observation_space["risk_nodes"].shape,
            "actor risk_nodes",
        )

        if not isinstance(info, Mapping) or "critic_observation" not in info:
            raise KeyError("reset info must contain critic_observation")
        critic_observation = info["critic_observation"]
        if not isinstance(critic_observation, Mapping):
            raise TypeError("critic_observation must be a mapping")
        if "vector" not in critic_observation or "risk_nodes" not in critic_observation:
            raise KeyError("critic_observation must contain vector and risk_nodes")
        self._require_shape(
            critic_observation["vector"],
            (self.env.critic_vector_dim,),
            "critic observation vector",
        )
        self._require_shape(
            critic_observation["risk_nodes"],
            self.env.critic_observation_space["risk_nodes"].shape,
            "critic risk_nodes",
        )

        actor_risk = torch.as_tensor(
            observation["risk_nodes"], dtype=torch.float32, device="cpu"
        )
        critic_risk = torch.as_tensor(
            critic_observation["risk_nodes"], dtype=torch.float32, device="cpu"
        )
        if not torch.equal(actor_risk, critic_risk):
            raise ValueError(
                "actor and critic risk_nodes must be identical because the "
                "rollout buffer stores one shared risk tensor"
            )

        expected_action_shapes = dict(self.actor.continuous_action_shapes)
        expected_action_shapes["ris_group_mode"] = (self.actor.G,)
        action_spaces = getattr(self.env.action_space, "spaces", None)
        if action_spaces is None:
            raise TypeError("environment action_space must expose named spaces")
        if set(action_spaces) != set(expected_action_shapes):
            raise ValueError(
                "environment and actor action keys differ: "
                f"environment={sorted(action_spaces)}, "
                f"actor={sorted(expected_action_shapes)}"
            )
        for name, expected_shape in expected_action_shapes.items():
            actual_shape = tuple(action_spaces[name].shape)
            if actual_shape != tuple(expected_shape):
                raise ValueError(
                    f"action {name} must have shape {tuple(expected_shape)}, "
                    f"got {actual_shape}"
                )

        expected_continuous_dim = sum(
            math.prod(shape)
            for shape in self.actor.continuous_action_shapes.values()
        )
        if self.actor.continuous_dim != expected_continuous_dim:
            raise ValueError("actor continuous action layout is inconsistent")

        with torch.no_grad():
            continuous, mode, pre_tanh, log_prob = self.actor.sample(
                observation["vector"],
                observation["risk_nodes"],
                deterministic=True,
            )
            self._require_shape(
                continuous,
                (1, self.actor.continuous_dim),
                "actor continuous output",
            )
            self._require_shape(mode, (1, self.actor.G), "actor mode output")
            self._require_shape(
                pre_tanh,
                (1, self.actor.continuous_dim),
                "actor pre_tanh output",
            )
            self._require_shape(log_prob, (1,), "actor log_prob output")

            action = self.actor.to_env_action(continuous[0], mode[0])
            if not self.env.action_space.contains(action):
                raise ValueError("actor action does not satisfy environment action_space")

            value = self.critic(
                critic_observation["vector"],
                critic_observation["risk_nodes"],
            )
            self._require_shape(value, (1,), "critic output")

    def update(self, episode_callback=None):
        policy_version = self.update_index
        episode_events = []

        def handle_episode(stats):
            self.total_episodes += 1
            self.total_env_steps += stats["length"]
            event = {
                **stats,
                "episode": self.total_episodes,
                "env_steps": self.total_env_steps,
                "policy_version": policy_version,
            }
            episode_events.append(event)

            if episode_callback is not None:
                episode_callback(dict(event))

        self.rollout.collect(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            episode_callback=handle_episode,
        )
        self.actor.train()
        self.critic.train()

        totals = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        sample_count = 0

        for _ in range(self.config.ppo_epochs):
            for batch in self.rollout.buffer.batches(
                self.config.batch_size, self.device
            ):
                new_log_prob, entropy = self.actor.evaluate(
                    batch["actor_vector"],
                    batch["risk_nodes"],
                    batch["pre_tanh"],
                    batch["mode"],
                )
                log_ratio = new_log_prob - batch["old_log_prob"]
                ratio = log_ratio.exp()
                surrogate_1 = ratio * batch["advantage"]
                surrogate_2 = ratio.clamp(
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                ) * batch["advantage"]
                policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
                mean_entropy = entropy.mean() / (
                    self.actor.continuous_dim + self.actor.G
                )
                actor_loss = (
                    policy_loss - self.config.entropy_coef * mean_entropy
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()

                value = self.critic(
                    batch["critic_vector"], batch["risk_nodes"]
                )
                critic_loss = 0.5 * (
                    value - batch["returns"]
                ).square().mean()

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optimizer.step()

                count = batch["advantage"].numel()
                sample_count += count
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_coef
                    ).float().mean()

                totals["actor_loss"] += actor_loss.item() * count
                totals["critic_loss"] += critic_loss.item() * count
                totals["entropy"] += mean_entropy.item() * count
                totals["approx_kl"] += approx_kl.item() * count
                totals["clip_fraction"] += clip_fraction.item() * count

        self.update_index += 1
        result = {
            name: value / sample_count for name, value in totals.items()
        }
        result.update({
            "update": self.update_index,
            "from_policy_version": policy_version,
            "to_policy_version": self.update_index,
            "total_episodes": self.total_episodes,
            "total_env_steps": self.total_env_steps,
            "mean_episode_return": sum(
                item["return"] for item in episode_events
            ) / len(episode_events),
            "mean_outage_rate": sum(
                item["outage_rate"] for item in episode_events
            ) / len(episode_events),
            "episodes": episode_events,
        })
        return result
