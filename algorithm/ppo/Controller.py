from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch

from algorithm.ppo.Rollout import Rollout
from algorithm.ppo.actor import HAPSRISActor, SatelliteActor
from algorithm.ppo.critic import PrivilegeCritic


@dataclass(frozen=True)
class PPOConfig:
    episodes_per_update: int = 8
    batch_size: int = 256
    ppo_epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    entropy_coef: float = 1e-3
    entropy_coef_final: float = 1e-4
    entropy_anneal_updates: int = 1000
    satellite_target_kl: float = 0.01
    haps_ris_target_kl: float = 0.02
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    risk_hidden_dim: int = 64
    risk_feature_dim: int = 64


class Controller:
    def __init__(self, env, config=None, seed=None, device=None):
        self.env = env
        self.config = PPOConfig() if config is None else config
        self._validate_config()
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        base_seed = int(env.params.seed if seed is None else seed)
        torch.manual_seed(base_seed)
        initial_state = env.reset(seed=base_seed)
        observation, info = initial_state
        params = env.params
        risk_shape = observation["risk_nodes"].shape

        common_actor_arguments = {
            "actor_dim": env.actor_vector_dim,
            "risk_shape": risk_shape,
            "edge_index": observation["edge_index"],
            "risk_grid_size": params.risk_grid_size,
            "hidden_dim": self.config.hidden_dim,
            "risk_hidden_dim": self.config.risk_hidden_dim,
            "risk_feature_dim": self.config.risk_feature_dim,
        }
        self.satellite_actor = SatelliteActor(
            **common_actor_arguments,
            num_sat_antennas=params.num_sat_antennas,
        ).to(self.device)
        self.haps_ris_actor = HAPSRISActor(
            **common_actor_arguments,
            num_ris_elements=params.num_ris_elements,
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
            satellite_actor=self.satellite_actor,
            haps_ris_actor=self.haps_ris_actor,
            critic=self.critic,
            episodes_per_update=self.config.episodes_per_update,
            initial_state=initial_state,
        )
        self.satellite_actor_optimizer = torch.optim.Adam(
            self.satellite_actor.parameters(), lr=self.config.actor_lr
        )
        self.haps_ris_actor_optimizer = torch.optim.Adam(
            self.haps_ris_actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

        self.update_index = 0
        self.total_episodes = 0
        self.total_env_steps = 0

    def _validate_config(self):
        if self.config.entropy_coef < 0.0 or self.config.entropy_coef_final < 0.0:
            raise ValueError("entropy coefficients must be non-negative")
        if self.config.entropy_anneal_updates <= 0:
            raise ValueError("entropy_anneal_updates must be positive")
        if self.config.satellite_target_kl <= 0.0:
            raise ValueError("satellite_target_kl must be positive")
        if self.config.haps_ris_target_kl <= 0.0:
            raise ValueError("haps_ris_target_kl must be positive")

    def _current_entropy_coef(self):
        if self.config.entropy_anneal_updates == 1:
            return float(self.config.entropy_coef_final)
        progress = min(
            self.update_index / (self.config.entropy_anneal_updates - 1),
            1.0,
        )
        return float(
            self.config.entropy_coef
            + progress
            * (self.config.entropy_coef_final - self.config.entropy_coef)
        )

    @staticmethod
    def _require_shape(value, expected_shape, name):
        actual_shape = tuple(torch.as_tensor(value).shape)
        expected_shape = tuple(expected_shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_shape}"
            )

    def _validate_interfaces(self, observation, info):
        for name, actor in (
            ("satellite", self.satellite_actor),
            ("HAPS-RIS", self.haps_ris_actor),
        ):
            if actor.actor_dim != int(self.env.actor_vector_dim):
                raise ValueError(
                    f"{name} actor input dimension does not match the environment"
                )
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
            critic_observation["risk_nodes"],
            dtype=torch.float32,
            device="cpu",
        )
        if not torch.equal(actor_risk, critic_risk):
            raise ValueError(
                "actor and critic risk_nodes must be identical because the "
                "rollout buffer stores one shared risk tensor"
            )

        expected_action_shapes = {
            **self.satellite_actor.continuous_action_shapes,
            **self.haps_ris_actor.continuous_action_shapes,
            "ris_element_mode": (self.haps_ris_actor.mode_dim,),
        }
        action_spaces = getattr(self.env.action_space, "spaces", None)
        if action_spaces is None:
            raise TypeError("environment action_space must expose named spaces")
        if set(action_spaces) != set(expected_action_shapes):
            raise ValueError(
                "environment and actors use different action keys: "
                f"environment={sorted(action_spaces)}, "
                f"actors={sorted(expected_action_shapes)}"
            )
        for name, expected_shape in expected_action_shapes.items():
            actual_shape = tuple(action_spaces[name].shape)
            if actual_shape != tuple(expected_shape):
                raise ValueError(
                    f"action {name} must have shape {tuple(expected_shape)}, "
                    f"got {actual_shape}"
                )

        for name, actor in (
            ("satellite", self.satellite_actor),
            ("HAPS-RIS", self.haps_ris_actor),
        ):
            expected_continuous_dim = sum(
                math.prod(shape)
                for shape in actor.continuous_action_shapes.values()
            )
            if actor.continuous_dim != expected_continuous_dim:
                raise ValueError(
                    f"{name} actor continuous action layout is inconsistent"
                )

        with torch.no_grad():
            (
                satellite_continuous,
                satellite_pre_tanh,
                satellite_log_prob,
            ) = self.satellite_actor.sample(
                observation["vector"],
                observation["risk_nodes"],
                deterministic=True,
            )
            (
                haps_ris_continuous,
                ris_mode,
                haps_ris_pre_tanh,
                haps_ris_log_prob,
            ) = self.haps_ris_actor.sample(
                observation["vector"],
                observation["risk_nodes"],
                deterministic=True,
            )
            self._require_shape(
                satellite_continuous,
                (1, self.satellite_actor.continuous_dim),
                "satellite actor continuous output",
            )
            self._require_shape(
                satellite_pre_tanh,
                (1, self.satellite_actor.continuous_dim),
                "satellite actor pre_tanh output",
            )
            self._require_shape(
                satellite_log_prob, (1,), "satellite actor log_prob output"
            )
            self._require_shape(
                haps_ris_continuous,
                (1, self.haps_ris_actor.continuous_dim),
                "HAPS-RIS actor continuous output",
            )
            self._require_shape(
                ris_mode,
                (1, self.haps_ris_actor.mode_dim),
                "HAPS-RIS actor mode output",
            )
            self._require_shape(
                haps_ris_pre_tanh,
                (1, self.haps_ris_actor.continuous_dim),
                "HAPS-RIS actor pre_tanh output",
            )
            self._require_shape(
                haps_ris_log_prob, (1,), "HAPS-RIS actor log_prob output"
            )

            action = self.satellite_actor.to_env_action(
                satellite_continuous[0]
            )
            action.update(
                self.haps_ris_actor.to_env_action(
                    haps_ris_continuous[0], ris_mode[0]
                )
            )
            if not self.env.action_space.contains(action):
                raise ValueError(
                    "combined actor action does not satisfy environment action_space"
                )

            value = self.critic(
                critic_observation["vector"],
                critic_observation["risk_nodes"],
            )
            self._require_shape(value, (1,), "critic output")

    def _ppo_actor_loss(
        self,
        new_log_prob,
        old_log_prob,
        advantage,
        entropy,
        entropy_coef,
    ):
        log_ratio = new_log_prob - old_log_prob
        ratio = log_ratio.exp()
        surrogate_1 = ratio * advantage
        surrogate_2 = ratio.clamp(
            1.0 - self.config.clip_coef,
            1.0 + self.config.clip_coef,
        ) * advantage
        policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
        mean_normalized_entropy = entropy.mean()
        actor_loss = (
            policy_loss
            - entropy_coef * mean_normalized_entropy
        )
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (ratio - 1.0).abs() > self.config.clip_coef
            ).float().mean()
        return actor_loss, mean_normalized_entropy, approx_kl, clip_fraction

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
        self.satellite_actor.train()
        self.haps_ris_actor.train()
        self.critic.train()

        entropy_coef = self._current_entropy_coef()
        satellite_totals = {
            "actor_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        haps_ris_totals = dict.fromkeys(satellite_totals, 0.0)
        critic_loss_total = 0.0
        satellite_sample_count = 0
        haps_ris_sample_count = 0
        critic_sample_count = 0
        satellite_early_stop = False
        haps_ris_early_stop = False
        satellite_stop_kl = 0.0
        haps_ris_stop_kl = 0.0

        for _ in range(self.config.ppo_epochs):
            for batch in self.rollout.buffer.batches(
                self.config.batch_size, self.device
            ):
                count = batch["advantage"].numel()

                if not satellite_early_stop:
                    satellite_log_prob, satellite_entropy = (
                        self.satellite_actor.evaluate(
                            batch["actor_vector"],
                            batch["risk_nodes"],
                            batch["satellite_pre_tanh"],
                        )
                    )
                    (
                        satellite_actor_loss,
                        mean_satellite_entropy,
                        satellite_approx_kl,
                        satellite_clip_fraction,
                    ) = self._ppo_actor_loss(
                        satellite_log_prob,
                        batch["old_satellite_log_prob"],
                        batch["advantage"],
                        satellite_entropy,
                        entropy_coef,
                    )
                    satellite_values = {
                        "actor_loss": satellite_actor_loss.item(),
                        "entropy": mean_satellite_entropy.item(),
                        "approx_kl": satellite_approx_kl.item(),
                        "clip_fraction": satellite_clip_fraction.item(),
                    }
                    for name, value in satellite_values.items():
                        satellite_totals[name] += value * count
                    satellite_sample_count += count
                    if (
                        satellite_values["approx_kl"]
                        > self.config.satellite_target_kl
                    ):
                        satellite_early_stop = True
                        satellite_stop_kl = satellite_values["approx_kl"]
                    else:
                        self.satellite_actor_optimizer.zero_grad(set_to_none=True)
                        satellite_actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.satellite_actor.parameters(),
                            self.config.max_grad_norm,
                        )
                        self.satellite_actor_optimizer.step()

                if not haps_ris_early_stop:
                    haps_ris_log_prob, haps_ris_entropy = (
                        self.haps_ris_actor.evaluate(
                            batch["actor_vector"],
                            batch["risk_nodes"],
                            batch["haps_ris_pre_tanh"],
                            batch["ris_mode"],
                        )
                    )
                    (
                        haps_ris_actor_loss,
                        mean_haps_ris_entropy,
                        haps_ris_approx_kl,
                        haps_ris_clip_fraction,
                    ) = self._ppo_actor_loss(
                        haps_ris_log_prob,
                        batch["old_haps_ris_log_prob"],
                        batch["advantage"],
                        haps_ris_entropy,
                        entropy_coef,
                    )
                    haps_ris_values = {
                        "actor_loss": haps_ris_actor_loss.item(),
                        "entropy": mean_haps_ris_entropy.item(),
                        "approx_kl": haps_ris_approx_kl.item(),
                        "clip_fraction": haps_ris_clip_fraction.item(),
                    }
                    for name, value in haps_ris_values.items():
                        haps_ris_totals[name] += value * count
                    haps_ris_sample_count += count
                    if (
                        haps_ris_values["approx_kl"]
                        > self.config.haps_ris_target_kl
                    ):
                        haps_ris_early_stop = True
                        haps_ris_stop_kl = haps_ris_values["approx_kl"]
                    else:
                        self.haps_ris_actor_optimizer.zero_grad(set_to_none=True)
                        haps_ris_actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.haps_ris_actor.parameters(),
                            self.config.max_grad_norm,
                        )
                        self.haps_ris_actor_optimizer.step()

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
                critic_loss_total += critic_loss.item() * count
                critic_sample_count += count

        if min(
            satellite_sample_count,
            haps_ris_sample_count,
            critic_sample_count,
        ) == 0:
            raise RuntimeError("PPO update received no rollout samples")

        satellite_metrics = {
            name: value / satellite_sample_count
            for name, value in satellite_totals.items()
        }
        haps_ris_metrics = {
            name: value / haps_ris_sample_count
            for name, value in haps_ris_totals.items()
        }
        self.update_index += 1
        result = {
            "satellite_actor_loss": satellite_metrics["actor_loss"],
            "haps_ris_actor_loss": haps_ris_metrics["actor_loss"],
            "actor_loss": 0.5
            * (
                satellite_metrics["actor_loss"]
                + haps_ris_metrics["actor_loss"]
            ),
            "critic_loss": critic_loss_total / critic_sample_count,
            "satellite_entropy": satellite_metrics["entropy"],
            "haps_ris_entropy": haps_ris_metrics["entropy"],
            "entropy": 0.5
            * (satellite_metrics["entropy"] + haps_ris_metrics["entropy"]),
            "satellite_approx_kl": satellite_metrics["approx_kl"],
            "haps_ris_approx_kl": haps_ris_metrics["approx_kl"],
            "approx_kl": 0.5
            * (
                satellite_metrics["approx_kl"]
                + haps_ris_metrics["approx_kl"]
            ),
            "satellite_clip_fraction": satellite_metrics["clip_fraction"],
            "haps_ris_clip_fraction": haps_ris_metrics["clip_fraction"],
            "clip_fraction": 0.5
            * (
                satellite_metrics["clip_fraction"]
                + haps_ris_metrics["clip_fraction"]
            ),
            "entropy_coef": entropy_coef,
            "satellite_early_stop": satellite_early_stop,
            "haps_ris_early_stop": haps_ris_early_stop,
            "satellite_stop_kl": satellite_stop_kl,
            "haps_ris_stop_kl": haps_ris_stop_kl,
        }
        result.update(
            {
                "update": self.update_index,
                "from_policy_version": policy_version,
                "to_policy_version": self.update_index,
                "total_episodes": self.total_episodes,
                "total_env_steps": self.total_env_steps,
                "mean_episode_return": sum(
                    item["return"] for item in episode_events
                )
                / len(episode_events),
                "mean_outage_rate": sum(
                    item["outage_rate"] for item in episode_events
                )
                / len(episode_events),
                "episodes": episode_events,
            }
        )
        return result
