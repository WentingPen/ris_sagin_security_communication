from dataclasses import dataclass
import math

import torch

from algorithm.IPPO.actor import HAPSRISActor, SatelliteActor
from algorithm.IPPO.critic import LocalCritic
from algorithm.IPPO.Rollout import Rollout


@dataclass(frozen=True)
class IPPOConfig:
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
    haps_ris_actor_lr: float | None = None
    critic_lr: float = 1e-4
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    risk_hidden_dim: int = 64
    risk_feature_dim: int = 64
    phase_only_training: bool = False
    fixed_csi_seed: int | None = None
    phase_log_std_init: float = -0.5
    phase_residual_max_deg: float = 30.0


class Controller:
    def __init__(self, env, config=None, seed=None, device=None):
        self.env = env
        self.config = IPPOConfig() if config is None else config
        self._validate_config()
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        base_seed = int(env.params.seed if seed is None else seed)
        if self.config.phase_only_training and self.config.fixed_csi_seed is not None:
            base_seed = int(self.config.fixed_csi_seed)
        torch.manual_seed(base_seed)
        initial_state = env.reset(seed=base_seed)
        observation, _ = initial_state
        params = env.params
        common = {
            "actor_dim": env.actor_vector_dim,
            "risk_shape": observation["risk_nodes"].shape,
            "edge_index": observation["edge_index"],
            "risk_grid_size": params.risk_grid_size,
            "hidden_dim": self.config.hidden_dim,
            "risk_hidden_dim": self.config.risk_hidden_dim,
            "risk_feature_dim": self.config.risk_feature_dim,
        }
        self.satellite_actor = SatelliteActor(
            **common, num_sat_antennas=params.num_sat_antennas,
        ).to(self.device)
        self.haps_ris_actor = HAPSRISActor(
            **common,
            num_ris_elements=params.num_ris_elements,
            num_sat_antennas=params.num_sat_antennas,
            num_ris_groups=params.num_ris_groups,
            element_to_group=env.uav_ris.element_to_group,
            phase_only_training=self.config.phase_only_training,
            phase_log_std_init=self.config.phase_log_std_init,
            phase_residual_max_deg=self.config.phase_residual_max_deg,
        ).to(self.device)
        if self.config.phase_only_training:
            for parameter in self.satellite_actor.parameters():
                parameter.requires_grad_(False)

        self.satellite_critic = LocalCritic(**common).to(self.device)
        self.haps_ris_critic = LocalCritic(**common).to(self.device)
        self.actors = (self.satellite_actor, self.haps_ris_actor)
        self.critics = (self.satellite_critic, self.haps_ris_critic)
        ris_lr = (
            self.config.actor_lr if self.config.haps_ris_actor_lr is None
            else self.config.haps_ris_actor_lr
        )
        self.satellite_actor_optimizer = torch.optim.Adam(
            self.satellite_actor.parameters(), lr=self.config.actor_lr,
        )
        self.haps_ris_actor_optimizer = torch.optim.Adam(
            (p for p in self.haps_ris_actor.parameters() if p.requires_grad),
            lr=ris_lr,
        )
        self.satellite_critic_optimizer = torch.optim.Adam(
            self.satellite_critic.parameters(), lr=self.config.critic_lr,
        )
        self.haps_ris_critic_optimizer = torch.optim.Adam(
            self.haps_ris_critic.parameters(), lr=self.config.critic_lr,
        )
        self.actor_optimizers = (
            self.satellite_actor_optimizer, self.haps_ris_actor_optimizer,
        )
        self.critic_optimizers = (
            self.satellite_critic_optimizer, self.haps_ris_critic_optimizer,
        )
        self.rollout = Rollout(
            env=env,
            satellite_actor=self.satellite_actor,
            haps_ris_actor=self.haps_ris_actor,
            satellite_critic=self.satellite_critic,
            haps_ris_critic=self.haps_ris_critic,
            episodes_per_update=self.config.episodes_per_update,
            initial_state=initial_state,
            phase_only_training=self.config.phase_only_training,
            fixed_csi_seed=base_seed if self.config.phase_only_training else None,
        )
        self.update_index = 0
        self.total_episodes = 0
        self.total_env_steps = 0

    def _validate_config(self):
        c = self.config
        if min(c.episodes_per_update, c.batch_size, c.ppo_epochs) <= 0:
            raise ValueError("rollout, minibatch, and epoch counts must be positive")
        if not 0.0 <= c.gamma <= 1.0 or not 0.0 <= c.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if not 0.0 < c.clip_coef < 1.0:
            raise ValueError("clip_coef must lie in (0, 1)")
        ris_lr = c.actor_lr if c.haps_ris_actor_lr is None else c.haps_ris_actor_lr
        if min(c.actor_lr, ris_lr, c.critic_lr, c.max_grad_norm) <= 0:
            raise ValueError("learning rates and max_grad_norm must be positive")
        if min(c.entropy_coef, c.entropy_coef_final) < 0:
            raise ValueError("entropy coefficients must be non-negative")
        if min(c.entropy_anneal_updates, c.satellite_target_kl, c.haps_ris_target_kl) <= 0:
            raise ValueError("entropy schedule and KL thresholds must be positive")
        if not math.isfinite(c.phase_log_std_init):
            raise ValueError("phase_log_std_init must be finite")
        if not 0.0 < c.phase_residual_max_deg <= 180.0:
            raise ValueError("phase_residual_max_deg must lie in (0, 180]")
        if c.fixed_csi_seed is not None and c.fixed_csi_seed < 0:
            raise ValueError("fixed_csi_seed must be non-negative")

    def _current_entropy_coef(self):
        c = self.config
        if c.entropy_anneal_updates == 1:
            return float(c.entropy_coef_final)
        progress = min(self.update_index / (c.entropy_anneal_updates - 1), 1.0)
        return float(c.entropy_coef + progress * (c.entropy_coef_final - c.entropy_coef))

    def _ppo_actor_loss(self, new_log_prob, old_log_prob, advantage, entropy, entropy_coef):
        log_ratio = new_log_prob - old_log_prob
        ratio = log_ratio.exp()
        surrogate = torch.minimum(
            ratio * advantage,
            ratio.clamp(
                1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef,
            ) * advantage,
        )
        mean_entropy = entropy.mean()
        loss = -surrogate.mean() - entropy_coef * mean_entropy
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (ratio - 1.0).abs() > self.config.clip_coef
            ).float().mean()
        return loss, mean_entropy, approx_kl, clip_fraction

    def update(self, episode_callback=None):
        policy_version = self.update_index
        episodes = []

        def handle_episode(stats):
            self.total_episodes += 1
            self.total_env_steps += stats["length"]
            event = {
                **stats,
                "episode": self.total_episodes,
                "env_steps": self.total_env_steps,
                "policy_version": policy_version,
            }
            episodes.append(event)
            if episode_callback is not None:
                episode_callback(dict(event))

        self.rollout.collect(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
            episode_callback=handle_episode,
        )
        for index, actor in enumerate(self.actors):
            actor.train(not (index == 0 and self.config.phase_only_training))
        for critic in self.critics:
            critic.train()
        names = ("satellite", "haps_ris")
        loss_fields = ("actor_loss", "entropy", "approx_kl", "clip_fraction")
        actor_totals = [dict.fromkeys(loss_fields, 0.0) for _ in names]
        actor_counts = [0, 0]
        critic_totals = [0.0, 0.0]
        critic_count = 0
        early_stop = [False, False]
        stop_kl = [0.0, 0.0]
        kl_limits = (
            self.config.satellite_target_kl, self.config.haps_ris_target_kl,
        )
        entropy_coef = self._current_entropy_coef()

        for _ in range(self.config.ppo_epochs):
            for batch in self.rollout.buffer.batches(self.config.batch_size, self.device):
                count = batch["reward"].numel()
                for index, actor in enumerate(self.actors):
                    if early_stop[index] or (
                        index == 0 and self.config.phase_only_training
                    ):
                        continue
                    if index == 0:
                        log_prob, entropy = actor.evaluate(
                            batch["actor_vector"], batch["risk_nodes"],
                            batch["satellite_pre_tanh"],
                        )
                    else:
                        log_prob, entropy = actor.evaluate(
                            batch["actor_vector"], batch["risk_nodes"],
                            batch["satellite_condition"], batch["haps_ris_pre_tanh"],
                            batch["ris_mode"],
                        )
                    loss, mean_entropy, kl, clip = self._ppo_actor_loss(
                        log_prob, batch[f"old_{names[index]}_log_prob"],
                        batch["advantage"][:, index], entropy, entropy_coef,
                    )
                    values = (loss.item(), mean_entropy.item(), kl.item(), clip.item())
                    for field, value in zip(loss_fields, values):
                        actor_totals[index][field] += value * count
                    actor_counts[index] += count
                    if kl.item() > kl_limits[index]:
                        early_stop[index] = True
                        stop_kl[index] = kl.item()
                        continue
                    optimizer = self.actor_optimizers[index]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), self.config.max_grad_norm,
                    )
                    optimizer.step()

                for index, critic in enumerate(self.critics):
                    prediction = critic(batch["actor_vector"], batch["risk_nodes"])
                    loss = 0.5 * (
                        prediction - batch["returns"][:, index]
                    ).square().mean()
                    optimizer = self.critic_optimizers[index]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), self.config.max_grad_norm,
                    )
                    optimizer.step()
                    critic_totals[index] += loss.item() * count
                critic_count += count

        if critic_count == 0 or actor_counts[1] == 0:
            raise RuntimeError("IPPO update received no rollout samples")
        active_agents = (1,) if self.config.phase_only_training else (0, 1)
        if any(actor_counts[index] == 0 for index in active_agents):
            raise RuntimeError("IPPO update received no active actor samples")
        actor_metrics = [
            {
                field: total / actor_counts[index] if actor_counts[index] else 0.0
                for field, total in actor_totals[index].items()
            }
            for index in range(2)
        ]
        result = {}
        for index, name in enumerate(names):
            for field in loss_fields:
                result[f"{name}_{field}"] = actor_metrics[index][field]
            result[f"{name}_critic_loss"] = critic_totals[index] / critic_count
            result[f"{name}_early_stop"] = early_stop[index]
            result[f"{name}_stop_kl"] = stop_kl[index]
        for field in loss_fields:
            result[field] = sum(
                actor_metrics[index][field] for index in active_agents
            ) / len(active_agents)
        result["critic_loss"] = sum(critic_totals) / (2 * critic_count)
        result["entropy_coef"] = entropy_coef
        self.update_index += 1
        result.update({
            "update": self.update_index,
            "from_policy_version": policy_version,
            "to_policy_version": self.update_index,
            "total_episodes": self.total_episodes,
            "total_env_steps": self.total_env_steps,
            "mean_episode_return": sum(e["return"] for e in episodes) / len(episodes),
            "mean_outage_rate": sum(e["outage_rate"] for e in episodes) / len(episodes),
            "episodes": episodes,
        })
        return result
