import torch

from algorithm.IPPO.buffer import RolloutBuffer


class Rollout:
    def __init__(
        self,
        env,
        satellite_actor,
        haps_ris_actor,
        satellite_critic,
        haps_ris_critic,
        episodes_per_update=8,
        initial_state=None,
        phase_only_training=False,
        fixed_csi_seed=None,
    ):
        self.env = env
        self.satellite_actor = satellite_actor
        self.haps_ris_actor = haps_ris_actor
        self.critics = (satellite_critic, haps_ris_critic)
        self.episodes_per_update = int(episodes_per_update)
        self._pending_state = initial_state
        self.phase_only_training = bool(phase_only_training)
        self.fixed_csi_seed = fixed_csi_seed
        self.buffer = RolloutBuffer(
            capacity=self.episodes_per_update * env.horizon,
            actor_dim=env.actor_vector_dim,
            risk_shape=env.observation_space["risk_nodes"].shape,
            satellite_continuous_dim=satellite_actor.continuous_dim,
            satellite_condition_dim=haps_ris_actor.satellite_condition_dim,
            haps_ris_continuous_dim=haps_ris_actor.continuous_dim,
            ris_mode_dim=haps_ris_actor.mode_dim,
        )

    def _reset_episode(self):
        if self._pending_state is not None:
            state = self._pending_state
            self._pending_state = None
            return state
        if self.phase_only_training:
            return self.env.reset(seed=self.fixed_csi_seed)
        return self.env.reset()

    def _values(self, observation):
        # Both agents receive the same public vector/risk map in this model.
        # Each critic has its own parameters; private info is never read here.
        return torch.stack([
            critic(observation["vector"], observation["risk_nodes"])[0]
            for critic in self.critics
        ])

    @torch.no_grad()
    def collect(self, gamma=0.99, gae_lambda=0.95, episode_callback=None):
        self.buffer.clear()
        self.satellite_actor.eval()
        self.haps_ris_actor.eval()
        for critic in self.critics:
            critic.eval()
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
                vector, risk = observation["vector"], observation["risk_nodes"]
                if self.phase_only_training:
                    reference, base_phase = self.env.get_phase_only_reference()
                    satellite_continuous = torch.as_tensor(
                        reference, dtype=torch.float32,
                        device=next(self.satellite_actor.parameters()).device,
                    ).unsqueeze(0)
                    satellite_pre_tanh = torch.zeros_like(satellite_continuous)
                    satellite_log_prob = satellite_continuous.new_zeros(1)
                else:
                    (
                        satellite_continuous,
                        satellite_pre_tanh,
                        satellite_log_prob,
                    ) = self.satellite_actor.sample(vector, risk)
                    base_phase = self.env.get_ris_phase_reference(
                        satellite_continuous[0].cpu().numpy()
                    )
                satellite_condition = self.haps_ris_actor.satellite_beam_condition(
                    satellite_continuous
                )
                (
                    haps_ris_continuous,
                    ris_mode,
                    haps_ris_pre_tanh,
                    haps_ris_log_prob,
                ) = self.haps_ris_actor.sample(vector, risk, satellite_condition)
                value = self._values(observation)
                action = self.satellite_actor.to_env_action(satellite_continuous[0])
                action.update(self.haps_ris_actor.to_env_action(
                    haps_ris_continuous[0], ris_mode[0], base_phase=base_phase,
                ))
                next_observation, reward, terminated, truncated, next_info = (
                    self.env.step(action)
                )
                next_value = (
                    torch.zeros_like(value) if terminated
                    else self._values(next_observation)
                )
                self.buffer.add(
                    actor_vector=vector,
                    risk_nodes=risk,
                    satellite_pre_tanh=satellite_pre_tanh[0],
                    satellite_condition=satellite_condition[0],
                    haps_ris_pre_tanh=haps_ris_pre_tanh[0],
                    ris_mode=ris_mode[0],
                    old_satellite_log_prob=satellite_log_prob[0],
                    old_haps_ris_log_prob=haps_ris_log_prob[0],
                    value=value,
                    reward=reward,
                    next_value=next_value,
                    terminated=terminated,
                    truncated=truncated,
                )

                # Eve metrics are used only for reporting; never as model input.
                bob = torch.as_tensor(next_info["bob_capacity"])
                eve = torch.as_tensor(next_info["max_eve_capacity"])
                bob_capacity_sum += bob.mean().item()
                max_eve_capacity_sum += eve.mean().item()
                step_max_bob_capacity_sum += bob.max().item()
                step_max_eve_capacity_sum += eve.max().item()
                jammer_inr_sum += torch.as_tensor(
                    next_info["bob_jammer_inr"]
                ).mean().item()
                ris_repair_count += int(next_info["ris_action_repaired"])
                episode_return += float(reward)
                observation, info = next_observation, next_info
                if terminated or truncated:
                    break

            length = step + 1
            max_bob = step_max_bob_capacity_sum / length
            max_eve = step_max_eve_capacity_sum / length
            stats = {
                "return": episode_return,
                "mean_reward": episode_return / length,
                "length": length,
                "outage_rate": float(info["episode_outage_rate"]),
                "num_eves": 1 + int(info["num_passive_eves"]),
                "mean_raw_security_gap": max_bob - max_eve,
                "mean_bob_capacity": bob_capacity_sum / length,
                "mean_max_eve_capacity": max_eve_capacity_sum / length,
                "max_bob_capacity": max_bob,
                "max_eve_capacity": max_eve,
                "mean_bob_jammer_inr": jammer_inr_sum / length,
                "ris_repair_rate": ris_repair_count / length,
            }
            episode_stats.append(stats)
            if episode_callback is not None:
                episode_callback(dict(stats))
        self.buffer.compute_advantages(gamma, gae_lambda, normalize=True)
        return episode_stats
