import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

from algorithm.ppo.risk_gnn import RiskGNN


def _squashed_normal_log_prob(distribution, pre_tanh):
    """Log probability of tanh(z), evaluated from its pre-tanh value z."""
    log_abs_det_jacobian = 2.0 * (
        math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
    )
    return (distribution.log_prob(pre_tanh) - log_abs_det_jacobian).sum(dim=-1)


class _ActorFeatureExtractor(nn.Module):
    def __init__(
        self,
        actor_dim,
        risk_shape,
        edge_index,
        risk_grid_size,
        hidden_dim,
        risk_hidden_dim,
        risk_feature_dim,
    ):
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.risk_gnn = RiskGNN(
            edge_index=edge_index,
            grid_size=risk_grid_size,
            node_dim=risk_shape[-1],
            hidden_dim=risk_hidden_dim,
            output_dim=risk_feature_dim,
        )
        self.feature_net = nn.Sequential(
            nn.Linear(self.actor_dim + self.risk_gnn.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, vector, risk_nodes):
        device = next(self.parameters()).device
        vector = torch.as_tensor(vector, dtype=torch.float32, device=device)
        risk_nodes = torch.as_tensor(
            risk_nodes, dtype=torch.float32, device=device
        )
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        if risk_nodes.ndim == 2:
            risk_nodes = risk_nodes.unsqueeze(0)
        if vector.ndim != 2 or vector.shape[1] != self.actor_dim:
            raise ValueError(f"vector must have shape (B, {self.actor_dim})")
        if risk_nodes.ndim != 3 or vector.shape[0] != risk_nodes.shape[0]:
            raise ValueError(
                "vector and risk_nodes must have matching batch dimensions"
            )

        risk_feature = self.risk_gnn(risk_nodes)
        return self.feature_net(torch.cat((vector, risk_feature), dim=-1))


class SatelliteActor(nn.Module):
    """Satellite policy with a shared encoder and complex beamforming heads."""

    def __init__(
        self,
        actor_dim,
        risk_shape,
        edge_index,
        risk_grid_size,
        num_sat_antennas,
        hidden_dim=256,
        risk_hidden_dim=64,
        risk_feature_dim=64,
    ):
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.num_antennas = int(num_sat_antennas)
        if self.num_antennas <= 0:
            raise ValueError("num_sat_antennas must be positive")

        self.continuous_action_shapes = {
            "sat_beam_real": (self.num_antennas,),
            "sat_beam_imag": (self.num_antennas,),
        }
        self.env_action_shapes = dict(self.continuous_action_shapes)
        self.continuous_dim = 2 * self.num_antennas
        self.feature_extractor = _ActorFeatureExtractor(
            actor_dim=self.actor_dim,
            risk_shape=risk_shape,
            edge_index=edge_index,
            risk_grid_size=risk_grid_size,
            hidden_dim=hidden_dim,
            risk_hidden_dim=risk_hidden_dim,
            risk_feature_dim=risk_feature_dim,
        )
        self.beam_real_head = nn.Linear(hidden_dim, self.num_antennas)
        self.beam_imag_head = nn.Linear(hidden_dim, self.num_antennas)
        self.beam_real_log_std = nn.Parameter(
            torch.full((self.num_antennas,), -0.5)
        )
        self.beam_imag_log_std = nn.Parameter(
            torch.full((self.num_antennas,), -0.5)
        )

    @staticmethod
    def _normal(mean, log_std):
        std = log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def _distributions(self, vector, risk_nodes):
        hidden = self.feature_extractor(vector, risk_nodes)
        real = self._normal(
            self.beam_real_head(hidden), self.beam_real_log_std
        )
        imag = self._normal(
            self.beam_imag_head(hidden), self.beam_imag_log_std
        )
        return real, imag

    def sample(self, vector, risk_nodes, deterministic=False):
        real_distribution, imag_distribution = self._distributions(
            vector, risk_nodes
        )
        if deterministic:
            real_pre_tanh = real_distribution.mean
            imag_pre_tanh = imag_distribution.mean
        else:
            real_pre_tanh = real_distribution.rsample()
            imag_pre_tanh = imag_distribution.rsample()

        pre_tanh = torch.cat((real_pre_tanh, imag_pre_tanh), dim=-1)
        continuous = torch.tanh(pre_tanh)
        log_prob = (
            _squashed_normal_log_prob(real_distribution, real_pre_tanh)
            + _squashed_normal_log_prob(imag_distribution, imag_pre_tanh)
        )
        return continuous, pre_tanh, log_prob

    def evaluate(self, vector, risk_nodes, pre_tanh):
        real_distribution, imag_distribution = self._distributions(
            vector, risk_nodes
        )
        device = self.beam_real_log_std.device
        pre_tanh = torch.as_tensor(
            pre_tanh, dtype=torch.float32, device=device
        )
        if pre_tanh.ndim == 1:
            pre_tanh = pre_tanh.unsqueeze(0)
        if pre_tanh.ndim != 2 or pre_tanh.shape[1] != self.continuous_dim:
            raise ValueError(
                f"pre_tanh must have shape (B, {self.continuous_dim})"
            )
        if pre_tanh.shape[0] != real_distribution.mean.shape[0]:
            raise ValueError("pre_tanh and observations must have the same batch size")

        real_pre_tanh, imag_pre_tanh = pre_tanh.split(
            self.num_antennas, dim=-1
        )
        log_prob = (
            _squashed_normal_log_prob(real_distribution, real_pre_tanh)
            + _squashed_normal_log_prob(imag_distribution, imag_pre_tanh)
        )
        normalized_entropy = (
            real_distribution.entropy().sum(dim=-1)
            + imag_distribution.entropy().sum(dim=-1)
        ) / self.continuous_dim
        return log_prob, normalized_entropy

    def to_env_action(self, continuous):
        continuous = (
            continuous.detach().cpu().reshape(-1).numpy().astype(np.float32)
        )
        if continuous.size != self.continuous_dim:
            raise ValueError("to_env_action expects one satellite action")
        real, imag = np.split(continuous, 2)
        return {
            "sat_beam_real": real,
            "sat_beam_imag": imag,
        }


class HAPSRISActor(nn.Module):
    """HAPS-RIS policy conditioned on the current satellite beam."""

    def __init__(
        self,
        actor_dim,
        risk_shape,
        edge_index,
        risk_grid_size,
        num_ris_elements,
        num_sat_antennas,
        num_ris_groups,
        element_to_group,
        hidden_dim=256,
        risk_hidden_dim=64,
        risk_feature_dim=64,
        phase_only_training=False,
        phase_log_std_init=-0.5,
        phase_residual_max_deg=30.0,
    ):
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.num_elements = int(num_ris_elements)
        self.num_sat_antennas = int(num_sat_antennas)
        self.num_groups = int(num_ris_groups)
        self.element_to_group = np.array(
            element_to_group, dtype=np.int64, copy=True
        )
        self.phase_only_training = bool(phase_only_training)
        self.phase_residual_max = np.radians(float(phase_residual_max_deg))
        if min(self.num_elements, self.num_sat_antennas, self.num_groups) <= 0:
            raise ValueError(
                "num_ris_elements, num_sat_antennas, and num_ris_groups "
                "must be positive"
            )
        if self.element_to_group.shape != (self.num_elements,) or not np.array_equal(
            np.unique(self.element_to_group), np.arange(self.num_groups)
        ):
            raise ValueError("element_to_group must map every element to a RIS group")
        if not np.isfinite(phase_log_std_init):
            raise ValueError("phase_log_std_init must be finite")
        if not 0.0 < self.phase_residual_max <= np.pi:
            raise ValueError("phase_residual_max_deg must lie in (0, 180]")

        self.continuous_action_shapes = {
            "ris_group_phase": (self.num_groups,),
        }
        self.env_action_shapes = {
            "ris_element_mode": (self.num_elements,),
            "ris_phase_real": (self.num_elements,),
            "ris_phase_imag": (self.num_elements,),
        }
        self.continuous_dim = self.num_groups
        self.mode_dim = self.num_groups
        self.satellite_condition_dim = 2 * self.num_sat_antennas
        self.feature_extractor = _ActorFeatureExtractor(
            actor_dim=self.actor_dim,
            risk_shape=risk_shape,
            edge_index=edge_index,
            risk_grid_size=risk_grid_size,
            hidden_dim=hidden_dim,
            risk_hidden_dim=risk_hidden_dim,
            risk_feature_dim=risk_feature_dim,
        )
        head_input_dim = hidden_dim + self.satellite_condition_dim
        self.conditioning_net = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mode_head = nn.Linear(hidden_dim, self.num_groups)
        self.phase_head = nn.Linear(hidden_dim, self.num_groups)
        self.phase_log_std = nn.Parameter(
            torch.full((self.num_groups,), float(phase_log_std_init))
        )
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        if self.phase_only_training:
            for parameter in self.mode_head.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _normal(mean, log_std):
        std = log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def satellite_beam_condition(self, satellite_continuous):
        device = self.phase_log_std.device
        satellite_continuous = torch.as_tensor(
            satellite_continuous,
            dtype=torch.float32,
            device=device,
        )
        if satellite_continuous.ndim == 1:
            satellite_continuous = satellite_continuous.unsqueeze(0)
        if (
            satellite_continuous.ndim != 2
            or satellite_continuous.shape[1] != self.satellite_condition_dim
        ):
            raise ValueError(
                "satellite_continuous must have shape "
                f"(B, {self.satellite_condition_dim})"
            )
        norm = torch.linalg.vector_norm(
            satellite_continuous, dim=-1, keepdim=True
        )
        return satellite_continuous / norm.clamp_min(1.0)

    def _distributions(self, vector, risk_nodes, satellite_continuous):
        hidden = self.feature_extractor(vector, risk_nodes)
        satellite_beam = self.satellite_beam_condition(
            satellite_continuous
        )
        if hidden.shape[0] != satellite_beam.shape[0]:
            raise ValueError(
                "observations and satellite_continuous must have the same "
                "batch size"
            )
        conditioned_hidden = self.conditioning_net(
            torch.cat((hidden, satellite_beam), dim=-1)
        )
        mode = Bernoulli(logits=self.mode_head(conditioned_hidden))
        phase = self._normal(
            self.phase_head(conditioned_hidden), self.phase_log_std
        )
        return mode, phase

    def sample(
        self,
        vector,
        risk_nodes,
        satellite_continuous,
        deterministic=False,
    ):
        mode_distribution, phase_distribution = self._distributions(
            vector, risk_nodes, satellite_continuous
        )
        if deterministic:
            mode = (
                torch.zeros_like(mode_distribution.probs)
                if self.phase_only_training
                else (mode_distribution.probs >= 0.5).float()
            )
            pre_tanh = phase_distribution.mean
        else:
            mode = (
                torch.zeros_like(mode_distribution.probs)
                if self.phase_only_training
                else mode_distribution.sample()
            )
            pre_tanh = phase_distribution.rsample()

        continuous = torch.tanh(pre_tanh)
        log_prob = _squashed_normal_log_prob(phase_distribution, pre_tanh)
        if not self.phase_only_training:
            log_prob = (
                mode_distribution.log_prob(mode).sum(dim=-1) + log_prob
            )
        return continuous, mode, pre_tanh, log_prob

    def evaluate(
        self,
        vector,
        risk_nodes,
        satellite_continuous,
        pre_tanh,
        mode,
    ):
        mode_distribution, phase_distribution = self._distributions(
            vector, risk_nodes, satellite_continuous
        )
        device = self.phase_log_std.device
        pre_tanh = torch.as_tensor(
            pre_tanh, dtype=torch.float32, device=device
        )
        mode = torch.as_tensor(mode, dtype=torch.float32, device=device)
        if pre_tanh.ndim == 1:
            pre_tanh = pre_tanh.unsqueeze(0)
        if mode.ndim == 1:
            mode = mode.unsqueeze(0)
        if pre_tanh.ndim != 2 or pre_tanh.shape[1] != self.continuous_dim:
            raise ValueError(
                f"pre_tanh must have shape (B, {self.continuous_dim})"
            )
        expected_mode_shape = (pre_tanh.shape[0], self.mode_dim)
        if tuple(mode.shape) != expected_mode_shape:
            raise ValueError(f"mode must have shape {expected_mode_shape}")
        if self.phase_only_training and torch.any(mode != 0.0):
            raise ValueError("phase-only training requires every RIS mode to be passive")
        if pre_tanh.shape[0] != phase_distribution.mean.shape[0]:
            raise ValueError("actions and observations must have the same batch size")

        log_prob = _squashed_normal_log_prob(
            phase_distribution, pre_tanh
        )
        normalized_entropy = phase_distribution.entropy().sum(
            dim=-1
        ) / self.continuous_dim
        if not self.phase_only_training:
            log_prob = (
                mode_distribution.log_prob(mode).sum(dim=-1) + log_prob
            )
            normalized_entropy = (
                mode_distribution.entropy().sum(dim=-1)
                + phase_distribution.entropy().sum(dim=-1)
            ) / (self.mode_dim + self.continuous_dim)
        return log_prob, normalized_entropy

    def to_env_action(self, continuous, mode, base_phase):
        continuous = (
            continuous.detach().cpu().reshape(-1).numpy().astype(np.float32)
        )
        mode = (
            mode.detach().cpu().reshape(-1).numpy().round().astype(np.int8)
        )
        if continuous.size != self.continuous_dim or mode.size != self.mode_dim:
            raise ValueError("to_env_action expects one HAPS-RIS action")
        base_phase = np.asarray(base_phase, dtype=float).reshape(-1)
        if base_phase.size != self.num_elements or not np.all(np.isfinite(base_phase)):
            raise ValueError(
                "base_phase must contain one finite value per RIS element"
            )
        if self.phase_only_training:
            mode = np.zeros(self.num_groups, dtype=np.int8)
        phase = base_phase + self.phase_residual_max * continuous[self.element_to_group]
        return {
            "ris_element_mode": mode[self.element_to_group],
            "ris_phase_real": np.cos(phase).astype(np.float32),
            "ris_phase_imag": np.sin(phase).astype(np.float32),
        }
