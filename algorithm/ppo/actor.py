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
    """HAPS-RIS policy with per-element mode and complex phase heads."""

    def __init__(
        self,
        actor_dim,
        risk_shape,
        edge_index,
        risk_grid_size,
        num_ris_elements,
        hidden_dim=256,
        risk_hidden_dim=64,
        risk_feature_dim=64,
    ):
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.num_elements = int(num_ris_elements)
        if self.num_elements <= 0:
            raise ValueError("num_ris_elements must be positive")

        self.continuous_action_shapes = {
            "ris_phase_real": (self.num_elements,),
            "ris_phase_imag": (self.num_elements,),
        }
        self.continuous_dim = 2 * self.num_elements
        self.mode_dim = self.num_elements
        self.feature_extractor = _ActorFeatureExtractor(
            actor_dim=self.actor_dim,
            risk_shape=risk_shape,
            edge_index=edge_index,
            risk_grid_size=risk_grid_size,
            hidden_dim=hidden_dim,
            risk_hidden_dim=risk_hidden_dim,
            risk_feature_dim=risk_feature_dim,
        )
        self.mode_head = nn.Linear(hidden_dim, self.num_elements)
        self.phase_real_head = nn.Linear(hidden_dim, self.num_elements)
        self.phase_imag_head = nn.Linear(hidden_dim, self.num_elements)
        self.phase_real_log_std = nn.Parameter(
            torch.full((self.num_elements,), -0.5)
        )
        self.phase_imag_log_std = nn.Parameter(
            torch.full((self.num_elements,), -0.5)
        )

    @staticmethod
    def _normal(mean, log_std):
        std = log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def _distributions(self, vector, risk_nodes):
        hidden = self.feature_extractor(vector, risk_nodes)
        mode = Bernoulli(logits=self.mode_head(hidden))
        phase_real = self._normal(
            self.phase_real_head(hidden), self.phase_real_log_std
        )
        phase_imag = self._normal(
            self.phase_imag_head(hidden), self.phase_imag_log_std
        )
        return mode, phase_real, phase_imag

    def sample(self, vector, risk_nodes, deterministic=False):
        mode_distribution, real_distribution, imag_distribution = (
            self._distributions(vector, risk_nodes)
        )
        if deterministic:
            mode = (mode_distribution.probs >= 0.5).float()
            real_pre_tanh = real_distribution.mean
            imag_pre_tanh = imag_distribution.mean
        else:
            mode = mode_distribution.sample()
            real_pre_tanh = real_distribution.rsample()
            imag_pre_tanh = imag_distribution.rsample()

        pre_tanh = torch.cat((real_pre_tanh, imag_pre_tanh), dim=-1)
        continuous = torch.tanh(pre_tanh)
        log_prob = (
            mode_distribution.log_prob(mode).sum(dim=-1)
            + _squashed_normal_log_prob(real_distribution, real_pre_tanh)
            + _squashed_normal_log_prob(imag_distribution, imag_pre_tanh)
        )
        return continuous, mode, pre_tanh, log_prob

    def evaluate(self, vector, risk_nodes, pre_tanh, mode):
        mode_distribution, real_distribution, imag_distribution = (
            self._distributions(vector, risk_nodes)
        )
        device = self.phase_real_log_std.device
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
        if pre_tanh.shape[0] != real_distribution.mean.shape[0]:
            raise ValueError("actions and observations must have the same batch size")

        real_pre_tanh, imag_pre_tanh = pre_tanh.split(
            self.num_elements, dim=-1
        )
        log_prob = (
            mode_distribution.log_prob(mode).sum(dim=-1)
            + _squashed_normal_log_prob(real_distribution, real_pre_tanh)
            + _squashed_normal_log_prob(imag_distribution, imag_pre_tanh)
        )
        normalized_entropy = (
            mode_distribution.entropy().sum(dim=-1)
            + real_distribution.entropy().sum(dim=-1)
            + imag_distribution.entropy().sum(dim=-1)
        ) / (self.mode_dim + self.continuous_dim)
        return log_prob, normalized_entropy

    def to_env_action(self, continuous, mode):
        continuous = (
            continuous.detach().cpu().reshape(-1).numpy().astype(np.float32)
        )
        mode = (
            mode.detach().cpu().reshape(-1).numpy().round().astype(np.int8)
        )
        if continuous.size != self.continuous_dim or mode.size != self.mode_dim:
            raise ValueError("to_env_action expects one HAPS-RIS action")
        real, imag = np.split(continuous, 2)
        return {
            "ris_element_mode": mode,
            "ris_phase_real": real,
            "ris_phase_imag": imag,
        }
