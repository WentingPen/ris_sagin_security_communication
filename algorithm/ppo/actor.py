import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from algorithm.ppo.risk_gnn import RiskGNN

class HybridActor(nn.Module):
    def __init__(self, actor_dim, risk_shape, edge_index, risk_grid_size, num_ris_group,
                 hidden_dim=256, risk_hidden_dim=64, risk_feature_dim=64):
        """risk_shape表示风险节点的形状，edge_index表示风险图的边索引，risk_grid_size表示风险网格的大小，
        risk_feature_dim表示风险特征维度。"""
        super().__init__()
        self.G = int(num_ris_group)
        self.actor_dim = int(actor_dim)
        self.continuous_action_shapes = {
            "uav_move": (2,),
            "ris_orientation_delta": (2,),
            "ris_amplitude_residual": (self.G,),
            "ris_phase_residual": (self.G,),
        }
        self.continuous_action_slices = {}
        offset = 0
        for name, shape in self.continuous_action_shapes.items():
            block_size = math.prod(shape)
            self.continuous_action_slices[name] = slice(offset, offset + block_size)
            offset += block_size
        self.continuous_dim = offset

        self.risk_gnn = RiskGNN(
            edge_index=edge_index,
            grid_size=risk_grid_size,
            node_dim=risk_shape[-1],
            hidden_dim=risk_hidden_dim,
            output_dim=risk_feature_dim,
        )

        input_dim = self.actor_dim + self.risk_gnn.output_dim

        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(hidden_dim, self.continuous_dim)
        self.log_std = nn.Parameter(torch.full((self.continuous_dim,), -0.5)) #表示 log_std 的初始值为 -0.5，表示标准差的初始值为 exp(-0.5) ≈ 0.6065
        self.mode_head = nn.Linear(hidden_dim, self.G)  # 表示离散动作，ris模式

    def _distribution(self, vector, risk_nodes):
        """根据输入的actor向量和风险节点特征，返回连续动作的正态分布和离散动作的伯努利分布。""" 
        device = self.log_std.device
        vector = torch.as_tensor(vector, dtype=torch.float32, device=device)
        risk_nodes = torch.as_tensor(risk_nodes, dtype=torch.float32, device=device)

        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        if risk_nodes.ndim == 2:
            risk_nodes = risk_nodes.unsqueeze(0)

        if vector.ndim != 2 or vector.shape[1] != self.actor_dim:
            raise ValueError(f"vector must have shape (B, {self.actor_dim})")
        if vector.shape[0] != risk_nodes.shape[0]:
            raise ValueError("vector and risk_nodes must have the same batch size")

        risk_feature = self.risk_gnn(risk_nodes)
        features = torch.cat((vector, risk_feature), dim=-1)
        hidden = self.body(features)

        mean = self.mean_head(hidden)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)  # 限制 log_std 的范围，防止数值不稳定

        normal = Normal(mean, std)
        bernoulli = Bernoulli(logits=self.mode_head(hidden))

        return normal, bernoulli


    @staticmethod
    def _joint_log_prob(normal, bernoulli, pre_tanh, mode):
        """计算连续和离散动作的联合对数概率。"""
        log_det = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        continuous_log_prob = (normal.log_prob(pre_tanh) - log_det).sum(dim=-1)
        mode_log_prob = bernoulli.log_prob(mode.float()).sum(dim=-1)

        return continuous_log_prob + mode_log_prob

    def sample(self, vector, risk_nodes, deterministic=False):
        """从联合分布中采样连续和离散动作，并返回相应的对数概率。"""
        normal, bernoulli = self._distribution(vector, risk_nodes)

        if deterministic:
            pre_tanh = normal.mean
            mode = (bernoulli.probs > 0.5).float()
        else:
            pre_tanh = normal.rsample()  # 使用 reparameterization trick 进行采样
            mode = bernoulli.sample()

        continuous = torch.tanh(pre_tanh)
        log_prob = self._joint_log_prob(normal, bernoulli, pre_tanh, mode)

        return continuous, mode, pre_tanh, log_prob


    def evaluate(self, vector, risk_nodes, pre_tanh, mode):
        """评估给定动作的对数概率和熵。"""
        normal, bernoulli = self._distribution(vector, risk_nodes)
        device = self.log_std.device
        #pre_tanh表示连续动作在tanh变换之前的值，mode表示离散动作的选择
        pre_tanh = torch.as_tensor(pre_tanh, dtype=torch.float32, device=device)
        mode = torch.as_tensor(mode, dtype=torch.float32, device=device)

        log_prob = self._joint_log_prob(normal, bernoulli, pre_tanh, mode)
        entropy_approx = (normal.entropy().sum(dim=-1) + bernoulli.entropy().sum(dim=-1))

        return log_prob, entropy_approx


    def to_env_action(self, continuous, mode):
        """ 将连续和离散动作转换为环境可接受的格式。"""
        continuous = continuous.detach().cpu().reshape(-1)
        continuous = continuous.numpy().astype(np.float32)

        mode = mode.detach().cpu().reshape(-1)
        mode = mode.numpy().round().astype(np.int8)

        if continuous.size != self.continuous_dim or mode.size != self.G:
            raise ValueError("to_env_action expects one action")

        blocks = {
            name: continuous[self.continuous_action_slices[name]].reshape(shape)
            for name, shape in self.continuous_action_shapes.items()
        }

        return {
            "uav_move": blocks["uav_move"],
            "ris_orientation_delta": blocks["ris_orientation_delta"],
            "ris_group_mode": mode,
            "ris_amplitude_residual": blocks["ris_amplitude_residual"],
            "ris_phase_residual": blocks["ris_phase_residual"],
        }
    
