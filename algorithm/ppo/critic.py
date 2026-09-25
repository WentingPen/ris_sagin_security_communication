import torch
import torch.nn as nn
from algorithm.ppo.risk_gnn import RiskGNN

class PrivilegeCritic(nn.Module):
    def __init__(self, critic_dim, risk_shape, edge_index, risk_grid_size, hidden_dim=256, risk_hidden_dim=64, risk_feature_dim=64):
        """
        :param critic_dim: critic 向量的维度,即输入的critic向量的维度
        :param risk_shape: 风险节点的形状
        :param edge_index: 风险图的边索引
        :param risk_grid_size: 风险网格的大小
        :param hidden_dim: 隐藏层的维度
        :param risk_hidden_dim: 风险 GNN 的隐藏层维度
        :param risk_feature_dim: 风险特征的维度
        """
        super().__init__()
        self.critic_dim = int(critic_dim)

        self.risk_gnn = RiskGNN(
            edge_index=edge_index,
            grid_size=risk_grid_size,
            node_dim=risk_shape[-1],
            hidden_dim=risk_hidden_dim,
            output_dim=risk_feature_dim
        )
        input_dim = self.critic_dim + self.risk_gnn.output_dim

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, critic_vector, risk_nodes):
        device = next(self.parameters()).device

        critic_vector = torch.as_tensor(critic_vector, dtype=torch.float32, device=device)


        if critic_vector.ndim == 1:
            critic_vector = critic_vector.unsqueeze(0)

        if critic_vector.ndim != 2 or critic_vector.shape[1] != self.critic_dim:
            raise ValueError(f"critic_vector must have shape (B, {self.critic_dim})")

        risk_feature = self.risk_gnn(risk_nodes)

        if critic_vector.shape[0] != risk_feature.shape[0]:
            raise ValueError("critic_vector and risk_nodes must have the same batch size")

        features = torch.cat((critic_vector, risk_feature), dim=-1)

        return self.value_net(features).squeeze(-1)
