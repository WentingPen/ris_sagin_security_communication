import torch
import torch.nn as nn

from algorithm.ppo.risk_gnn import RiskGNN


class LocalCritic(nn.Module):
    """State value from the public observation only; no private Eve features."""

    def __init__(
        self,
        actor_dim,
        risk_shape,
        edge_index,
        risk_grid_size,
        hidden_dim=256,
        risk_hidden_dim=64,
        risk_feature_dim=64,
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
        self.value_net = nn.Sequential(
            nn.Linear(self.actor_dim + risk_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actor_vector, risk_nodes):
        device = next(self.parameters()).device
        vector = torch.as_tensor(actor_vector, dtype=torch.float32, device=device)
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        if vector.ndim != 2 or vector.shape[1] != self.actor_dim:
            raise ValueError(
                f"local critic vector must have shape (B, {self.actor_dim})"
            )
        risk_feature = self.risk_gnn(risk_nodes)
        if vector.shape[0] != risk_feature.shape[0]:
            raise ValueError("vector and risk_nodes must have the same batch size")
        return self.value_net(torch.cat((vector, risk_feature), dim=-1)).squeeze(-1)
