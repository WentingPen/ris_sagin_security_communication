import torch
import torch.nn as nn
import torch.nn.functional as F

class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, nodes, adjacency):
        projected = self.linear(nodes)
        batch_size, num_nodes, hidden_dim = projected.shape
        flattened = projected.permute(1, 0, 2).reshape(
            num_nodes, batch_size * hidden_dim
        )
        output = torch.sparse.mm(adjacency, flattened)
        output = output.reshape(num_nodes, batch_size, hidden_dim).permute(1, 0, 2)
        return output + self.bias

class RiskGNN(nn.Module):
    def __init__(self, edge_index, grid_size, node_dim=3, hidden_dim=64, output_dim=64):
        """edge_index表示图的边连接关系，grid_size表示RIS网格的尺寸，node_dim表示节点特征维度，hidden_dim表示隐藏层维度，output_dim表示输出特征维度。"""
        super().__init__()

        self.grid_size = int(grid_size)
        self.node_dim = int(node_dim)
        self.num_nodes = self.grid_size ** 2
        self.output_dim = int(output_dim)

        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")

        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu").detach()

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a 2D tensor with shape (2, num_edges)")
        if edge_index.numel() > 0:
            minimum = int(edge_index.min().item())
            maximum = int(edge_index.max().item())

            if minimum < 0 or maximum >= self.num_nodes:
                raise ValueError("edge_index contains an invalid node index")

        non_self = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, non_self]

        self_loops = torch.arange(self.num_nodes, dtype=torch.long)

        source = torch.cat((edge_index[0], self_loops))
        target = torch.cat((edge_index[1], self_loops))

        degree = torch.bincount(target, minlength=self.num_nodes).float()

        edge_weight = (degree[source].rsqrt() * degree[target].rsqrt())

        axis = (2.0 * (torch.arange(self.grid_size, dtype=torch.float32) + 0.5) / self.grid_size - 1.0)

        yy, xx = torch.meshgrid(axis, axis, indexing="ij")

        node_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

        with torch.sparse.check_sparse_tensor_invariants():
            adjacency = torch.sparse_coo_tensor(
                torch.stack((target, source)),
                edge_weight,
                size=(self.num_nodes, self.num_nodes),
            ).coalesce()
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("node_xy", node_xy)

        self.conv1 = _GraphConv(self.node_dim + 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = _GraphConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.readout = nn.Sequential(nn.Linear(2 * hidden_dim, self.output_dim), nn.Tanh())

    def forward(self, risk_nodes):
        risk_nodes = torch.as_tensor(risk_nodes, dtype=torch.float32, device=self.node_xy.device)

        if risk_nodes.ndim == 2:
            risk_nodes = risk_nodes.unsqueeze(0)

        expected_shape = (self.num_nodes, self.node_dim)

        if risk_nodes.ndim != 3 or tuple(risk_nodes.shape[1:]) != expected_shape:
            raise ValueError("risk_nodes must have shape " f"(B, {self.num_nodes}, {self.node_dim})")

        coordinates = self.node_xy.unsqueeze(0)
        coordinates = coordinates.expand(risk_nodes.size(0), -1, -1)

        hidden = torch.cat((risk_nodes, coordinates), dim=-1)

        hidden = self.conv1(
            hidden,
            self.adjacency,
        )
        hidden = F.relu(self.norm1(hidden))

        hidden = self.conv2(
            hidden,
            self.adjacency,
        )
        hidden = F.relu(self.norm2(hidden))

        graph_feature = torch.cat((hidden.mean(dim=1), hidden.amax(dim=1)), dim=-1)

        return self.readout(graph_feature)
