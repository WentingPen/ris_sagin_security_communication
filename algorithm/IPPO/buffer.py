import torch


class RolloutBuffer:
    """Public observations and two independent value/GAE/return columns."""

    def __init__(
        self,
        capacity,
        actor_dim,
        risk_shape,
        satellite_continuous_dim,
        satellite_condition_dim,
        haps_ris_continuous_dim,
        ris_mode_dim,
    ):
        self.capacity = int(capacity)
        self.pos = 0
        shapes = {
            "actor_vector": (int(actor_dim),),
            "risk_nodes": tuple(risk_shape),
            "satellite_pre_tanh": (int(satellite_continuous_dim),),
            "satellite_condition": (int(satellite_condition_dim),),
            "haps_ris_pre_tanh": (int(haps_ris_continuous_dim),),
            "ris_mode": (int(ris_mode_dim),),
            "old_satellite_log_prob": (),
            "old_haps_ris_log_prob": (),
            "value": (2,),
            "reward": (),
            "next_value": (2,),
            "terminated": (),
            "truncated": (),
            "advantage": (2,),
            "returns": (2,),
        }
        self.data = {
            name: torch.empty(
                (self.capacity, *shape),
                dtype=(
                    torch.bool if name in ("terminated", "truncated")
                    else torch.float32
                ),
            )
            for name, shape in shapes.items()
        }

    def add(self, **sample):
        if self.pos >= self.capacity:
            raise RuntimeError("RolloutBuffer is full")
        expected = self.data.keys() - {"advantage", "returns"}
        if sample.keys() != expected:
            raise ValueError("rollout sample fields do not match the buffer")
        for name, value in sample.items():
            target = self.data[name][self.pos]
            tensor = torch.as_tensor(
                value, dtype=target.dtype, device="cpu"
            ).detach()
            if tensor.shape != target.shape:
                raise ValueError(f"{name} must have shape {tuple(target.shape)}")
            target.copy_(tensor)
        self.pos += 1

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95, normalize=True):
        if self.pos == 0:
            raise RuntimeError("RolloutBuffer is empty")
        gae = torch.zeros(2)
        for index in reversed(range(self.pos)):
            bootstrap = 1.0 - self.data["terminated"][index].float()
            continues = 1.0 - (
                self.data["terminated"][index] | self.data["truncated"][index]
            ).float()
            delta = (
                self.data["reward"][index]
                + gamma * self.data["next_value"][index] * bootstrap
                - self.data["value"][index]
            )
            gae = delta + gamma * gae_lambda * continues * gae
            self.data["advantage"][index].copy_(gae)
        advantage = self.data["advantage"][:self.pos]
        self.data["returns"][:self.pos] = (
            advantage + self.data["value"][:self.pos]
        )
        if normalize and self.pos > 1:
            # Normalize each agent separately, after constructing raw returns.
            advantage.copy_(
                (advantage - advantage.mean(dim=0))
                / (advantage.std(dim=0, unbiased=False) + 1e-8)
            )

    def batches(self, batch_size, device="cpu"):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        order = torch.randperm(self.pos)
        for start in range(0, self.pos, batch_size):
            indices = order[start:start + batch_size]
            yield {
                name: tensor[indices].to(device)
                for name, tensor in self.data.items()
            }

    def clear(self):
        self.pos = 0
