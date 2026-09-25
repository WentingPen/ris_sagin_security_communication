import torch

class RolloutBuffer:
    def __init__(self, capacity, actor_dim, critic_dim, risk_shape, continuous_dim, mode_dim):
        capacity = int(capacity)
        actor_dim = int(actor_dim)
        critic_dim = int(critic_dim)
        continuous_dim = int(continuous_dim)
        mode_dim = int(mode_dim)
        risk_shape = tuple(int(value) for value in risk_shape)

        self.capacity = capacity
        self.pos = 0

        self.actor_vector = torch.empty((capacity, actor_dim), dtype=torch.float32)
        self.risk_nodes = torch.empty((capacity, *risk_shape), dtype=torch.float32)
        self.critic_vector = torch.empty((capacity, critic_dim), dtype=torch.float32)
        self.pre_tanh = torch.empty((capacity, continuous_dim), dtype=torch.float32)
        self.mode = torch.empty((capacity, mode_dim), dtype=torch.float32)
        self.log_prob = torch.empty(capacity, dtype=torch.float32)
        self.value = torch.empty(capacity, dtype=torch.float32)
        self.reward = torch.empty(capacity, dtype=torch.float32)
        self.next_value = torch.empty(capacity, dtype=torch.float32)
        self.terminated = torch.empty(capacity, dtype=torch.bool)
        self.truncated = torch.empty(capacity, dtype=torch.bool)
        self.advantage = torch.empty(capacity, dtype=torch.float32)
        self.returns = torch.empty(capacity, dtype=torch.float32)

    @staticmethod
    def _cpu_tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device="cpu").detach()

    @classmethod
    def _copy_sample(cls, target, value, name, flatten=False):
        tensor = cls._cpu_tensor(value)
        if flatten:
            tensor = tensor.reshape(-1)
        if tensor.shape != target.shape:
            raise ValueError(f"{name} must have shape {tuple(target.shape)}")
        target.copy_(tensor)

    def add(self, actor_vector, risk_nodes, critic_vector, pre_tanh, mode, log_prob, value, reward, next_value, terminated, truncated):
        if self.pos >= self.capacity:
            raise RuntimeError("RolloutBuffer is full, cannot add more samples")
        index = self.pos
        self._copy_sample(self.actor_vector[index], actor_vector, "actor_vector", flatten=True)
        self._copy_sample(self.risk_nodes[index], risk_nodes, "risk_nodes")
        self._copy_sample(self.critic_vector[index], critic_vector, "critic_vector", flatten=True)
        self._copy_sample(self.pre_tanh[index], pre_tanh, "pre_tanh", flatten=True)
        self._copy_sample(self.mode[index], mode, "mode", flatten=True)

        self.log_prob[index] = self._cpu_tensor(log_prob).item()
        self.value[index] = self._cpu_tensor(value).item()
        self.reward[index] = float(reward)
        self.next_value[index] = self._cpu_tensor(next_value).item()
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.pos += 1

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95, normalize=True):
        """计算优势和回报，使用 GAE（广义优势估计）方法。"""
        if self.pos == 0:
            raise RuntimeError("RolloutBuffer is empty, cannot compute advantages")
        gae = torch.tensor(0.0)

        for index in reversed(range(self.pos)):
            bootstrap = 1.0 - self.terminated[index].float()
            episode_continues = 1.0 - (self.terminated[index] | self.truncated[index]).float()

            delta = self.reward[index] + gamma * self.next_value[index] * bootstrap - self.value[index]
            gae = delta + gamma * gae_lambda * episode_continues * gae
            self.advantage[index] = gae

        self.returns[:self.pos] = self.advantage[:self.pos] + self.value[:self.pos]

        if normalize and self.pos > 1:
            advantages = self.advantage[:self.pos]
            normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            self.advantage[:self.pos] = normalized

    def batches(self, batch_size, device="cpu"):
        """生成批次数据，用于训练。"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        order = torch.randperm(self.pos)
        for start in range(0, self.pos, batch_size):
            indices = order[start:start + batch_size]
            yield {
                "actor_vector": self.actor_vector[indices].to(device),
                "risk_nodes": self.risk_nodes[indices].to(device),
                "critic_vector": self.critic_vector[indices].to(device),
                "pre_tanh": self.pre_tanh[indices].to(device),
                "mode": self.mode[indices].to(device),
                "old_log_prob": self.log_prob[indices].to(device),
                "old_value": self.value[indices].to(device),
                "advantage": self.advantage[indices].to(device),
                "returns": self.returns[indices].to(device),
            }

    def clear(self):
        """清空缓冲区，准备下一轮采样。"""
        self.pos = 0

