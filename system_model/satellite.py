import numpy as np


def _position3(position):
    position = np.asarray(position, dtype=float).reshape(-1)
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError("satellite position must contain 3 finite coordinates")
    return position


class SATELLITE:
    def __init__(self, params, position):
        self.position = _position3(position)
        self.num_sat_antennas = int(params.num_sat_antennas)
        self.num_users = int(params.num_users)
        self.max_power = float(params.sat_max_power)
        if self.num_sat_antennas <= 0 or self.num_users <= 0 or not np.isfinite(self.max_power) or self.max_power <= 0:
            raise ValueError("satellite dimensions and maximum power must be positive")
        shape = (self.num_sat_antennas, self.num_users)
        self.amplitude = np.zeros(shape, dtype=float)
        self.phase = np.zeros(shape, dtype=float)
        self.W = np.zeros(shape, dtype=np.complex128)

    def set_mrt_beam(self, channel):
        """Configure a full-power single-user MRT beam for y = h @ w."""
        if self.num_users != 1:
            raise NotImplementedError(
                "the current MRT/RIS base design supports one user only"
            )
        channel = np.asarray(channel, dtype=np.complex128).reshape(-1)
        if channel.size != self.num_sat_antennas or not np.all(np.isfinite(channel)):
            raise ValueError("channel must contain num_sat_antennas finite values")
        norm = np.linalg.norm(channel)
        if norm <= 1e-15:
            raise ValueError("MRT channel norm is too small")
        beam = np.sqrt(self.max_power) * np.conj(channel) / norm
        self.W = beam[:, None]
        self.phase = np.angle(self.W)
        self.amplitude = np.abs(self.W)
        return self.W.copy()

    @property
    def transmit_power(self):
        """返回卫星的发射功率。"""
        return float(np.vdot(self.W, self.W).real)
        

    def transmit(self, symbols):
        """将给定的符号向量通过卫星的发射矩阵 W 进行传输。"""
        symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        if symbols.size != self.num_users or not np.all(np.isfinite(symbols)):
            raise ValueError(f"symbols must contain {self.num_users} finite values")
        return self.W @ symbols
