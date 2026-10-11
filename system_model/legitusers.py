import numpy as np

from system_model.channel_model import SAGIN_ChannelModel


def _position3(position):
    position = np.asarray(position, dtype=float).reshape(-1)
    if position.size == 2:
        position = np.r_[position, 0.0]
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError("position must contain 2 or 3 finite coordinates")
    return position.copy()


def _ris_vector(value, size, name):
    value = np.asarray(value, dtype=np.complex128).reshape(-1)
    if value.size != size or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain {size} finite coefficients")
    return value


class LEGITUSER:
    def __init__(self, params, position, user_index, channel_model=None):
        self.params = params
        self.position = _position3(position)
        self.user_index = int(user_index)
        self.channel_model = channel_model or SAGIN_ChannelModel(params)
        self.num_sat_antennas = int(params.num_sat_antennas)
        self.num_ris_elements = int(params.num_ris_elements)
        # Common fixed environment loss used by every satellite-ground link.
        self.direct_environment_loss_db = float(
            self.channel_model.s2g_clutter_loss_db
            + self.channel_model.s2g_fixed_shadowing_loss_db
            + self.channel_model.s2g_building_penetration_loss_db
        )
        if not 0 <= self.user_index < int(params.num_users):
            raise ValueError("user_index is out of range")
        self._h_direct = None
        self._h_ris = None
        self._sat_pos = None
        self._uav_pos = None
        self._ris_rotation = None

    @property
    def direct_channel(self):
        if self._h_direct is None:
            raise RuntimeError("legitimate channel has not been sampled for this step")
        return self._h_direct.copy()

    @property
    def ris_channel(self):
        if self._h_ris is None:
            raise RuntimeError("channel has not been generated for this step")
        return self._h_ris.copy()

    def sample_channels(self, sat_pos, uav_pos, ris_rotation=None):
        """Sample and cache one legitimate-link realization for this step."""
        sat_pos = _position3(sat_pos)
        uav_pos = _position3(uav_pos)
        self._h_direct = self.channel_model.get_s2g_channel(
            sat_pos,
            self.position,
            self.num_sat_antennas,
            1,
        ).reshape(-1)
        self._h_ris = self.channel_model.get_ris_to_ground_channel(
            uav_pos,
            self.position,
            self.num_ris_elements,
            1,
            ris_rotation=ris_rotation,
        ).reshape(-1)
        self._sat_pos = sat_pos.copy()
        self._uav_pos = uav_pos.copy()
        self._ris_rotation = (
            None
            if ris_rotation is None
            else np.asarray(ris_rotation, dtype=float).copy()
        )
        return self._h_direct.copy(), self._h_ris.copy()

    def get_effective_channel(self, sat_pos, uav_pos, ris_coeff, h_sat_ris, ris_rotation=None):
        sat_pos = _position3(sat_pos)
        uav_pos = _position3(uav_pos)
        ris_coeff = _ris_vector(ris_coeff, self.num_ris_elements, "ris_coeff")
        h_sat_ris = np.asarray(h_sat_ris, dtype=np.complex128)
        if h_sat_ris.shape != (self.num_ris_elements, self.num_sat_antennas):
            raise ValueError("h_sat_ris must have shape (N, M)")
        if self._h_direct is None or self._h_ris is None:
            raise RuntimeError(
                "sample_channels() must be called before get_effective_channel()"
            )
        if not np.allclose(sat_pos, self._sat_pos) or not np.allclose(uav_pos, self._uav_pos):
            raise RuntimeError("cached legitimate channel does not match the current geometry")
        rotation = (
            None
            if ris_rotation is None
            else np.asarray(ris_rotation, dtype=float)
        )
        if (rotation is None) != (self._ris_rotation is None) or (
            rotation is not None and not np.allclose(rotation, self._ris_rotation)
        ):
            raise RuntimeError("cached legitimate channel does not match the current RIS rotation")
        return self._h_direct + (self._h_ris * ris_coeff) @ h_sat_ris

    def get_bob_jammer_noise(self, uav_pos, active_eve_pos, ris_coeff, active_tx_to_ris_channel, direct_tx_gain,):
        """计算 Bob 接收器的干扰噪声功率，来自active Eve 的直接和 RIS 反射路径。"""
        uav_pos = _position3(uav_pos)
        active_eve_pos = _position3(active_eve_pos)
        ris_coeff = _ris_vector(ris_coeff, self.num_ris_elements, "ris_coeff")
        active_tx_to_ris_channel = _ris_vector(active_tx_to_ris_channel, self.num_ris_elements, "active_tx_to_ris_channel")

        if not np.isfinite(direct_tx_gain) or direct_tx_gain < 0.0:
            raise ValueError("direct_tx_gain must be non-negative")

        if self._h_ris is None or not np.allclose(uav_pos, self._uav_pos):
            raise RuntimeError("effective channel must first be generated at the same UAV position")

        h_direct_base = self.channel_model.get_g2g_channel(active_eve_pos, self.position)
        h_direct = np.sqrt(direct_tx_gain) * h_direct_base

        h_reflected = np.dot(self._h_ris * ris_coeff, active_tx_to_ris_channel)

        return float(self.params.jammer_power * np.abs(h_direct + h_reflected) ** 2)


    def get_bob_sinr(self, h_eff, W, active_ris_coeff=None, jammer_interference=0.0):
        """计算 Bob 接收器的信噪比 (SINR)，考虑多用户干扰、RIS 噪声和来自active Eve 的干扰。"""
        h_eff = np.asarray(h_eff, dtype=np.complex128).reshape(-1)
        W = np.asarray(W, dtype=np.complex128)
        if W.ndim != 2 or W.shape[0] != h_eff.size:
            raise ValueError("W must have shape (M, U)")
        if not 0 <= self.user_index < W.shape[1]:
            raise IndexError("user_index is out of range")
        stream_power = np.abs(h_eff @ W) ** 2
        desired_power = float(stream_power[self.user_index])
        multiuser_interference = float(stream_power.sum() - desired_power)
        ris_noise = 0.0
        if active_ris_coeff is not None:
            active_ris_coeff = _ris_vector(active_ris_coeff, self.num_ris_elements, "active_ris_coeff")
            ris_noise = float(self.params.ris_noise_power * np.linalg.norm(self.ris_channel * active_ris_coeff) ** 2)
        denominator = multiuser_interference + ris_noise + float(self.params.bob_noise_power) + float(jammer_interference)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("total interference-plus-noise power must be finite and positive")
        return float(desired_power / denominator)

    @staticmethod
    def get_capacity(sinr):
        sinr = float(sinr)
        if not np.isfinite(sinr) or sinr < 0.0:
            raise ValueError("sinr must be finite and non-negative")
        return float(np.log2(1.0 + sinr))
