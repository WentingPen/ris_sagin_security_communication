import numpy as np

from system_model.channel_model import SAGIN_ChannelModel


def _positions3(positions):
    """将输入位置转换为三维坐标数组，并检查其形状和有限性。"""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions[None, :]
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (K, 2) or (K, 3)")
    if positions.shape[1] == 2:
        positions = np.c_[positions, np.zeros(len(positions))]
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite coordinates")
    return positions


def _ris_vector(value, size, name):
    """将输入值转换为复数向量，并检查其大小和有限性。"""
    value = np.asarray(value, dtype=np.complex128).reshape(-1)
    if value.size != size or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain {size} finite coefficients")
    return value


def _per_user(value, count, name):
    """将输入值转换为每个用户的数组，并检查其大小和有限性。"""
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.size == 1:
        value = np.full(count, value.item())
    if value.size != count or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be scalar or contain one finite value per user")
    return value


def _area_bounds(params):
    bounds = np.asarray(params.area_bounds, dtype=float).reshape(-1)
    if bounds.size != 4 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("area_bounds must be [x_min, x_max, y_min, y_max]")
    return bounds


def eve_parent_intensity(points, user_pos, params):
    """计算给定点的 Eve 父 PPP 强度，基于背景密度、每个用户的峰值密度和高斯衰减。"""
    points = _positions3(points)[:, :2]
    users = _positions3(user_pos)[:, :2]
    background = float(params.bg_density) 
    peaks = _per_user(params.peak_density, len(users), "peak_density")
    sigmas = _per_user(params.eve_cluster_sigma, len(users), "eve_cluster_sigma")
    if background < 0.0 or np.any(peaks < 0.0) or np.any(sigmas <= 0.0):
        raise ValueError("Eve intensities must be non-negative and sigmas positive")
    distance2 = np.sum((points[:, None, :] - users[None, :, :]) ** 2, axis=2)
    return background + np.sum(peaks[None, :] * np.exp(-distance2 / (2.0 * sigmas[None, :] ** 2)), axis=1)


def get_eve_hcpp_sampled_pos(user_pos, params, rng=None):
    """生成一个硬核点过程 (HCPP) 的 Eve 位置样本，确保每个 Eve 至少距离其他 Eve 和用户一定的最小距离。"""
    users = _positions3(user_pos)
    bounds = _area_bounds(params)
    rng = np.random.default_rng(params.seed) if rng is None else rng
    background = float(params.bg_density)
    peaks = _per_user(params.peak_density, len(users), "peak_density")
    lambda_upper = background + peaks.sum()
    if lambda_upper <= 0.0:
        raise ValueError("the parent PPP intensity must be positive")
    width, height = bounds[1] - bounds[0], bounds[3] - bounds[2]
    protect_radius = float(params.bob_protect_radius)
    hard_core_radius = float(params.eve_min_distance)
    max_tries = int(params.max_hcpp_resamples)
    if protect_radius < 0.0 or hard_core_radius < 0.0 or max_tries < 1:
        raise ValueError("invalid HCPP radius or resample count")

    for _ in range(max_tries):
        parent_count = int(rng.poisson(lambda_upper * width * height))
        if parent_count == 0:
            continue
        candidates = np.column_stack((rng.uniform(bounds[0], bounds[1], parent_count), rng.uniform(bounds[2], bounds[3], parent_count)))
        accepted = rng.random(parent_count) < eve_parent_intensity(candidates, users, params) / lambda_upper
        candidates = candidates[accepted]
        if len(candidates) == 0:
            continue
        distance2_to_users = np.sum((candidates[:, None, :] - users[None, :, :2]) ** 2, axis=2)
        candidates = candidates[np.all(distance2_to_users >= protect_radius ** 2, axis=1)]
        if len(candidates) == 0:
            continue
        marks = rng.random(len(candidates))
        pair_distance2 = np.sum((candidates[:, None, :] - candidates[None, :, :]) ** 2, axis=2)
        blocked = np.any((pair_distance2 < hard_core_radius ** 2) & (marks[None, :] < marks[:, None]), axis=1)
        kept = candidates[~blocked]
        if len(kept):
            return np.c_[kept, np.full(len(kept), float(params.ground_height))]
    raise RuntimeError("failed to generate a non-empty HCPP realization")


def split_active_eve(eve_positions, rng=None):
    """从给定的 Eve 位置中随机选择一个作为活动 Eve，并返回剩余的被动 Eve 位置。"""
    eve_positions = _positions3(eve_positions)
    if len(eve_positions) == 0:
        raise ValueError("at least one Eve is required")
    rng = np.random.default_rng() if rng is None else rng
    active_index = int(rng.integers(len(eve_positions)))
    return eve_positions[active_index].copy(), np.delete(eve_positions, active_index, axis=0), active_index


class PassiveEVE:
    def __init__(self, params, position, channel_model=None):
        self.params = params
        self.position = _positions3(position)[0]
        self.channel_model = channel_model or SAGIN_ChannelModel(params)
        self.num_sat_antennas = int(params.num_sat_antennas)
        self.num_ris_elements = int(params.num_ris_elements)
        self._h_ris = None
        self._uav_pos = None

    @property
    def ris_channel(self):
        if self._h_ris is None:
            raise RuntimeError("channel has not been generated for this step")
        return self._h_ris.copy()

    def sample_ris_channel(self, uav_pos, ris_rotation=None):
        """Generate and cache the configured platform-to-ground RIS channel."""
        uav_pos = _positions3(uav_pos)[0]
        self._h_ris = self.channel_model.get_ris_to_ground_channel(
            uav_pos,
            self.position,
            self.num_ris_elements,
            1,
            ris_rotation=ris_rotation,
        ).reshape(-1)
        self._uav_pos = uav_pos.copy()
        return self._h_ris.copy()

    def get_eve_eff_channel(
        self,
        sat_pos,
        uav_pos,
        ris_coeff,
        h_sat_ris,
        receiver_ris_channel=None,
        ris_rotation=None,
    ):
        sat_pos = _positions3(sat_pos)[0]
        uav_pos = _positions3(uav_pos)[0]
        ris_coeff = _ris_vector(ris_coeff, self.num_ris_elements, "ris_coeff")
        h_sat_ris = np.asarray(h_sat_ris, dtype=np.complex128)
        if h_sat_ris.shape != (self.num_ris_elements, self.num_sat_antennas):
            raise ValueError("h_sat_ris must have shape (N, M)")
        h_direct = self.channel_model.get_s2g_channel(sat_pos, self.position, self.num_sat_antennas, 1).reshape(-1)
        if receiver_ris_channel is None:
            self.sample_ris_channel(uav_pos, ris_rotation=ris_rotation)
        else:
            self._h_ris = _ris_vector(receiver_ris_channel, self.num_ris_elements, "receiver_ris_channel")
            self._uav_pos = uav_pos.copy()
        return h_direct + (self._h_ris * ris_coeff) @ h_sat_ris

    def get_passive_eve_jammer_noise(self, uav_pos, active_eve_pos, ris_coeff, active_tx_to_ris_channel, direct_tx_gain):
        uav_pos = _positions3(uav_pos)[0]
        active_eve_pos = _positions3(active_eve_pos)[0]
        ris_coeff = _ris_vector(ris_coeff, self.num_ris_elements, "ris_coeff")
        active_tx_to_ris_channel = _ris_vector(active_tx_to_ris_channel, self.num_ris_elements, "active_tx_to_ris_channel")

        if not np.isfinite(direct_tx_gain) or direct_tx_gain < 0.0:
            raise ValueError("direct_tx_gain must be non-negative")
        if self._h_ris is None or not np.allclose(uav_pos, self._uav_pos):
            raise RuntimeError("effective channel must first be generated at the same UAV position")

        h_direct_base = self.channel_model.get_g2g_channel(active_eve_pos, self.position)
        h_direct = h_direct_base * np.sqrt(direct_tx_gain)
        
        h_reflected = np.dot(self._h_ris * ris_coeff, active_tx_to_ris_channel)
        residual = float(self.params.passive_eve_cancel_factor)
        return float(residual * self.params.jammer_power * np.abs(h_direct + h_reflected) ** 2)

    def get_pass_eve_sinr(self, h_eff, W, active_ris_coeff, user_index, jammer_interference):
        return self._get_sinr(h_eff, W, active_ris_coeff, user_index, jammer_interference)

    def _get_sinr(self, h_eff, W, active_ris_coeff, user_index, extra_interference):
        h_eff = np.asarray(h_eff, dtype=np.complex128).reshape(-1)
        W = np.asarray(W, dtype=np.complex128)
        if W.ndim != 2 or W.shape[0] != h_eff.size:
            raise ValueError("W must have shape (M, U)")
        user_index = int(user_index)
        if not 0 <= user_index < W.shape[1]:
            raise IndexError("user_index is out of range")
        stream_power = np.abs(h_eff @ W) ** 2
        desired_power = float(stream_power[user_index])
        multiuser_interference = float(stream_power.sum() - desired_power)
        active_ris_coeff = _ris_vector(active_ris_coeff, self.num_ris_elements, "active_ris_coeff")
        ris_noise = float(self.params.ris_noise_power * np.linalg.norm(self.ris_channel * active_ris_coeff) ** 2)
        denominator = multiuser_interference + float(self.params.eve_noise_power) + float(extra_interference) + ris_noise
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("total interference-plus-noise power must be finite and positive")
        return float(desired_power / denominator)

    @staticmethod
    def get_capacity(sinr):
        sinr = float(sinr)
        if not np.isfinite(sinr) or sinr < 0.0:
            raise ValueError("sinr must be finite and non-negative")
        return float(np.log2(1.0 + sinr))


class ActiveEVE(PassiveEVE):
    def __init__(self, params, position, channel_model=None):
        super().__init__(params, position, channel_model)
        self.target_user_index = 0
        self.tx_boresight_azimuth = 0.0

    def _bearing(self, receiver_position):
        receiver_position = _positions3(receiver_position)[0]
        delta = receiver_position - self.position
        return float(np.arctan2(delta[1], delta[0]))
    
    def get_tx_gain(self, receiver_position):
        receiver_position = _positions3(receiver_position)[0]
        direction = receiver_position - self.position
        distance = np.linalg.norm(direction)
        if distance < 1e-12:
            raise ValueError("receiver_position is too close to the Eve position")
        direction /= distance
        boresight = np.array([np.cos(self.tx_boresight_azimuth), np.sin(self.tx_boresight_azimuth), 0.0])

        cosine = np.clip(np.dot(direction, boresight), -1.0, 1.0)
        off_axis_deg = float(np.degrees(np.arccos(cosine)))
        attenuation = min(
            12.0 * (off_axis_deg / float(self.params.jammer_beamwidth_deg)) ** 2,
            float(self.params.jammer_max_attenuation_db),
        )
        gain_db = float(self.params.jammer_main_gain_db) - attenuation
        return float(10.0 ** (gain_db / 10.0))


    def select_jammer_direction(
        self,
        bobs,
        passive_eves,
        user_margin=None,
        uav_pos=None,
        ris_coeff=None,
        ris_rotation=None,
    ):
        """选择干扰器的方向，以最大化对 Bob 的干扰，同时惩罚对被动 Eve 的残余泄漏."""
        if not bobs:
            raise ValueError("at least one Bob is required")
        max_offset = np.radians(float(self.params.jammer_max_steering_offset_deg))
        offsets = np.linspace(-max_offset, max_offset, int(self.params.jammer_steering_samples))
        priority = np.ones(len(bobs), dtype=float)
        if user_margin is not None:
            margin = _per_user(user_margin, len(bobs), "user_margin")
            priority = np.exp(np.clip(-margin, -50.0, 50.0))
            priority /= priority.max()
        if (uav_pos is None) != (ris_coeff is None):
            raise ValueError("uav_pos and ris_coeff must be provided together")
        reflected_scale = 0.0
        if uav_pos is not None:
            uav_pos = _positions3(uav_pos)[0]
            ris_coeff = _ris_vector(ris_coeff, self.num_ris_elements, "ris_coeff")
            active_ris_gain = self.channel_model.get_ris_to_ground_large_scale_gain(
                uav_pos,
                self.position,
                ris_rotation=ris_rotation,
            )
            reflected_scale = active_ris_gain * np.sum(np.abs(ris_coeff) ** 2) / self.channel_model.ground_rx_gain
        passive_links = [
            (
                eve.position,
                self.channel_model.get_g2g_large_scale_gain(self.position, eve.position),
                reflected_scale
                * self.channel_model.get_ris_to_ground_large_scale_gain(
                    uav_pos,
                    eve.position,
                    ris_rotation=ris_rotation,
                )
                if uav_pos is not None else 0.0,
            )
            for eve in passive_eves
        ]
        residual = float(self.params.passive_eve_cancel_factor)
        leakage_weight = float(self.params.jammer_leakage_weight)
        best_score, best_target = -np.inf, 0
        best_azimuth = self._bearing(bobs[0].position)
        for user_index, bob in enumerate(bobs):
            bob_bearing = self._bearing(bob.position)
            bob_link = self.channel_model.get_g2g_large_scale_gain(self.position, bob.position)
            bob_ris_link = (
                reflected_scale
                * self.channel_model.get_ris_to_ground_large_scale_gain(
                    uav_pos,
                    bob.position,
                    ris_rotation=ris_rotation,
                )
                if uav_pos is not None else 0.0
            )
            for offset in offsets:
                self.tx_boresight_azimuth = bob_bearing + offset
                ris_gain = self.get_tx_gain(uav_pos) if uav_pos is not None else 0.0
                desired = priority[user_index] * (self.get_tx_gain(bob.position) * bob_link + ris_gain * bob_ris_link)
                leakage = residual * np.mean([
                    self.get_tx_gain(position) * direct_link + ris_gain * ris_link
                    for position, direct_link, ris_link in passive_links
                ]) if passive_links else 0.0
                score = desired - leakage_weight * leakage
                if score > best_score:
                    best_score, best_target, best_azimuth = score, user_index, self.tx_boresight_azimuth
        self.target_user_index = best_target
        self.tx_boresight_azimuth = best_azimuth
        return best_target, best_azimuth

    def get_active_eve_sinr(self, h_eff, W, ris_coeff, active_ris_coeff, active_tx_to_ris_channel, user_index):
        """计算活动 Eve 的信噪比 (SINR)，考虑多用户干扰、RIS 噪声和来自自身发射器的残余干扰."""
        ris_coeff = _ris_vector(ris_coeff, self.num_ris_elements, "ris_coeff")
        active_tx_to_ris_channel = _ris_vector(
            active_tx_to_ris_channel, self.num_ris_elements, "active_tx_to_ris_channel"
        )
        jammer_power = float(self.params.jammer_power)
        direct_residual_si = (
            float(self.params.residual_si_factor) * jammer_power * float(self.params.self_interference_channel_gain)
        )
        echo_channel = np.dot(self.ris_channel * ris_coeff, active_tx_to_ris_channel)
        reflected_residual_si = (
            float(self.params.active_eve_echo_cancel_factor) * jammer_power * float(np.abs(echo_channel) ** 2)
        )
        return self._get_sinr(h_eff, W, active_ris_coeff, user_index, direct_residual_si + reflected_residual_si)
