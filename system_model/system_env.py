from collections.abc import Mapping
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from system_model.channel_model import SAGIN_ChannelModel
from system_model.hcpp_eve import ActiveEVE, PassiveEVE, eve_parent_intensity, get_eve_hcpp_sampled_pos, split_active_eve
from system_model.legitusers import LEGITUSER
from system_model.satellite import SATELLITE
from system_model.uav_ris import RISPowerInfeasibleError, UAV_RIS


def _positions3(value, name, default_height=0.0):
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (K, 2) or (K, 3)")
    if value.shape[1] == 2:
        value = np.c_[value, np.full(len(value), float(default_height))]
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value.copy()


def _per_user(value, count, name):
    """函数将标量或数组值转换为每个用户的数组，并验证其大小和有限性。"""
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.size == 1:
        value = np.full(count, value.item())
    if value.size != count or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be scalar or contain {count} finite values")
    return value


def _action_array(value, shape, name):
    """将给定的动作值转换为指定形状的数组，并验证其有限性。"""
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if value.shape == shape:
        return value.copy()
    if value.ndim == 1 and value.size == int(np.prod(shape)):
        return value.reshape(shape).copy()
    raise ValueError(f"{name} must have shape {shape} or be a flat array with {int(np.prod(shape))} values")


def _watts_to_dbm(power):
    power = np.asarray(power, dtype=float)
    return 10.0 * np.log10(np.maximum(power, 1e-300)) + 30.0


def  _feature_slices(block_dims, start=0):
    """将特征块的维度映射为切片对象，以便在向量中提取每个块."""
    slices = {}
    offset = int(start)
    for name, size in block_dims.items():
        size = int(size)
        if size <= 0:
            raise ValueError(f"feature block {name} must have positive size")
        slices[name] = slice(offset, offset + size)
        offset += size
    return slices, offset


def _assemble_features(blocks, expected_dims, name):
    """将特征块组装成一个单一的特征向量."""
    if tuple(blocks) != tuple(expected_dims):
        raise RuntimeError(f"{name} feature blocks do not match the declared layout")
    arrays = []
    for block_name, expected_size in expected_dims.items():
        block = np.asarray(blocks[block_name], dtype=float).reshape(-1)
        if block.size != expected_size:
            raise RuntimeError(
                f"{name} block {block_name} has size {block.size}, expected {expected_size}"
            )
        if not np.all(np.isfinite(block)):
            raise RuntimeError(f"{name} block {block_name} contains non-finite values")
        arrays.append(block)
    return np.concatenate(arrays)


class SAGIN_ENV(gym.Env):
    metadata = {"render_modes": []}
    ACTION_KEYS = {
        "sat_beam_real",
        "sat_beam_imag",
        "ris_element_mode",
        "ris_phase_real",
        "ris_phase_imag",
    }

    def __init__(self, params, sat_positions, uav_positions, user_positions):
        super().__init__()
        self.params = params
        self.num_users = int(params.num_users)
        if self.num_users != 1:
            raise NotImplementedError(
                "the current dual-agent beamforming model supports num_users=1 only"
            )
        sat_positions = _positions3(sat_positions, "sat_positions")
        uav_positions = _positions3(uav_positions, "uav_positions", params.uav_height)

        self.user_positions = _positions3(user_positions, "user_positions", params.ground_height)
        if not np.allclose(sat_positions[:, 2], float(params.satellite_height)):
            raise ValueError("all satellite positions must use satellite_height")
        if not np.allclose(uav_positions[:, 2], float(params.uav_height)):
            raise ValueError("all UAV positions must use the fixed uav_height")
        if not np.allclose(self.user_positions[:, 2], float(params.ground_height)):
            raise ValueError("all user positions must use the fixed ground_height")
        if len(sat_positions) != 1 or len(uav_positions) != 1:
            raise ValueError("the current model requires exactly one satellite and one UAV-RIS")
        if len(self.user_positions) != self.num_users:
            raise ValueError("len(user_positions) must equal num_users")
        
        self.initial_sat_position = sat_positions[0].copy()
        self.initial_uav_position = uav_positions[0].copy()
        self.threshold = _per_user(params.secrecy_rate_threshold, self.num_users, "secrecy_rate_threshold")

        if np.any(self.threshold <= 0.0):
            raise ValueError("secrecy_rate_threshold must be positive")
        self.bounds = np.asarray(params.area_bounds, dtype=float).reshape(-1)
        if self.bounds.size != 4 or self.bounds[0] >= self.bounds[1] or self.bounds[2] >= self.bounds[3]:
            raise ValueError("area_bounds must be [x_min, x_max, y_min, y_max]")
        expected_center = np.array([
            0.5 * (self.bounds[0] + self.bounds[1]),
            0.5 * (self.bounds[2] + self.bounds[3]),
        ])
        expected_haps_xy = np.array([
            float(params.ris_fixed_x),
            float(params.ris_fixed_y),
        ])
        if not np.allclose(sat_positions[0, :2], expected_center):
            raise ValueError("the satellite must be fixed at the area center")
        if not np.allclose(uav_positions[0, :2], expected_haps_xy):
            raise ValueError("the HAPS-RIS must use the configured fixed horizontal position")
        inside_x = (self.bounds[0] <= self.user_positions[:, 0]) & (self.user_positions[:, 0] <= self.bounds[1])
        inside_y = (self.bounds[2] <= self.user_positions[:, 1]) & (self.user_positions[:, 1] <= self.bounds[3])
        if not np.all(inside_x & inside_y):
            raise ValueError("all user positions must lie inside area_bounds")
        self.horizon = int(params.max_episode_steps)
        if self.horizon <= 0:
            raise ValueError("max_episode_steps must be positive")
        self._seed_sequence = None
        self._ready = False
        M, U, N = int(params.num_sat_antennas), self.num_users, int(params.num_ris_elements)
        S = int(params.risk_grid_size)
        self.actor_feature_dims = {     #表示actor网络输入每个特征块的维度
            "time": 1,
            "position": 2 * U + 6,
            "channel": 4 * U + 2,
            "instant_csi": 2 * (M * U + N * M + N * U),
            "ris": 4 * N,
            "beam": 2 * M * U,
            "feedback": 3 * U + 3,
        }
        self.critic_private_feature_dims = {    #表示critic网络输入每个特征块的维度
            "eve_user_metrics": 4 * U,
            "system_metrics": 2,
            "active_eve_xy": 2,
            "jammer_target": U,
            "jammer_heading": 2,
            "passive_eve_count": 1,
        }
        #critic的输入特征向量由actor的特征向量和critic的私有特征向量组成
        self.actor_feature_slices, self.actor_vector_dim = _feature_slices(self.actor_feature_dims)
        self.critic_private_feature_slices, critic_end = _feature_slices(self.critic_private_feature_dims, start=self.actor_vector_dim)
        self.critic_vector_dim = critic_end

        #定义action_space和observation_space，
        #spaces.box表示连续动作空间，spaces.MultiBinary表示离散动作空间，spaces.Dict表示字典类型的空间
        #表示每个时隙都必须选择全部动作、全部观测，不能只选择部分
        self.action_space = spaces.Dict({
            "sat_beam_real": spaces.Box(-1.0, 1.0, shape=(M,), dtype=np.float32),
            "sat_beam_imag": spaces.Box(-1.0, 1.0, shape=(M,), dtype=np.float32),
            "ris_element_mode": spaces.MultiBinary(N),
            "ris_phase_real": spaces.Box(-1.0, 1.0, shape=(N,), dtype=np.float32),
            "ris_phase_imag": spaces.Box(-1.0, 1.0, shape=(N,), dtype=np.float32),
        })
        risk_nodes_space = spaces.Box(0.0, 1.0, shape=(S * S, 3), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(-1.0, 1.0, shape=(self.actor_vector_dim,), dtype=np.float32),
            "risk_nodes": risk_nodes_space,
            "edge_index": spaces.Box(0, S * S - 1, shape=(2, 4 * S * (S - 1)), dtype=np.int64),
        })
        self.critic_vector_space = spaces.Box(-1.0, 1.0, shape=(self.critic_vector_dim,), dtype=np.float32)
        self.critic_observation_space = spaces.Dict({
            "vector": self.critic_vector_space,
            "risk_nodes": risk_nodes_space,
        })

    def reset(self, *, seed=None, options=None):
        """重置环境状态，采样新的卫星、UAV-RIS和用户位置，并初始化信道模型和相关参数。"""
        super().reset(seed=seed)
        self._ready = False
        if seed is not None or self._seed_sequence is None:
            self._seed_sequence = np.random.SeedSequence(self.params.seed if seed is None else seed)
        geometry_seed, selection_seed, channel_seed = self._seed_sequence.spawn(3)
        self.geometry_rng = np.random.default_rng(geometry_seed)
        self.selection_rng = np.random.default_rng(selection_seed)
        self.channel_rng = np.random.default_rng(channel_seed)
        self.channel_model = SAGIN_ChannelModel(self.params, rng=self.channel_rng)
        self.satellite = SATELLITE(self.params, self.initial_sat_position)
        self.uav_ris = UAV_RIS(self.params, self.initial_uav_position)
        self.bobs = [LEGITUSER(self.params, position, index, self.channel_model) for index, position in enumerate(self.user_positions)]
        sampled_eves = get_eve_hcpp_sampled_pos(self.user_positions, self.params, self.geometry_rng)
        active_position, passive_positions, self.active_eve_index = split_active_eve(sampled_eves, self.selection_rng)
        self.active_eve = ActiveEVE(self.params, active_position, self.channel_model)
        self.passive_eves = [PassiveEVE(self.params, position, self.channel_model) for position in passive_positions]
        self.eve_positions = np.vstack((active_position[None, :], passive_positions))
        self._refresh_common_channels()
        self._refresh_legitimate_channels()
        self.satellite.set_mrt_beam(self.bobs[0].direct_channel)
        self.uav_ris.configure_elements(
            element_mode=np.zeros(self.uav_ris.num_elements, dtype=float),
            phase_real=np.ones(self.uav_ris.num_elements, dtype=float),
            phase_imag=np.zeros(self.uav_ris.num_elements, dtype=float),
        )
        self._update_active_eve_transmission()
        self.ris_power = 0.0
        self.ris_action_repaired = False
        self.jammer_ris_correction_applied = False
        self.risk_nodes, self.edge_index = self._build_risk_graph()
        self.step_count = 0
        self.outage_count = 0
        self.metrics = self._evaluate()
        self._add_costs()
        observation, info = self._get_actor_observation(), self._get_info()
        self._ready = True
        return observation, info

    def step(self, action):
        """执行给定的动作，更新环境状态，并返回新的观测、奖励、终止标志和信息."""
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        if self.step_count >= self.horizon:
            raise RuntimeError("episode is finished; call reset()")
        action = self._validate_action(action)
        try:
            self._apply_action(action)
            self.metrics = self._evaluate()
            self._add_costs()
            reward = self._get_reward()
            self.step_count += 1
            self.outage_count += int(self.metrics["outage_sample"])
            terminated = False
            truncated = self.step_count >= self.horizon
            # The action is evaluated on the CSI contained in the current
            # observation.  Only then is the next slot's CSI sampled.  The
            # time-limit observation also needs H_(t+1), because PPO
            # bootstraps transitions truncated by the finite horizon.
            self._refresh_common_channels()
            self._refresh_legitimate_channels()
            observation, info = self._get_actor_observation(), self._get_info()
        except Exception:
            self._ready = False
            raise
        return observation, reward, terminated, truncated, info

    def get_phase_only_reference(self):
        """Return the normalized MRT action and Bob-coherent RIS base phase."""
        if not self._ready:
            raise RuntimeError("call reset() before requesting a phase reference")

        direct_channel = self.bobs[0].direct_channel
        channel_norm = float(np.linalg.norm(direct_channel))
        if channel_norm <= 1e-15:
            raise RuntimeError("the direct Bob channel is too small for MRT")

        unit_beam = np.conj(direct_channel) / channel_norm
        satellite_continuous = np.r_[unit_beam.real, unit_beam.imag].astype(np.float32)
        return satellite_continuous, self.get_ris_phase_reference(satellite_continuous)

    def get_ris_phase_reference(self, satellite_continuous):
        """Align RIS elements with Bob's direct signal for the executed beam."""
        if not self._ready:
            raise RuntimeError("call reset() before requesting a phase reference")

        num_antennas = self.satellite.num_sat_antennas
        satellite_continuous = _action_array(
            satellite_continuous, (2 * num_antennas,), "satellite_continuous"
        )
        if np.any(np.abs(satellite_continuous) > 1.0):
            raise ValueError("satellite_continuous must lie in [-1, 1]")
        unit_beam = satellite_continuous[:num_antennas] + 1j * satellite_continuous[num_antennas:]
        beam = np.sqrt(self.satellite.max_power) * unit_beam / max(float(np.linalg.norm(unit_beam)), 1.0)
        direct_signal = self.bobs[0].direct_channel @ beam
        reflected_terms = self.bobs[0].ris_channel * (
            self.h_sat_ris @ beam
        )
        base_phase = np.zeros(self.uav_ris.num_elements, dtype=float)
        nonzero = np.abs(reflected_terms) > 1e-30
        base_phase[nonzero] = (
            np.angle(direct_signal) - np.angle(reflected_terms[nonzero])
        )
        base_phase = (base_phase + np.pi) % (2.0 * np.pi) - np.pi
        return base_phase

    def _validate_action(self, action):
        """验证给定的动作字典是否包含所有必需的键，并将其转换为适当的数组形状."""
        if not isinstance(action, Mapping): #mapping是一个抽象基类，表示键值对的集合，类似于字典
            raise TypeError("action must be a mapping")
        missing = self.ACTION_KEYS.difference(action)
        if missing:
            raise KeyError(f"action is missing keys: {sorted(missing)}")
        extra = set(action).difference(self.ACTION_KEYS)
        if extra:
            raise KeyError(f"action contains unexpected keys: {sorted(extra)}")
        M = self.satellite.num_sat_antennas
        N = self.uav_ris.num_elements
        validated = {
            "sat_beam_real": _action_array(
                action["sat_beam_real"], (M,), "sat_beam_real"
            ),
            "sat_beam_imag": _action_array(
                action["sat_beam_imag"], (M,), "sat_beam_imag"
            ),
            "ris_element_mode": _action_array(
                action["ris_element_mode"], (N,), "ris_element_mode"
            ),
            "ris_phase_real": _action_array(
                action["ris_phase_real"], (N,), "ris_phase_real"
            ),
            "ris_phase_imag": _action_array(
                action["ris_phase_imag"], (N,), "ris_phase_imag"
            ),
        }
        if not np.all(np.isin(validated["ris_element_mode"], (0.0, 1.0))):
            raise ValueError("ris_element_mode must contain only 0 or 1")
        for name in (
            "sat_beam_real",
            "sat_beam_imag",
            "ris_phase_real",
            "ris_phase_imag",
        ):
            if np.any(validated[name] < -1.0) or np.any(validated[name] > 1.0):
                raise ValueError(f"{name} must lie in [-1, 1]")
        return validated

    def _apply_action(self, action):
        previous_margin = self.metrics["user_margin"].copy()
        self.satellite.set_beamforming(
            action["sat_beam_real"], action["sat_beam_imag"]
        )
        self.uav_ris.configure_elements(
            element_mode=action["ris_element_mode"],
            phase_real=action["ris_phase_real"],
            phase_imag=action["ris_phase_imag"],
        )
        self._update_active_eve_transmission(previous_margin)

        first_power, first_repaired = self._enforce_current_ris_power(action)
        self.jammer_ris_correction_applied = bool(first_repaired)

        if first_repaired:
            # One bounded correction: keep this step's fading realization, update
            # only the jammer direction, then project again from the requested RIS.
            self._update_active_eve_transmission(previous_margin)
            self.uav_ris.configure_elements(
                element_mode=action["ris_element_mode"],
                phase_real=action["ris_phase_real"],
                phase_imag=action["ris_phase_imag"],
            )
            self.ris_power, self.ris_action_repaired = (
                self._enforce_current_ris_power(action)
            )
        else:
            self.ris_power = first_power
            self.ris_action_repaired = False

    def _enforce_current_ris_power(self, action):
        """确保当前激活的 RIS 元素满足最大功率约束，如果不满足，则按比例缩放幅度。"""
        try:
            power = self.uav_ris.enforce_power_constraint(
                self.h_sat_ris, self.satellite.W, self.active_tx_to_ris_channel
            )
            repaired = self.uav_ris.last_power_projection_scale < 1.0 - 1e-12
        except RISPowerInfeasibleError:
            self.uav_ris.configure_elements(
                element_mode=np.zeros(self.uav_ris.num_elements, dtype=float),
                phase_real=action["ris_phase_real"],
                phase_imag=action["ris_phase_imag"],
            )
            power = 0.0
            repaired = True
        return float(power), bool(repaired)

    def _refresh_common_channels(self):
        self.h_sat_ris = self.channel_model.get_s2a_channel(
            self.satellite.position,
            self.uav_ris.position,
            self.satellite.num_sat_antennas,
            self.uav_ris.num_elements,
            ris_rotation=self.uav_ris.rotation_matrix,
        )
        self.active_ris_rx_channel = self.active_eve.sample_ris_channel(
            self.uav_ris.position,
            ris_rotation=self.uav_ris.rotation_matrix,
        )
        for eve in self.passive_eves:
            eve.sample_ris_channel(
                self.uav_ris.position,
                ris_rotation=self.uav_ris.rotation_matrix,
            )
        # Reciprocal propagation is shared; the RX gain is removed before applying the directional TX gain.
        self.active_tx_to_ris_base_channel = (
            self.active_ris_rx_channel / np.sqrt(self.channel_model.ground_rx_gain)
        )

    def _refresh_legitimate_channels(self):
        for bob in self.bobs:
            bob.sample_channels(
                sat_pos=self.satellite.position,
                uav_pos=self.uav_ris.position,
                ris_rotation=self.uav_ris.rotation_matrix,
            )

    def _update_active_eve_transmission(self, user_margin=None):
        """选择active Eve的发射方向，以最大化对Bob的干扰，同时惩罚对被动Eve的残余泄漏."""
        self.active_eve.select_jammer_direction(
            self.bobs,
            self.passive_eves,
            user_margin,
            self.uav_ris.position,
            self.uav_ris.ris_coeff,
            ris_rotation=self.uav_ris.rotation_matrix,
        )
        tx_gain_to_ris = self.active_eve.get_tx_gain(self.uav_ris.position)
        self.active_tx_to_ris_channel = np.sqrt(tx_gain_to_ris) * self.active_tx_to_ris_base_channel

    def _evaluate(self):
        """计算当前系统状态下的所有用户的信噪比、容量和安全性指标."""
        ris_coeff = self.uav_ris.ris_coeff  #ris_coeff是一个复数向量，表示RIS的反射系数，包括幅度和相位信息
        active_coeff = self.uav_ris.active_ris_coeff  #active_coeff表示长度为N的反射向量，但是Passive阵子元素为0
        sat_pos, uav_pos = self.satellite.position, self.uav_ris.position
        rotation = self.uav_ris.rotation_matrix #rotation_matrix是RIS的三维旋转矩阵，由 RIS 当前 yaw 和 pitch 决定
        bob_channels = [
            bob.get_effective_channel(
                sat_pos,
                uav_pos,
                ris_coeff,
                self.h_sat_ris,
                ris_rotation=rotation,
            )
            for bob in self.bobs
        ]
        bob_direct_signal = np.asarray([
            bob.direct_channel @ self.satellite.W[:, index]
            for index, bob in enumerate(self.bobs)
        ])
        bob_ris_signal = np.asarray([
            ((bob.ris_channel * ris_coeff) @ self.h_sat_ris)
            @ self.satellite.W[:, index]
            for index, bob in enumerate(self.bobs)
        ])
        bob_coherent_signal = bob_direct_signal + bob_ris_signal
        bob_direct_rx_power_dbm = _watts_to_dbm(np.abs(bob_direct_signal) ** 2)
        bob_ris_rx_power_dbm = _watts_to_dbm(np.abs(bob_ris_signal) ** 2)
        bob_coherent_rx_power_dbm = _watts_to_dbm(np.abs(bob_coherent_signal) ** 2)
        bob_ris_to_direct_power_ratio_db = (
            bob_ris_rx_power_dbm - bob_direct_rx_power_dbm
        )
        active_channel = self.active_eve.get_eve_eff_channel(
            sat_pos,
            uav_pos,
            ris_coeff,
            self.h_sat_ris,
            self.active_ris_rx_channel,
            ris_rotation=rotation,
        )
        passive_channels = [
            eve.get_eve_eff_channel(
                sat_pos,
                uav_pos,
                ris_coeff,
                self.h_sat_ris,
                receiver_ris_channel=eve.ris_channel,
                ris_rotation=rotation,
            )
            for eve in self.passive_eves
        ]
        bob_jammer = np.asarray(
            [bob.get_bob_jammer_noise(uav_pos, self.active_eve.position, ris_coeff, self.active_tx_to_ris_channel, self.active_eve.get_tx_gain(bob.position)) for bob in self.bobs], dtype=float
        )
        bob_ris_noise = np.asarray(
            [self.params.ris_noise_power * np.linalg.norm(bob.ris_channel * active_coeff) ** 2 for bob in self.bobs], dtype=float
        )
        bob_jammer_inr = bob_jammer / (float(self.params.bob_noise_power) + bob_ris_noise)
        passive_jammer = np.asarray(
            [eve.get_passive_eve_jammer_noise(uav_pos, self.active_eve.position, ris_coeff, self.active_tx_to_ris_channel, self.active_eve.get_tx_gain(eve.position)) for eve in self.passive_eves], dtype=float
        )
        bob_sinr = np.asarray(
            [bob.get_bob_sinr(bob_channels[index], self.satellite.W, active_coeff, bob_jammer[index]) for index, bob in enumerate(self.bobs)], dtype=float
        )
        active_sinr = np.asarray(
            [self.active_eve.get_active_eve_sinr(active_channel, self.satellite.W, ris_coeff, active_coeff, self.active_tx_to_ris_channel, user_index) for user_index in range(self.num_users)], dtype=float
        )
        passive_sinr = np.empty((len(self.passive_eves), self.num_users), dtype=float)
        for eve_index, eve in enumerate(self.passive_eves):
            for user_index in range(self.num_users):
                passive_sinr[eve_index, user_index] = eve.get_pass_eve_sinr(
                    passive_channels[eve_index], self.satellite.W, active_coeff, user_index, passive_jammer[eve_index]
                )
        eve_sinr = np.concatenate((active_sinr[None, :], passive_sinr), axis=0)
        bob_capacity = np.log2(1.0 + bob_sinr)
        eve_capacity = np.log2(1.0 + eve_sinr)
        max_eve_capacity = eve_capacity.max(axis=0)
        max_eve_sinr = eve_sinr.max(axis=0)
        raw_security_gap = bob_capacity - max_eve_capacity
        secrecy_rate = np.maximum(raw_security_gap, 0.0)
        user_margin = raw_security_gap - self.threshold
        user_outage = secrecy_rate < self.threshold
        system_margin = float(user_margin.min())
        outage_sample = bool(np.any(user_outage))
        all_eve_channels = [active_channel, *passive_channels]  #*表示将元素解包单独传入列表中，形成一个新的列表
        return {
            "bob_sinr": bob_sinr,
            "bob_direct_rx_power_dbm": bob_direct_rx_power_dbm,
            "bob_ris_rx_power_dbm": bob_ris_rx_power_dbm,
            "bob_coherent_rx_power_dbm": bob_coherent_rx_power_dbm,
            "bob_ris_to_direct_power_ratio_db": bob_ris_to_direct_power_ratio_db,
            "bob_jammer_power": bob_jammer,
            "bob_ris_noise_power": bob_ris_noise,
            "bob_jammer_inr": bob_jammer_inr,
            "active_eve_sinr": active_sinr,
            "passive_eve_sinr": passive_sinr,
            "eve_sinr": eve_sinr,
            "max_eve_sinr": max_eve_sinr,
            "bob_capacity": bob_capacity,
            "eve_capacity": eve_capacity,
            "max_eve_capacity": max_eve_capacity,
            "raw_security_gap": raw_security_gap,
            "secrecy_rate": secrecy_rate,
            "user_margin": user_margin,
            "user_outage": user_outage,
            "system_margin": system_margin,
            "outage_sample": outage_sample,
            "outage": outage_sample,
            "bob_effective_channel": np.vstack(bob_channels),
            "eve_effective_channel": np.vstack(all_eve_channels),
        }

    def _add_costs(self):
        """计算并更新与卫星功率、RIS功率和UAV移动相关的成本指标."""
        sat_ratio = self.satellite.transmit_power / self.satellite.max_power
        ris_ratio = self.ris_power / self.uav_ris.max_power
        move_ratio = np.linalg.norm(self.uav_ris.last_displacement) / max(self.uav_ris.max_step, 1e-12)
        self.metrics.update(
            sat_power_ratio=float(sat_ratio),
            ris_power_ratio=float(ris_ratio),
            move_ratio=float(move_ratio),
            orientation_move_ratio=float(self.uav_ris.orientation_move_ratio),
        )

    def _get_reward(self):
        m = self.metrics
        normalized_margin = np.clip(
            m["system_margin"] / float(np.min(self.threshold)),
            -2.0,
            5.0,
        )
        return float(
            -self.params.reward_outage_weight * float(m["outage_sample"])
            + self.params.reward_margin_weight * normalized_margin
            - self.params.reward_sat_power_weight * m["sat_power_ratio"]
            - self.params.reward_ris_power_weight * m["ris_power_ratio"]
            - self.params.reward_move_weight * m["move_ratio"] ** 2
            - self.params.reward_orientation_weight * m["orientation_move_ratio"]
        )

    def _get_actor_observation(self):
        return {
            "vector": self._get_actor_vector(),
            "risk_nodes": self.risk_nodes.copy(),
            "edge_index": self.edge_index.copy(),
        }

    def _get_observation(self):
        return self._get_actor_observation()

    def _get_critic_observation(self):
        return {
            "vector": self._get_critic_vector(),
            "risk_nodes": self.risk_nodes.copy(),
        }

    def _get_actor_vector(self):
        """计算actor网络的输入特征向量，包括时间、位置、方向、信道、RIS状态、波束和反馈信息."""
        m = self.metrics
        x_min, x_max, y_min, y_max = self.bounds
        ground_xy = np.vstack((self.uav_ris.position[:2], self.user_positions[:, :2]))  #vstask表示纵向堆叠
        normalized_xy = 2.0 * (ground_xy - [x_min, y_min]) / [x_max - x_min, y_max - y_min] - 1.0
        sat_scale = float(self.params.satellite_position_scale)
        position_state = np.r_[np.clip(self.satellite.position / sat_scale, -1.0, 1.0), normalized_xy.ravel(), self.uav_ris.position[2] / self.params.uav_height]
        channel_state = self._large_scale_channel_state()
        instant_csi_state = self._instantaneous_csi_state()
        amplitude_scale = max(self.uav_ris.max_amplitude - 1.0, 1e-12)
        ris_state = np.r_[  #np.r_[]表示将多个数组沿着第一个轴进行拼接，形成一个新的数组
            self.uav_ris.element_mode, (self.uav_ris.amplitude - 1.0) / amplitude_scale,
            np.sin(self.uav_ris.phase), np.cos(self.uav_ris.phase)
        ]
        beam_scale = np.sqrt(self.satellite.max_power)
        beam_state = np.r_[(self.satellite.W.real / beam_scale).ravel(), (self.satellite.W.imag / beam_scale).ravel()]
        feedback_state = np.r_[
            self._normalize_db(m["bob_sinr"], self.params.sinr_db_min, self.params.sinr_db_max),
            self._normalize_db(m["bob_jammer_inr"], self.params.inr_db_min, self.params.inr_db_max),
            self.threshold / (1.0 + self.threshold), m["sat_power_ratio"],
            m["ris_power_ratio"], float(self.ris_action_repaired),
        ]
        # 时间状态
        time_state = np.asarray([self.step_count / self.horizon], dtype=float)
        blocks = {
            "time": time_state,
            "position": position_state,
            "channel": channel_state,
            "instant_csi": instant_csi_state,
            "ris": ris_state,
            "beam": beam_state,
            "feedback": feedback_state,
        }
        vector = _assemble_features(blocks, self.actor_feature_dims, "actor")
        return np.clip(vector, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _normalize_complex_block(value):
        value = np.asarray(value, dtype=np.complex128).reshape(-1)
        if not np.all(np.isfinite(value)):
            raise RuntimeError("instantaneous CSI contains non-finite values")
        norm = float(np.linalg.norm(value))
        if norm <= 1e-30:
            normalized = np.zeros_like(value)
        else:
            normalized = value / norm
        return np.r_[normalized.real, normalized.imag]

    def _instantaneous_csi_state(self):
        """Return normalized observable CSI for the current fading slot."""
        direct = np.vstack([bob.direct_channel for bob in self.bobs])
        ris_to_bob = np.vstack([bob.ris_channel for bob in self.bobs])
        return np.concatenate((
            self._normalize_complex_block(direct),
            self._normalize_complex_block(self.h_sat_ris),
            self._normalize_complex_block(ris_to_bob),
        ))

    def _get_critic_vector(self):
        m = self.metrics
        x_min, x_max, y_min, y_max = self.bounds
        active_xy = 2.0 * (self.active_eve.position[:2] - [x_min, y_min]) / [x_max - x_min, y_max - y_min] - 1.0
        target = np.zeros(self.num_users, dtype=float)
        target[int(self.active_eve.target_user_index)] = 1.0
        private_blocks = {
            "eve_user_metrics": np.r_[
                self._normalize_db(
                    m["max_eve_sinr"], self.params.sinr_db_min, self.params.sinr_db_max
                ),
                m["secrecy_rate"] / (1.0 + m["secrecy_rate"]),
                np.clip(m["user_margin"] / self.threshold, -1.0, 1.0),
                m["user_outage"].astype(float),
            ],
            "system_metrics": np.asarray([
                np.clip(m["system_margin"] / np.min(self.threshold), -1.0, 1.0),
                float(m["outage_sample"]),
            ]),
            "active_eve_xy": np.clip(active_xy, -1.0, 1.0),
            "jammer_target": target,
            "jammer_heading": np.asarray([
                np.sin(self.active_eve.tx_boresight_azimuth),
                np.cos(self.active_eve.tx_boresight_azimuth),
            ]),
            "passive_eve_count": np.asarray([
                len(self.passive_eves) / (1.0 + len(self.passive_eves))
            ]),
        }
        private_state = _assemble_features(
            private_blocks, self.critic_private_feature_dims, "critic private"
        )
        vector = np.concatenate((self._get_actor_vector(), private_state))
        if vector.size != self.critic_vector_dim:
            raise RuntimeError(
                f"critic vector has size {vector.size}, expected {self.critic_vector_dim}"
            )
        return np.clip(vector, -1.0, 1.0).astype(np.float32)

    def _large_scale_channel_state(self):
        """计算大尺度信道增益和LOS概率的归一化特征向量.
        包括:卫星到RIS的信道增益、RIS到Bob的信道增益、卫星到Bob的信道增益、LOS概率."""
        rotation = self.uav_ris.rotation_matrix
        beta_sr = self.channel_model.get_s2a_large_scale_gain(  #beta_sr表示卫星到RIS的信道增益
            self.satellite.position,
            self.uav_ris.position,
            ris_rotation=rotation,
        )
        beta_rb = np.asarray([  #beta_rb表示RIS到Bob的信道增益
            self.channel_model.get_ris_to_ground_large_scale_gain(
                self.uav_ris.position,
                bob.position,
                ris_rotation=rotation,
            )
            for bob in self.bobs
        ])
        #beta_sb表示卫星到Bob的信道增益
        beta_sb = np.asarray([
            self.channel_model.get_s2g_large_scale_gain(
                self.satellite.position,
                bob.position,
            )
            for bob in self.bobs
        ])
        rg_visibility = np.asarray(
            [
                self.channel_model.get_platform_to_ground_los_probability(
                    self.uav_ris.position,
                    bob.position,
                    ris_rotation=rotation,
                )
                if beta_rb[index] > 0.0
                else 0.0
                for index, bob in enumerate(self.bobs)
            ]
        )
        lower, upper = self.params.channel_gain_db_min, self.params.channel_gain_db_max
        return np.r_[
            self._normalize_db(beta_sr, lower, upper), self._normalize_db(beta_rb, lower, upper),
            self._normalize_db(beta_sb, lower, upper), float(beta_sr > 0.0), (beta_rb > 0.0).astype(float), rg_visibility
        ]

    @staticmethod
    def _normalize_db(value, lower, upper):
        value_db = 10.0 * np.log10(np.maximum(np.asarray(value, dtype=float), 1e-30))
        return 2.0 * np.clip((value_db - float(lower)) / (float(upper) - float(lower)), 0.0, 1.0) - 1.0

    def _build_risk_graph(self):
        # 构建风险图，包括节点特征和边索引，用于表示环境中潜在的安全风险.
        size = int(self.params.risk_grid_size)
        x_min, x_max, y_min, y_max = self.bounds
        dx, dy = (x_max - x_min) / size, (y_max - y_min) / size
        xs = x_min + (np.arange(size) + 0.5) * dx
        ys = y_min + (np.arange(size) + 0.5) * dy
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        points = np.column_stack((xx.ravel(), yy.ravel())) #.ravel()将多维数组展平为一维数组，column_stack()将多个一维数组按列堆叠成二维数组
        distance_to_bobs = np.linalg.norm(points[:, None, :] - self.user_positions[None, :, :2], axis=2)
        nearest_distance = distance_to_bobs.min(axis=1)
        protected = nearest_distance <= float(self.params.bob_protect_radius)
        parent_density = eve_parent_intensity(points, self.user_positions, self.params)
        allowed_parent_density = parent_density.copy()
        allowed_parent_density[protected] = 0.0
        hard_core = float(self.params.eve_min_distance)
        if hard_core > 0.0:
            density_grid = allowed_parent_density.reshape(size, size) # 将一维数组重塑为二维数组，内容为每个网格点的父密度值
            radius_x = min(size - 1, int(np.ceil(hard_core / dx)))
            radius_y = min(size - 1, int(np.ceil(hard_core / dy)))
            offset_y, offset_x = np.mgrid[-radius_y:radius_y + 1, -radius_x:radius_x + 1]
            #kernel表示卷积核，用于计算每个网格点的局部平均父密度，卷积核的形状为(2*radius_y+1, 2*radius_x+1)，其中每个元素表示该位置是否在硬核距离内
            kernel = ((offset_x * dx) ** 2 + (offset_y * dy) ** 2 <= hard_core ** 2).astype(float)
            #fft_shape表示进行快速傅里叶变换的形状，计算卷积时需要的尺寸
            fft_shape = (size + kernel.shape[0] - 1, size + kernel.shape[1] - 1)
            #density_fft表示父密度网格的快速傅里叶变换，kernel_fft表示卷积核的快速傅里叶变换，convolution表示卷积结果
            density_fft = np.fft.rfftn(density_grid, s=fft_shape, axes=(0, 1))
            kernel_fft = np.fft.rfftn(kernel, s=fft_shape, axes=(0, 1))
            convolution = np.fft.irfftn(density_fft * kernel_fft, s=fft_shape, axes=(0, 1))
            local_mean = np.maximum(
                convolution[radius_y:radius_y + size, radius_x:radius_x + size].ravel(), 0.0
            ) * dx * dy
            retention = np.ones_like(local_mean)
            nonzero = local_mean > 1e-12
            retention[nonzero] = -np.expm1(-local_mean[nonzero]) / local_mean[nonzero]
        else:
            retention = np.ones_like(allowed_parent_density)
        hcpp_density = allowed_parent_density * retention
        distance_risk = np.exp(-nearest_distance / float(self.params.risk_distance_scale))
        grid_3d = np.c_[points, np.full(len(points), float(self.params.ground_height))]
        beta_se = np.asarray([self.channel_model.get_s2g_large_scale_gain(self.satellite.position, point) for point in grid_3d])
        beta_eb = np.asarray(
            [max(self.channel_model.get_g2g_large_scale_gain(point, bob.position) for bob in self.bobs) for point in grid_3d]
        )
        beta_eb *= 10.0 ** (float(self.params.jammer_main_gain_db) / 10.0)
        eavesdrop_snr = self.params.sat_max_power * beta_se / self.params.eve_noise_power
        jamming_inr = self.params.jammer_power * beta_eb / self.params.bob_noise_power
        sensing_weights = np.asarray([self.params.risk_eavesdrop_weight, self.params.risk_jamming_weight], dtype=float)
        sensing_weights /= sensing_weights.sum()
        eavesdrop_risk = 0.5 * (self._normalize_db(eavesdrop_snr, self.params.sinr_db_min, self.params.sinr_db_max) + 1.0)
        jamming_risk = 0.5 * (self._normalize_db(jamming_inr, self.params.inr_db_min, self.params.inr_db_max) + 1.0)
        density_risk = hcpp_density / (hcpp_density + float(self.params.risk_density_reference))
        sensing_risk = sensing_weights[0] * eavesdrop_risk + sensing_weights[1] * jamming_risk
        weights = np.asarray(
            [self.params.risk_distance_weight, self.params.risk_density_weight, self.params.risk_sensing_weight], dtype=float
        )
        weights /= weights.sum()
        omega = weights[0] * distance_risk + weights[1] * density_risk + weights[2] * sensing_risk
        risk_nodes = np.column_stack((density_risk, density_risk * omega, protected.astype(float))).astype(np.float32)
        node_ids = np.arange(size * size).reshape(size, size)
        left, right = node_ids[:, :-1].ravel(), node_ids[:, 1:].ravel()
        upper, lower = node_ids[:-1, :].ravel(), node_ids[1:, :].ravel()
        edge_index = np.vstack((np.r_[left, right, upper, lower], np.r_[right, left, lower, upper])).astype(np.int64)
        #risk_nodes表示每个网格点的风险特征，包括密度风险、加权风险和保护状态，edge_index表示网格点之间的连接关系，用于构建图结构
        return risk_nodes, edge_index

    def _ris_front_cosine(self, node_position):
        """计算RIS表面法向量与给定节点位置之间的余弦值，用于评估RIS对该节点的可见性和方向性."""
        direction = np.asarray(node_position, dtype=float) - self.uav_ris.position
        distance = np.linalg.norm(direction)
        if distance <= 1e-12:
            return 0.0
        return float(max(np.dot(self.uav_ris.normal, direction / distance), 0.0))

    def _ris_normal_angle_deg(self, node_position):
        direction = np.asarray(node_position, dtype=float) - self.uav_ris.position
        distance = np.linalg.norm(direction)
        if distance <= 1e-12:
            return 0.0
        cosine = np.clip(
            np.dot(self.uav_ris.normal, direction / distance),
            -1.0,
            1.0,
        )
        return float(np.degrees(np.arccos(cosine)))

    def _get_info(self):
        m = self.metrics
        critic_vector = self._get_critic_vector()
        episode_outage_rate = None if self.step_count == 0 else self.outage_count / self.step_count
        x_min, x_max, y_min, y_max = self.bounds
        eve_xy = 2.0 * (self.eve_positions[:, :2] - [x_min, y_min]) / [x_max - x_min, y_max - y_min] - 1.0
        critic_eve_nodes = np.c_[np.clip(eve_xy, -1.0, 1.0), np.r_[1.0, np.zeros(len(self.passive_eves))]].astype(np.float32)
        satellite_incidence_cosine = self._ris_front_cosine(self.satellite.position)
        bob_departure_cosine = np.asarray([self._ris_front_cosine(bob.position) for bob in self.bobs], dtype=float)
        rotation = self.uav_ris.rotation_matrix
        beta_sr = self.channel_model.get_s2a_large_scale_gain(
            self.satellite.position,
            self.uav_ris.position,
            ris_rotation=rotation,
        )
        beta_rb = np.asarray([
            self.channel_model.get_ris_to_ground_large_scale_gain(
                self.uav_ris.position,
                bob.position,
                ris_rotation=rotation,
            )
            for bob in self.bobs
        ])
        active = self.uav_ris.element_mode.astype(bool)
        effective_active_amplitude = (
            float(np.max(self.uav_ris.amplitude[active]))
            if np.any(active)
            else 1.0
        )
        return {
            "bob_sinr": m["bob_sinr"].copy(),
            "bob_direct_rx_power_dbm": m["bob_direct_rx_power_dbm"].copy(),
            "bob_ris_rx_power_dbm": m["bob_ris_rx_power_dbm"].copy(),
            "bob_coherent_rx_power_dbm": m["bob_coherent_rx_power_dbm"].copy(),
            "bob_ris_to_direct_power_ratio_db": m[
                "bob_ris_to_direct_power_ratio_db"
            ].copy(),
            "bob_direct_environment_loss_db": np.asarray([
                bob.direct_environment_loss_db for bob in self.bobs
            ]),
            "bob_jammer_power": m["bob_jammer_power"].copy(),
            "bob_ris_noise_power": m["bob_ris_noise_power"].copy(),
            "bob_jammer_inr": m["bob_jammer_inr"].copy(),
            "active_eve_sinr": m["active_eve_sinr"].copy(),
            "passive_eve_sinr": m["passive_eve_sinr"].copy(),
            "eve_sinr": m["eve_sinr"].copy(),
            "max_eve_sinr": m["max_eve_sinr"].copy(),
            "bob_capacity": m["bob_capacity"].copy(),
            "eve_capacity": m["eve_capacity"].copy(),
            "max_eve_capacity": m["max_eve_capacity"].copy(),
            "raw_security_gap": m["raw_security_gap"].copy(),
            "secrecy_rate": m["secrecy_rate"].copy(),
            "user_margin": m["user_margin"].copy(),
            "user_outage": m["user_outage"].copy(),
            "system_margin": m["system_margin"],
            "outage_sample": m["outage_sample"],
            "outage": m["outage_sample"],
            "episode_outage_rate": episode_outage_rate,
            "satellite_power": self.satellite.transmit_power,
            "ris_power": self.ris_power,
            "sat_power_ratio": m["sat_power_ratio"],
            "ris_power_ratio": m["ris_power_ratio"],
            "move_ratio": m["move_ratio"],
            "orientation_move_ratio": m["orientation_move_ratio"],
            "ris_rotation_cost": m["orientation_move_ratio"],
            "ris_action_repaired": self.ris_action_repaired,
            "ris_projection_scale": self.uav_ris.last_power_projection_scale,
            "ris_active_element_count": int(np.count_nonzero(active)),
            "uav_boundary_clipped": self.uav_ris.boundary_clipped,
            "ris_yaw_deg": float(np.degrees(self.uav_ris.yaw)),
            "ris_pitch_deg": float(np.degrees(self.uav_ris.pitch)),
            "ris_normal": self.uav_ris.normal.copy(),
            "ris_yaw_delta_deg": float(np.degrees(self.uav_ris.last_yaw_delta)),
            "ris_pitch_delta_deg": float(np.degrees(self.uav_ris.last_pitch_delta)),
            "satellite_ris_incidence_cosine": satellite_incidence_cosine,
            "ris_bob_departure_cosine": bob_departure_cosine,
            "satellite_ris_distance_m": float(
                np.linalg.norm(self.satellite.position - self.uav_ris.position)
            ),
            "ris_bob_distance_m": np.asarray([
                np.linalg.norm(bob.position - self.uav_ris.position)
                for bob in self.bobs
            ]),
            "satellite_ris_incidence_angle_deg": self._ris_normal_angle_deg(
                self.satellite.position
            ),
            "ris_bob_departure_angle_deg": np.asarray([
                self._ris_normal_angle_deg(bob.position) for bob in self.bobs
            ]),
            "ris_effective_active_amplitude": effective_active_amplitude,
            "satellite_ris_large_scale_gain_db": float(
                10.0 * np.log10(max(beta_sr, 1e-300))
            ),
            "ris_bob_large_scale_gain_db": 10.0 * np.log10(
                np.maximum(beta_rb, 1e-300)
            ),
            "ris_active_eve_cosine": self._ris_front_cosine(self.active_eve.position),
            "jammer_ris_correction_applied": self.jammer_ris_correction_applied,
            "satellite_ris_visible": bool(
                self.channel_model.get_s2a_large_scale_gain(
                    self.satellite.position,
                    self.uav_ris.position,
                    ris_rotation=self.uav_ris.rotation_matrix,
                ) > 0.0),
            "active_eve_position": self.active_eve.position.copy(),
            "active_eve_sample_index": self.active_eve_index,
            "num_passive_eves": len(self.passive_eves),
            "jammer_target_user": self.active_eve.target_user_index,
            "jammer_boresight_deg": float(np.degrees(self.active_eve.tx_boresight_azimuth)),
            "jammer_gain_to_bobs": np.asarray([self.active_eve.get_tx_gain(bob.position) for bob in self.bobs]),
            "jammer_gain_to_passive_eves": np.asarray([self.active_eve.get_tx_gain(eve.position) for eve in self.passive_eves]),
            "jammer_gain_to_ris": self.active_eve.get_tx_gain(self.uav_ris.position),
            "critic_vector": critic_vector,
            "critic_observation": {
                "vector": critic_vector.copy(),
                "risk_nodes": self.risk_nodes.copy(),
            },
            "critic_eve_nodes": critic_eve_nodes,
        }
