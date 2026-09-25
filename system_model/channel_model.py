import numpy as np


class SAGIN_ChannelModel:
    """SAGIN channel model. S2A: (N_ris, M_sat), A2G: (N_ground, N_ris), S2G: (N_ground, M_sat)."""

    def __init__(self, params, rng=None):
        self.params = params
        self.carrier_freq = float(params.carrier_freq)
        self.c = float(params.c)
        if not np.isfinite(self.carrier_freq) or not np.isfinite(self.c) or self.carrier_freq <= 0.0 or self.c <= 0.0:
            raise ValueError("carrier_freq and c must be finite and positive")

        self.wavelength = self.c / self.carrier_freq
        self.a2g_a = float(getattr(params, "a2g_a", getattr(params, "s2g_a", 9.61)))
        self.a2g_b = float(getattr(params, "a2g_b", getattr(params, "s2g_b", 0.16)))
        self.eta_los = float(params.eta_los)
        self.eta_nlos = float(params.eta_nlos)
        self.atmospheric_loss_db = float(getattr(params, "atmospheric_loss_db", 0.0))
        self.s2a_atmospheric_loss_db = float(getattr(params, "s2a_atmospheric_loss_db", self.atmospheric_loss_db))
        self.sat_tx_gain = 10.0 ** (float(getattr(params, "sat_tx_gain_db", 0.0)) / 10.0)
        self.ground_rx_gain = 10.0 ** (float(getattr(params, "ground_rx_gain_db", 0.0)) / 10.0)
        self.ris_link_gain = 10.0 ** (float(getattr(params, "ris_link_gain_db", 0.0)) / 10.0)
        self.g2g_reference_gain_db = float(getattr(params, "g2g_reference_gain_db", -30.0))
        self.g2g_path_loss_exponent = float(getattr(params, "g2g_path_loss_exponent", 3.0))
        self.use_3d_array_response = bool(getattr(params, "use_3d_array_response", True))
        self.array_spacing_wavelengths = float(getattr(params, "array_spacing_wavelengths", 0.5))
        self.sat_array_rows = int(getattr(params, "sat_array_rows", 2))
        self.ris_array_rows = int(getattr(params, "ris_array_rows", 8))
        self.ground_array_rows = int(getattr(params, "ground_array_rows", 1))
        self.sat_array_plane = str(getattr(params, "sat_array_plane", "xy")).lower()
        self.ris_array_plane = str(getattr(params, "ris_array_plane", "yz")).lower()
        self.ground_array_plane = str(getattr(params, "ground_array_plane", "xy")).lower()
        self.enforce_ris_half_space = bool(getattr(params, "enforce_ris_half_space", True))
        self.ris_angle_gain_exponent = float(getattr(params, "ris_angle_gain_exponent", 1.0))

        if self.array_spacing_wavelengths <= 0.0: #阵列间距必须为正
            raise ValueError("array_spacing_wavelengths must be positive")
        if min(self.sat_array_rows, self.ris_array_rows, self.ground_array_rows) < 1:
            raise ValueError("array row counts must be positive")
        if any(plane not in {"xy", "xz", "yz"} for plane in (self.sat_array_plane, self.ris_array_plane, self.ground_array_plane)):
            raise ValueError("array planes must be xy, xz, or yz")
        if not np.isfinite(self.ris_angle_gain_exponent) or self.ris_angle_gain_exponent < 0.0:
            raise ValueError("ris_angle_gain_exponent must be finite and non-negative")

        seed = getattr(params, "seed", None)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def _free_space_path_loss(self, distance):
        distance = np.maximum(np.asarray(distance, dtype=float), 1.0)
        fspl_db = 20.0 * np.log10(distance) + 20.0 * np.log10(self.carrier_freq) + 20.0 * np.log10(4.0 * np.pi / self.c)
        return 10.0 ** (-fspl_db / 10.0)

    @staticmethod
    def _as_pos(pos):
        pos = np.asarray(pos, dtype=float).reshape(-1)
        if pos.size != 3 or not np.all(np.isfinite(pos)):
            raise ValueError("position must contain 3 finite coordinates")
        return pos

    @staticmethod
    def _array_shape(num_antennas, preferred_rows):
        """计算阵列的行数和列数，确保阵列的总天线数为 num_antennas，并且首选行数为 preferred_rows。"""
        num_antennas = int(num_antennas)
        preferred_rows = int(preferred_rows)
        if num_antennas < 1 or preferred_rows < 1:
            raise ValueError("array dimensions must be positive")
        if num_antennas % preferred_rows != 0:
            raise ValueError(f"num_antennas={num_antennas} cannot be divided by preferred_rows={preferred_rows}")
        return preferred_rows, num_antennas // preferred_rows

    def _steering_vector(self, num_antennas, azimuth): 
        """将阵列响应向量 a(u) 计算为给定阵列方向的 steering vector，其中 u 是 azimuth 角度。"""
        idx = np.arange(int(num_antennas), dtype=float)
        return np.exp(-2.0j * np.pi * self.array_spacing_wavelengths * idx * np.sin(azimuth))

    @staticmethod
    def _direction_angles(origin, target):
        """计算从 origin 到 target 的传播方向的方位角azimuth和仰角elevation。"""
        delta = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
        horizontal = np.hypot(delta[0], delta[1])
        return float(np.arctan2(delta[1], delta[0])), float(np.arctan2(delta[2], horizontal))

    def _array_response(self, num_antennas, azimuth, elevation, preferred_rows=1, plane="xy"):
        """计算阵列响应，考虑阵列的几何结构和传播方向。"""
        num_antennas = int(num_antennas)
        if not self.use_3d_array_response:
            return self._steering_vector(num_antennas, azimuth)

        rows, cols = self._array_shape(num_antennas, preferred_rows)
        row_idx, col_idx = np.meshgrid(np.arange(rows, dtype=float), np.arange(cols, dtype=float), indexing="ij")
        ux = np.cos(elevation) * np.cos(azimuth)
        uy = np.cos(elevation) * np.sin(azimuth)
        uz = np.sin(elevation)

        if plane == "xy":
            phase_term = col_idx * ux + row_idx * uy
        elif plane == "yz":
            phase_term = col_idx * uz + row_idx * uy
        elif plane == "xz":
            phase_term = col_idx * ux + row_idx * uz
        else:
            raise ValueError(f"unsupported array plane: {plane}")

        phase = -2.0j * np.pi * self.array_spacing_wavelengths * phase_term
        return np.exp(phase).reshape(-1)

    @staticmethod
    def _as_rotation(rotation):
        if rotation is None:
            return np.eye(3, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("RIS rotation must be a finite 3 x 3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0
        ):
            raise ValueError("RIS rotation must be a proper orthogonal matrix")
        return rotation

    def ris_element_offsets(self, rotation=None, num_elements=None):
        """Return centered world-frame RIS element offsets for a local yz array."""
        num_elements = int(getattr(self.params, "num_ris_elements")) if num_elements is None else int(num_elements)
        rows, cols = self._array_shape(num_elements, self.ris_array_rows)
        row_idx, col_idx = np.meshgrid(np.arange(rows, dtype=float), np.arange(cols, dtype=float), indexing="ij")
        local_offsets = np.column_stack(
            (
                np.zeros(num_elements, dtype=float),
                (row_idx - 0.5 * (rows - 1)).reshape(-1),
                (col_idx - 0.5 * (cols - 1)).reshape(-1),
            )
        )
        local_offsets *= self.array_spacing_wavelengths * self.wavelength
        return local_offsets @ self._as_rotation(rotation).T

    def _ris_array_response(self, num_elements, propagation_direction, rotation=None):
        propagation_direction = np.asarray(propagation_direction, dtype=float).reshape(-1)
        if propagation_direction.size != 3 or not np.all(np.isfinite(propagation_direction)):
            raise ValueError("propagation_direction must contain 3 finite values")
        norm = np.linalg.norm(propagation_direction)
        if norm <= 1e-12:
            raise ValueError("propagation_direction cannot be zero")
        offsets = self.ris_element_offsets(rotation, num_elements=num_elements)
        phase = -2.0j * np.pi / self.wavelength * (offsets @ (propagation_direction / norm))
        return np.exp(phase)

    def _los_array_matrix(
        self,
        tx_pos,
        rx_pos,
        num_tx,
        num_rx,
        tx_rows,
        rx_rows,
        tx_plane,
        rx_plane,
        tx_rotation=None,
        rx_rotation=None,
    ):
        """返回 a_rx(u) a_tx(u)^H，其中 u 始终是 tx 到 rx 的传播方向。
        a_rx(u) a_tx(u)^H表示接收端和发射端的阵列响应向量的外积，表示在给定传播方向下的线性阵列响应。
        """
        tx_pos = self._as_pos(tx_pos)
        rx_pos = self._as_pos(rx_pos)
        propagation_direction = rx_pos - tx_pos
        distance = np.linalg.norm(propagation_direction)
        if distance <= 1e-12:
            raise ValueError("transmitter and receiver positions must be distinct")
        propagation_direction /= distance
        azimuth, elevation = self._direction_angles(tx_pos, rx_pos)
        a_tx = (
            self._ris_array_response(num_tx, propagation_direction, tx_rotation)
            if tx_rotation is not None
            else self._array_response(num_tx, azimuth, elevation, tx_rows, plane=tx_plane)
        )
        a_rx = (
            self._ris_array_response(num_rx, propagation_direction, rx_rotation)
            if rx_rotation is not None
            else self._array_response(num_rx, azimuth, elevation, rx_rows, plane=rx_plane)
        )
        return np.outer(a_rx, np.conj(a_tx))

    def _rayleigh(self, shape):
        """返回一个复高斯随机矩阵，表示瑞利衰落信道的随机小尺度衰落分量。"""
        return (self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)) / np.sqrt(2.0)

    def ris_angular_power_gain(self, ris_pos, node_pos, rotation=None):
        """Return max(n dot u, 0)^q as a power-domain gain for the RIS-to-node direction."""
        if not self.enforce_ris_half_space:
            return 1.0
        direction = self._as_pos(node_pos) - self._as_pos(ris_pos)
        distance = np.linalg.norm(direction)
        if distance <= 1e-12:
            return 0.0
        normal = self._as_rotation(rotation)[:, 0]
        cosine = float(np.clip(np.dot(normal, direction / distance), -1.0, 1.0))
        # Treat round-off at exact grazing incidence as zero.  This is
        # especially important for q=0, where a tiny positive artifact would
        # otherwise jump to unit gain.
        if cosine <= 1e-12:
            return 0.0
        if self.ris_angle_gain_exponent == 0.0:
            return 1.0
        return float(cosine ** self.ris_angle_gain_exponent)

    def _ris_link_visible(self, ris_pos, node_pos, ris_rotation=None):
        """检查 RIS 是否可见，即节点是否位于 RIS 的半空间内。"""
        if not self.enforce_ris_half_space:
            return True
        return self.ris_angular_power_gain(ris_pos, node_pos, ris_rotation) > 0.0

    def get_a2g_los_probability(self, uav_pos, ground_pos):
        """计算 UAV 到地面节点的 LOS 概率，基于仰角和环境参数。"""
        uav_pos = self._as_pos(uav_pos)
        ground_pos = self._as_pos(ground_pos)
        _, elevation_g2u = self._direction_angles(ground_pos, uav_pos)
        elevation_deg = np.clip(np.degrees(elevation_g2u), 0.0, 90.0)
        return float(1.0 / (1.0 + self.a2g_a * np.exp(-self.a2g_b * (elevation_deg - self.a2g_a))))

    def get_a2g_large_scale_gain(self, uav_pos, ground_pos, ris_rotation=None):
        """计算 UAV 到地面节点的 A2G 大尺度增益，考虑 LOS 概率和额外的路径损耗。"""
        uav_pos = self._as_pos(uav_pos)
        ground_pos = self._as_pos(ground_pos)
        angular_power_gain = self.ris_angular_power_gain(uav_pos, ground_pos, ris_rotation)
        if angular_power_gain <= 0.0:
            return 0.0
        distance = max(float(np.linalg.norm(uav_pos - ground_pos)), 1.0)
        prob_los = self.get_a2g_los_probability(uav_pos, ground_pos)
        excess_gain = prob_los * 10.0 ** (-self.eta_los / 10.0) + (1.0 - prob_los) * 10.0 ** (-self.eta_nlos / 10.0)
        return float(
            self._free_space_path_loss(distance)
            * excess_gain
            * self.ris_link_gain
            * self.ground_rx_gain
            * angular_power_gain
        )

    def get_s2g_large_scale_gain(self, sat_pos, ground_pos):
        """计算卫星到地面节点的 S2G 大尺度增益，考虑自由空间路径损耗和大气衰减。"""
        sat_pos = self._as_pos(sat_pos)
        ground_pos = self._as_pos(ground_pos)
        distance = max(float(np.linalg.norm(sat_pos - ground_pos)), 1.0)
        atmospheric = 10.0 ** (-self.atmospheric_loss_db / 10.0)
        return float(self._free_space_path_loss(distance) * atmospheric * self.sat_tx_gain * self.ground_rx_gain)

    def get_s2a_large_scale_gain(self, sat_pos, uav_pos, ris_rotation=None):
        """计算卫星到 UAV 的 S2A 大尺度增益，考虑自由空间路径损耗和大气衰减。"""
        sat_pos = self._as_pos(sat_pos)
        uav_pos = self._as_pos(uav_pos)
        angular_power_gain = self.ris_angular_power_gain(uav_pos, sat_pos, ris_rotation)
        if angular_power_gain <= 0.0:
            return 0.0
        distance = max(float(np.linalg.norm(sat_pos - uav_pos)), 1.0)
        atmospheric = 10.0 ** (-self.s2a_atmospheric_loss_db / 10.0)
        return float(
            self._free_space_path_loss(distance)
            * atmospheric
            * self.sat_tx_gain
            * self.ris_link_gain
            * angular_power_gain
        )

    def get_g2g_large_scale_gain(self, pos_tx, pos_rx):
        """计算地面节点到地面节点的 G2G 大尺度增益，考虑自由空间路径损耗和额外的路径损耗。"""
        pos_tx = self._as_pos(pos_tx)
        pos_rx = self._as_pos(pos_rx)
        distance = max(float(np.linalg.norm(pos_tx - pos_rx)), 1.0)
        beta0_linear = 10.0 ** (self.g2g_reference_gain_db / 10.0)
        return float(beta0_linear * distance ** (-self.g2g_path_loss_exponent) * self.ground_rx_gain)
 
    def get_a2g_channel(
        self,
        uav_pos,
        ground_pos,
        num_uav_antennas,
        num_ground_antennas,
        rician_factor=None,
        ris_rotation=None,
    ):
        uav_pos = self._as_pos(uav_pos)
        ground_pos = self._as_pos(ground_pos)
        num_uav_antennas = int(num_uav_antennas)
        num_ground_antennas = int(num_ground_antennas)
        angular_power_gain = self.ris_angular_power_gain(uav_pos, ground_pos, ris_rotation)
        if angular_power_gain <= 0.0:
            return np.zeros((num_ground_antennas, num_uav_antennas), dtype=np.complex128)
        rician_factor = float(getattr(self.params, "rician_factor_a2g_db", 10.0) if rician_factor is None else rician_factor)
        distance = max(float(np.linalg.norm(uav_pos - ground_pos)), 1.0)
        prob_los = self.get_a2g_los_probability(uav_pos, ground_pos)
        is_los = self.rng.random() < prob_los
        extra_loss_db = self.eta_los if is_los else self.eta_nlos
        large_scale = (
            self._free_space_path_loss(distance)
            * 10.0 ** (-extra_loss_db / 10.0)
            * self.ris_link_gain
            * self.ground_rx_gain
            * angular_power_gain
        )
        h_los = self._los_array_matrix(
            uav_pos,
            ground_pos,
            num_uav_antennas,
            num_ground_antennas,
            self.ris_array_rows,
            self.ground_array_rows,
            self.ris_array_plane,
            self.ground_array_plane,
            tx_rotation=self._as_rotation(ris_rotation),
        )
        path_phase = np.exp(-1j * 2.0 * np.pi * distance / self.wavelength)

        if is_los:
            k_linear = 10.0 ** (rician_factor / 10.0)
            h_small = np.sqrt(k_linear / (k_linear + 1.0)) * path_phase * h_los + np.sqrt(1.0 / (k_linear + 1.0)) * self._rayleigh(h_los.shape)
        else:
            h_small = self._rayleigh(h_los.shape)
        return np.sqrt(large_scale) * h_small

    def get_s2g_channel(self, sat_pos, ground_pos, num_sat_antennas, num_ground_antennas):
        sat_pos = self._as_pos(sat_pos)
        ground_pos = self._as_pos(ground_pos)
        distance = max(float(np.linalg.norm(sat_pos - ground_pos)), 1.0)
        large_scale = self.get_s2g_large_scale_gain(sat_pos, ground_pos)
        h_los = self._los_array_matrix(sat_pos, ground_pos, num_sat_antennas, num_ground_antennas, self.sat_array_rows, self.ground_array_rows, self.sat_array_plane, self.ground_array_plane)
        return np.sqrt(large_scale) * h_los * np.exp(-1j * 2.0 * np.pi * distance / self.wavelength)

    def get_s2a_channel(self, sat_pos, uav_pos, num_sat_antennas, num_uav_antennas, ris_rotation=None):
        sat_pos = self._as_pos(sat_pos)
        uav_pos = self._as_pos(uav_pos)
        distance = max(float(np.linalg.norm(sat_pos - uav_pos)), 1.0)
        large_scale = self.get_s2a_large_scale_gain(sat_pos, uav_pos, ris_rotation=ris_rotation)
        if large_scale <= 0.0:
            return np.zeros((num_uav_antennas, num_sat_antennas), dtype=np.complex128)
        h_los = self._los_array_matrix(
            sat_pos,
            uav_pos,
            num_sat_antennas,
            num_uav_antennas,
            self.sat_array_rows,
            self.ris_array_rows,
            self.sat_array_plane,
            self.ris_array_plane,
            rx_rotation=self._as_rotation(ris_rotation),
        )
        return np.sqrt(large_scale) * h_los * np.exp(-1j * 2.0 * np.pi * distance / self.wavelength)

    def get_g2g_channel(self, pos_tx, pos_rx):
        large_scale = self.get_g2g_large_scale_gain(pos_tx, pos_rx)
        return np.sqrt(large_scale) * self._rayleigh(())

    def add_bounded_csi_error(self, nominal_channel, epsilon, relative=True):
        """在给定的名义信道上添加有界的 CSI 误差，误差的大小由 epsilon 控制。"""
        nominal_channel = np.asarray(nominal_channel, dtype=np.complex128)
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")
        noise = self._rayleigh(nominal_channel.shape)
        noise_norm = np.linalg.norm(noise)
        if noise_norm <= 1e-12:
            return nominal_channel.copy()
        nominal_norm = np.linalg.norm(nominal_channel)
        radius = epsilon * (nominal_norm if relative else 1.0)
        return nominal_channel + noise / noise_norm * self.rng.uniform(0.0, radius)
