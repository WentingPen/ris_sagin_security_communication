import numpy as np

try:
    from scipy.special import jv as _bessel_jv
except ImportError:  # The optional Bessel pattern is disabled by default.
    _bessel_jv = None


class SAGIN_ChannelModel:
    """SAGIN channel model for satellite, fixed HAPS-RIS, and ground nodes."""

    def __init__(self, params, rng=None):
        self.params = params
        self.carrier_freq = float(params.carrier_freq)
        self.c = float(params.c)
        if not np.isfinite(self.carrier_freq) or not np.isfinite(self.c) or self.carrier_freq <= 0.0 or self.c <= 0.0:
            raise ValueError("carrier_freq and c must be finite and positive")

        self.wavelength = self.c / self.carrier_freq
        self.rician_factor_sr_db = float(
            getattr(params, "rician_factor_sr_db", 15.0)
        )
        self.m_rg = float(getattr(params, "m_rg", 2.0))
        self.omega_rg = float(getattr(params, "omega_rg", 1.0))
        self.alpha_rg = float(getattr(params, "alpha_rg", 2.0))
        self.rg_reference_distance = float(
            getattr(params, "rg_reference_distance_m", 1.0)
        )
        self.use_sat_bessel_pattern = bool(
            getattr(params, "use_sat_bessel_pattern", False)
        )
        self.sat_bessel_3db_beamwidth = np.radians(
            float(getattr(params, "sat_bessel_3db_beamwidth_deg", 9.0))
        )
        self.sat_bessel_beam_center = np.asarray(
            getattr(params, "sat_bessel_beam_center", (0.0, 0.0, 0.0)),
            dtype=float,
        ).reshape(-1)
        self.a2g_a = float(getattr(params, "a2g_a", getattr(params, "s2g_a", 9.61)))
        self.a2g_b = float(getattr(params, "a2g_b", getattr(params, "s2g_b", 0.16)))
        self.eta_los = float(params.eta_los)
        self.eta_nlos = float(params.eta_nlos)
        self.platform_channel_model = str(getattr(params, "platform_channel_model", "a2g")).lower()
        self.rician_factor_h2g_db = float(getattr(params, "rician_factor_h2g_db", 15.0))
        self.h2g_atmospheric_loss_db = float(getattr(params, "h2g_atmospheric_loss_db", 0.0))
        self.h2g_clutter_loss_db = float(getattr(params, "h2g_clutter_loss_db", 0.0))
        self.atmospheric_loss_db = float(getattr(params, "atmospheric_loss_db", 0.0))
        self.s2a_atmospheric_loss_db = float(getattr(params, "s2a_atmospheric_loss_db", self.atmospheric_loss_db))
        self.s2g_clutter_loss_db = float(getattr(params, "s2g_clutter_loss_db", 0.0))
        self.s2g_fixed_shadowing_loss_db = float(
            getattr(params, "s2g_fixed_shadowing_loss_db", 0.0)
        )
        self.s2g_shadowing_std_db = float(getattr(params, "s2g_shadowing_std_db", 0.0))
        self.s2g_building_penetration_loss_db = float(
            getattr(params, "s2g_building_penetration_loss_db", 0.0)
        )
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

        if min(
            self.m_rg,
            self.omega_rg,
            self.alpha_rg,
            self.rg_reference_distance,
            self.sat_bessel_3db_beamwidth,
        ) <= 0.0:
            raise ValueError(
                "RG, reference-distance, and Bessel parameters must be positive"
            )
        if not np.isfinite(self.rician_factor_sr_db):
            raise ValueError("rician_factor_sr_db must be finite")
        if self.sat_bessel_3db_beamwidth >= np.pi:
            raise ValueError("sat_bessel_3db_beamwidth_deg must be smaller than 180 degrees")
        if self.sat_bessel_beam_center.size != 3 or not np.all(
            np.isfinite(self.sat_bessel_beam_center)
        ):
            raise ValueError(
                "sat_bessel_beam_center must contain 3 finite coordinates"
            )
        if self.use_sat_bessel_pattern and _bessel_jv is None:
            raise ImportError("use_sat_bessel_pattern=1 requires scipy.special")

        if self.array_spacing_wavelengths <= 0.0: #阵列间距必须为正
            raise ValueError("array_spacing_wavelengths must be positive")
        if min(self.sat_array_rows, self.ris_array_rows, self.ground_array_rows) < 1:
            raise ValueError("array row counts must be positive")
        if any(plane not in {"xy", "xz", "yz"} for plane in (self.sat_array_plane, self.ris_array_plane, self.ground_array_plane)):
            raise ValueError("array planes must be xy, xz, or yz")
        if not np.isfinite(self.ris_angle_gain_exponent) or self.ris_angle_gain_exponent < 0.0:
            raise ValueError("ris_angle_gain_exponent must be finite and non-negative")
        if self.platform_channel_model not in {"a2g", "h2g"}:
            raise ValueError("platform_channel_model must be 'a2g' or 'h2g'")
        if not np.isfinite(self.rician_factor_h2g_db):
            raise ValueError("rician_factor_h2g_db must be finite")
        if min(
            self.h2g_atmospheric_loss_db,
            self.h2g_clutter_loss_db,
            self.atmospheric_loss_db,
            self.s2a_atmospheric_loss_db,
            self.s2g_clutter_loss_db,
            self.s2g_fixed_shadowing_loss_db,
            self.s2g_shadowing_std_db,
            self.s2g_building_penetration_loss_db,
        ) < 0.0:
            raise ValueError("configured propagation losses must be non-negative")

        seed = getattr(params, "seed", None)
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self._s2g_shadowing_cache = {}

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

    def _sr_rician(self, los_response, path_phase):
        """Sample a unit-average-power LoS-dominant Satellite-RIS channel."""
        los_response = np.asarray(los_response, dtype=np.complex128)
        k_linear = 10.0 ** (self.rician_factor_sr_db / 10.0)
        return (
            np.sqrt(k_linear / (k_linear + 1.0))
            * complex(path_phase)
            * los_response
            + np.sqrt(1.0 / (k_linear + 1.0))
            * self._rayleigh(los_response.shape)
        )

    def _s2g_shadowing_db(self, sat_pos, ground_pos):
        """Return one cached zero-mean dB shadowing sample per fixed link."""
        if self.s2g_shadowing_std_db == 0.0:
            return 0.0
        key = (
            tuple(np.asarray(sat_pos, dtype=np.float64)),
            tuple(np.asarray(ground_pos, dtype=np.float64)),
        )
        if key not in self._s2g_shadowing_cache:
            self._s2g_shadowing_cache[key] = float(
                self.rng.normal(0.0, self.s2g_shadowing_std_db)
            )
        return self._s2g_shadowing_cache[key]

    def _nakagami_complex(self, array_response, path_phase):
        """Sample complex Nakagami coefficients with E[|g_rg|^2]=Omega_rg."""
        array_response = np.asarray(array_response, dtype=np.complex128)
        power = self.rng.gamma(
            shape=self.m_rg,
            scale=self.omega_rg / self.m_rg,
            size=array_response.shape,
        )
        random_phase = self.rng.uniform(
            -np.pi,
            np.pi,
            size=array_response.shape,
        )
        return (
            np.sqrt(power)
            * np.exp(1j * random_phase)
            * complex(path_phase)
            * array_response
        )

    def satellite_bessel_power_gain(self, sat_pos, node_pos):
        """Return a fixed Bessel envelope with unit peak and no absolute gain."""
        if not self.use_sat_bessel_pattern:
            return 1.0
        sat_pos = self._as_pos(sat_pos)
        node_pos = self._as_pos(node_pos)
        link_direction = node_pos - sat_pos
        beam_direction = self.sat_bessel_beam_center - sat_pos
        link_norm = np.linalg.norm(link_direction)
        beam_norm = np.linalg.norm(beam_direction)
        if link_norm <= 1e-12 or beam_norm <= 1e-12:
            raise ValueError(
                "satellite, node, and Bessel beam center must be distinct"
            )
        cosine = np.clip(
            np.dot(link_direction, beam_direction) / (link_norm * beam_norm),
            -1.0,
            1.0,
        )
        if cosine <= 0.0:
            return 0.0
        theta = float(np.arccos(cosine))
        rho = (
            2.07123
            * np.sin(theta)
            / np.sin(self.sat_bessel_3db_beamwidth)
        )
        if abs(rho) <= 1e-8:
            amplitude = 1.0
        else:
            amplitude = (
                _bessel_jv(1, rho) / (2.0 * rho)
                + 36.0 * _bessel_jv(3, rho) / rho ** 3
            )
        return float(np.clip(np.abs(amplitude) ** 2, 0.0, 1.0))

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

    def get_h2g_los_probability(self, haps_pos, ground_pos):
        """Return the clear-sky LoS probability for the fixed HAPS baseline."""
        self._as_pos(haps_pos)
        self._as_pos(ground_pos)
        return 1.0

    def get_h2g_large_scale_gain(self, haps_pos, ground_pos, ris_rotation=None):
        """HAPS-ground gain: FSPL, explicit excess losses, and RIS element pattern."""
        haps_pos = self._as_pos(haps_pos)
        ground_pos = self._as_pos(ground_pos)
        angular_power_gain = self.ris_angular_power_gain(haps_pos, ground_pos, ris_rotation)
        if angular_power_gain <= 0.0:
            return 0.0
        distance = max(float(np.linalg.norm(haps_pos - ground_pos)), 1.0)
        excess_loss = 10.0 ** (
            -(self.h2g_atmospheric_loss_db + self.h2g_clutter_loss_db) / 10.0
        )
        return float(
            self._free_space_path_loss(distance)
            * excess_loss
            * self.ris_link_gain
            * self.ground_rx_gain
            * angular_power_gain
        )

    def get_ris_to_ground_large_scale_gain(
        self,
        ris_pos,
        ground_pos,
        ris_rotation=None,
    ):
        """Return the dimensionless RIS-ground power gain at a physical 1 m reference."""
        ris_pos = self._as_pos(ris_pos)
        ground_pos = self._as_pos(ground_pos)
        angular_power_gain = self.ris_angular_power_gain(
            ris_pos,
            ground_pos,
            ris_rotation,
        )
        if angular_power_gain <= 0.0:
            return 0.0
        distance = max(
            float(np.linalg.norm(ris_pos - ground_pos)),
            self.rg_reference_distance,
        )
        beta0 = (
            self.wavelength
            / (4.0 * np.pi * self.rg_reference_distance)
        ) ** 2
        path_gain = beta0 * (
            distance / self.rg_reference_distance
        ) ** (-self.alpha_rg)
        return float(
            path_gain
            * self.ris_link_gain
            * self.ground_rx_gain
            * angular_power_gain
        )

    def get_platform_to_ground_los_probability(
        self,
        platform_pos,
        ground_pos,
        ris_rotation=None,
    ):
        """Compatibility feature: one means the ground node is on the RIS front side."""
        return float(
            self._ris_link_visible(
                platform_pos,
                ground_pos,
                ris_rotation,
            )
        )

    def get_platform_to_ground_large_scale_gain(
        self,
        platform_pos,
        ground_pos,
        ris_rotation=None,
    ):
        """Compatibility alias for the unified RIS-ground large-scale model."""
        return self.get_ris_to_ground_large_scale_gain(
            platform_pos,
            ground_pos,
            ris_rotation,
        )

    def get_s2g_large_scale_gain(
        self,
        sat_pos,
        ground_pos,
        additional_environment_loss_db=0.0,
    ):
        """Return S2G power gain with global and receiver-local dB losses."""
        sat_pos = self._as_pos(sat_pos)
        ground_pos = self._as_pos(ground_pos)
        additional_environment_loss_db = float(additional_environment_loss_db)
        if (
            not np.isfinite(additional_environment_loss_db)
            or additional_environment_loss_db < 0.0
        ):
            raise ValueError(
                "additional_environment_loss_db must be finite and non-negative"
            )
        distance = max(float(np.linalg.norm(sat_pos - ground_pos)), 1.0)
        shadowing_db = self._s2g_shadowing_db(sat_pos, ground_pos)
        excess_loss_db = (
            self.atmospheric_loss_db
            + self.s2g_clutter_loss_db
            + self.s2g_fixed_shadowing_loss_db
            + self.s2g_building_penetration_loss_db
            + shadowing_db
            + additional_environment_loss_db
        )
        return float(
            self._free_space_path_loss(distance)
            * 10.0 ** (-excess_loss_db / 10.0)
            * self.sat_tx_gain
            * self.ground_rx_gain
            * self.satellite_bessel_power_gain(sat_pos, ground_pos)
        )

    def get_s2a_large_scale_gain(self, sat_pos, uav_pos, ris_rotation=None):
        """计算卫星到 UAV 的 S2A 大尺度增益，考虑自由空间路径损耗和大气衰减。"""
        sat_pos = self._as_pos(sat_pos)
        uav_pos = self._as_pos(uav_pos)
        angular_power_gain = self.ris_angular_power_gain(uav_pos, sat_pos, ris_rotation)
        if angular_power_gain <= 0.0:
            return 0.0
        distance = max(float(np.linalg.norm(sat_pos - uav_pos)), 1.0)
        atmospheric = 10.0 ** (-self.s2a_atmospheric_loss_db / 10.0)
        bessel_power_gain = self.satellite_bessel_power_gain(sat_pos, uav_pos)
        return float(
            self._free_space_path_loss(distance)
            * atmospheric
            * self.sat_tx_gain
            * self.ris_link_gain
            * angular_power_gain
            * bessel_power_gain
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

    def get_h2g_channel(
        self,
        haps_pos,
        ground_pos,
        num_haps_antennas,
        num_ground_antennas,
        ris_rotation=None,
    ):
        """Generate the fixed-HAPS Rician channel to one ground node."""
        haps_pos = self._as_pos(haps_pos)
        ground_pos = self._as_pos(ground_pos)
        num_haps_antennas = int(num_haps_antennas)
        num_ground_antennas = int(num_ground_antennas)
        large_scale = self.get_h2g_large_scale_gain(
            haps_pos,
            ground_pos,
            ris_rotation=ris_rotation,
        )
        if large_scale <= 0.0:
            return np.zeros((num_ground_antennas, num_haps_antennas), dtype=np.complex128)
        distance = max(float(np.linalg.norm(haps_pos - ground_pos)), 1.0)
        h_los = self._los_array_matrix(
            haps_pos,
            ground_pos,
            num_haps_antennas,
            num_ground_antennas,
            self.ris_array_rows,
            self.ground_array_rows,
            self.ris_array_plane,
            self.ground_array_plane,
            tx_rotation=self._as_rotation(ris_rotation),
        )
        path_phase = np.exp(-1j * 2.0 * np.pi * distance / self.wavelength)
        k_linear = 10.0 ** (self.rician_factor_h2g_db / 10.0)
        h_small = (
            np.sqrt(k_linear / (k_linear + 1.0)) * path_phase * h_los
            + np.sqrt(1.0 / (k_linear + 1.0)) * self._rayleigh(h_los.shape)
        )
        return np.sqrt(large_scale) * h_small

    def get_ris_to_ground_channel(
        self,
        ris_pos,
        ground_pos,
        num_ris_elements,
        num_ground_antennas,
        ris_rotation=None,
    ):
        """Generate one RIS-ground channel with Nakagami envelope fading."""
        ris_pos = self._as_pos(ris_pos)
        ground_pos = self._as_pos(ground_pos)
        num_ris_elements = int(num_ris_elements)
        num_ground_antennas = int(num_ground_antennas)
        large_scale = self.get_ris_to_ground_large_scale_gain(
            ris_pos,
            ground_pos,
            ris_rotation=ris_rotation,
        )
        shape = (num_ground_antennas, num_ris_elements)
        if large_scale <= 0.0:
            return np.zeros(shape, dtype=np.complex128)
        distance = max(
            float(np.linalg.norm(ris_pos - ground_pos)),
            self.rg_reference_distance,
        )
        array_response = self._los_array_matrix(
            ris_pos,
            ground_pos,
            num_ris_elements,
            num_ground_antennas,
            self.ris_array_rows,
            self.ground_array_rows,
            self.ris_array_plane,
            self.ground_array_plane,
            tx_rotation=self._as_rotation(ris_rotation),
        )
        path_phase = np.exp(-1j * 2.0 * np.pi * distance / self.wavelength)
        h_small = self._nakagami_complex(array_response, path_phase)
        return np.sqrt(large_scale) * h_small

    def get_platform_to_ground_channel(
        self,
        platform_pos,
        ground_pos,
        num_platform_antennas,
        num_ground_antennas,
        ris_rotation=None,
    ):
        """Compatibility alias for the unified RIS-ground channel model."""
        return self.get_ris_to_ground_channel(
            platform_pos,
            ground_pos,
            num_platform_antennas,
            num_ground_antennas,
            ris_rotation=ris_rotation,
        )

    def get_s2g_channel(
        self,
        sat_pos,
        ground_pos,
        num_sat_antennas,
        num_ground_antennas,
        additional_environment_loss_db=0.0,
    ):
        sat_pos = self._as_pos(sat_pos)
        ground_pos = self._as_pos(ground_pos)
        distance = max(float(np.linalg.norm(sat_pos - ground_pos)), 1.0)
        large_scale = self.get_s2g_large_scale_gain(
            sat_pos,
            ground_pos,
            additional_environment_loss_db=additional_environment_loss_db,
        )
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
        path_phase = np.exp(-1j * 2.0 * np.pi * distance / self.wavelength)
        h_small = self._sr_rician(h_los, path_phase)
        return np.sqrt(large_scale) * h_small

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
