import numpy as np


def _vector(value, size, name, dtype=float):
    value = np.asarray(value, dtype=dtype).reshape(-1)
    if value.size != size or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain {size} finite values")
    return value


def _build_element_to_group(
    num_elements,
    array_rows,
    group_rows,
    group_cols,
    num_groups,
):
    if num_elements % array_rows != 0:
        raise ValueError("num_elements must be divisible by array_rows")
    array_cols = num_elements // array_rows
    if array_rows % group_rows != 0 or array_cols % group_cols != 0:
        raise ValueError("RIS group block must tile the full RIS array")
    mapping = np.empty((array_rows, array_cols), dtype=np.int64)
    group_index = 0
    for row in range(0, array_rows, group_rows):
        for col in range(0, array_cols, group_cols):
            mapping[row:row + group_rows, col:col + group_cols] = group_index
            group_index += 1
    if group_index != num_groups:
        raise ValueError(
            f"group layout creates {group_index} groups, expected {num_groups}"
        )
    return mapping.reshape(-1)


class RISPowerInfeasibleError(ValueError):
    """The selected active elements cannot satisfy the RIS power constraint."""


class UAV_RIS:
    def __init__(self, params, position):
        position = np.asarray(position, dtype=float).reshape(-1)
        if position.size not in (2, 3) or not np.all(np.isfinite(position)):
            raise ValueError("UAV position must contain 2 or 3 finite coordinates")
        self.height = float(params.uav_height)
        if position.size == 3 and not np.isclose(position[2], self.height):
            raise ValueError("UAV position height must equal uav_height")
        self.position = np.array([position[0], position[1], self.height], dtype=float)
        self.num_elements = int(params.num_ris_elements)
        self.num_groups = int(params.num_ris_groups)
        self.array_rows = int(params.ris_array_rows)
        self.group_rows = int(params.ris_group_rows)
        self.group_cols = int(params.ris_group_cols)
        self.phase_residual_max = np.radians(float(params.ris_phase_residual_max_deg))
        self.element_to_group = _build_element_to_group(
            self.num_elements,
            self.array_rows,
            self.group_rows,
            self.group_cols,
            self.num_groups,
        )
        self.max_amplitude = float(params.ris_max_amplitude)
        self.max_power = float(params.ris_max_power)
        self.noise_power = float(params.ris_noise_power)
        self.jammer_power = float(params.jammer_power)
        self.max_step = float(params.uav_max_step)
        self.max_yaw_step = np.radians(float(getattr(params, "ris_max_yaw_step_deg", 10.0)))
        self.max_pitch_step = np.radians(float(getattr(params, "ris_max_pitch_step_deg", 10.0)))
        self.pitch_min = np.radians(float(getattr(params, "ris_pitch_min_deg", -60.0)))
        self.pitch_max = np.radians(float(getattr(params, "ris_pitch_max_deg", 60.0)))
        initial_yaw = np.radians(float(getattr(params, "ris_initial_yaw_deg", 0.0)))
        self.yaw = (initial_yaw + np.pi) % (2.0 * np.pi) - np.pi
        self.pitch = np.radians(float(getattr(params, "ris_initial_pitch_deg", 0.0)))
        self.bounds = np.asarray(params.area_bounds, dtype=float).reshape(-1)
        if self.num_elements < 1 or self.num_groups < 1 or self.num_elements % self.num_groups != 0:
            raise ValueError("RIS dimensions must be positive and num_ris_elements must be divisible by num_ris_groups")
        if self.max_amplitude < 1.0 or not np.isfinite(self.max_power) or self.max_power <= 0.0:
            raise ValueError("invalid RIS amplitude or power limit")
        if self.noise_power < 0.0 or self.jammer_power < 0.0 or self.max_step < 0.0:
            raise ValueError("RIS noise, jammer power, and UAV step limit must be non-negative")
        if self.max_yaw_step <= 0.0 or self.max_pitch_step <= 0.0:
            raise ValueError("RIS yaw and pitch step limits must be positive")
        if not -0.5 * np.pi <= self.pitch_min < self.pitch_max <= 0.5 * np.pi:
            raise ValueError("RIS pitch limits must be ordered and lie within [-pi/2, pi/2]")
        if not self.pitch_min <= self.pitch <= self.pitch_max:
            raise ValueError("initial RIS pitch must lie within the configured pitch limits")
        if self.bounds.size != 4 or self.bounds[0] >= self.bounds[1] or self.bounds[2] >= self.bounds[3]:
            raise ValueError("area_bounds must be [x_min, x_max, y_min, y_max]")
        if not (self.bounds[0] <= self.position[0] <= self.bounds[1] and self.bounds[2] <= self.position[1] <= self.bounds[3]):
            raise ValueError("initial UAV position is outside area_bounds")
        self.last_requested_displacement = np.zeros(2, dtype=float)
        self.last_displacement = np.zeros(2, dtype=float)
        self.boundary_clipped = False
        self.last_yaw_delta = 0.0
        self.last_pitch_delta = 0.0
        self.orientation_move_ratio = 0.0
        self.rotation_matrix = self.rotation_matrix_from_angles(self.yaw, self.pitch)
        self.normal = self.rotation_matrix[:, 0].copy()
        self.last_power_projection_scale = 1.0
        self.group_mode = np.zeros(self.num_groups, dtype=int)
        self.element_mode = np.zeros(self.num_elements, dtype=int)
        self.group_amplitude = np.ones(self.num_groups, dtype=float)
        self.group_phase_residual = np.zeros(self.num_groups, dtype=float)
        self.base_phase = np.zeros(self.num_elements, dtype=float)
        self.amplitude = np.ones(self.num_elements, dtype=float)
        self.phase = np.zeros(self.num_elements, dtype=float)

    @property
    def ris_coeff(self):
        gain = 1.0 + self.element_mode * (self.amplitude - 1.0)
        return gain * np.exp(1j * self.phase)

    @property
    def active_ris_coeff(self):
        return self.element_mode * self.amplitude * np.exp(1j * self.phase)

    @staticmethod
    def rotation_matrix_from_angles(yaw, pitch):
        """Return Rz(yaw) @ Ry(-pitch); angles are in radians."""
        yaw = float(yaw)
        pitch = float(pitch)
        if not np.isfinite(yaw) or not np.isfinite(pitch):
            raise ValueError("yaw and pitch must be finite")
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        return np.array(
            [
                [cy * cp, -sy, -cy * sp],
                [sy * cp, cy, -sy * sp],
                [sp, 0.0, cp],
            ],
            dtype=float,
        )

    def update_position(self, move_action):
        """更新 UAV 的位置，基于给定的移动动作，并确保其在指定的边界内。"""
        direction = np.clip(_vector(move_action, 2, "move_action"), -1.0, 1.0)
        norm = np.linalg.norm(direction)
        if norm > 1.0:
            direction /= norm
        old_xy = self.position[:2].copy()
        requested_xy = old_xy + self.max_step * direction
        new_xy = np.clip(requested_xy, [self.bounds[0], self.bounds[2]], [self.bounds[1], self.bounds[3]])
        self.position = np.array([new_xy[0], new_xy[1], self.height], dtype=float)
        self.last_requested_displacement = requested_xy - old_xy
        self.last_displacement = new_xy - old_xy
        self.boundary_clipped = not np.allclose(requested_xy, new_xy)
        return self.position.copy()

    def update_orientation(self, orientation_action):
        """Apply normalized yaw/pitch increments and return the realized angles in radians."""
        action = np.clip(_vector(orientation_action, 2, "orientation_action"), -1.0, 1.0)
        requested_yaw_delta = self.max_yaw_step * action[0]
        requested_pitch_delta = self.max_pitch_step * action[1]
        old_pitch = self.pitch

        self.yaw = (self.yaw + requested_yaw_delta + np.pi) % (2.0 * np.pi) - np.pi
        self.pitch = float(np.clip(old_pitch + requested_pitch_delta, self.pitch_min, self.pitch_max))
        self.last_yaw_delta = float(requested_yaw_delta)
        self.last_pitch_delta = float(self.pitch - old_pitch)
        self.orientation_move_ratio = 0.5 * (
            (self.last_yaw_delta / self.max_yaw_step) ** 2
            + (self.last_pitch_delta / self.max_pitch_step) ** 2
        )
        self.rotation_matrix = self.rotation_matrix_from_angles(self.yaw, self.pitch)
        self.normal = self.rotation_matrix[:, 0].copy()
        return self.yaw, self.pitch

    def configure_ris(
        self,
        group_mode,
        amplitude_residual,
        phase_residual,
        base_phase,
    ):
        """Compose element coefficients from an analytic base and group residuals."""
        group_mode = _vector(group_mode, self.num_groups, "group_mode")
        amplitude_residual = np.clip(
            _vector(amplitude_residual, self.num_groups, "amplitude_residual"),
            -1.0,
            1.0,
        )
        phase_residual = np.clip(
            _vector(phase_residual, self.num_groups, "phase_residual"),
            -1.0,
            1.0,
        )
        base_phase = _vector(base_phase, self.num_elements, "base_phase")
        if not np.all(np.isin(group_mode, (0.0, 1.0))):
            raise ValueError("group_mode must contain only 0 or 1")

        self.group_mode = group_mode.astype(int)
        self.element_mode = self.group_mode[self.element_to_group]

        if self.max_amplitude == 1.0:
            requested_group_amplitude = np.ones(self.num_groups, dtype=float)
        else:
            amplitude_base = np.sqrt(self.max_amplitude)
            log_span = 0.5 * np.log(self.max_amplitude)
            requested_group_amplitude = np.clip(
                amplitude_base * np.exp(log_span * amplitude_residual),
                1.0,
                self.max_amplitude,
            )
        self.group_amplitude = np.where(
            self.group_mode == 1,
            requested_group_amplitude,
            1.0,
        )
        self.amplitude = self.group_amplitude[self.element_to_group]

        self.base_phase = (base_phase + np.pi) % (2.0 * np.pi) - np.pi
        self.group_phase_residual = self.phase_residual_max * phase_residual
        element_phase_residual = self.group_phase_residual[self.element_to_group]
        self.phase = (
            self.base_phase + element_phase_residual + np.pi
        ) % (2.0 * np.pi) - np.pi
        return self.ris_coeff.copy(), self.active_ris_coeff.copy()

    def configure_elements(self, element_mode, phase_real, phase_imag):
        """Configure per-element active modes and unit-modulus complex phases."""
        element_mode = _vector(element_mode, self.num_elements, "element_mode")
        phase_real = _vector(phase_real, self.num_elements, "phase_real")
        phase_imag = _vector(phase_imag, self.num_elements, "phase_imag")
        if not np.all(np.isin(element_mode, (0.0, 1.0))):
            raise ValueError("element_mode must contain only 0 or 1")

        requested_phase = phase_real + 1j * phase_imag
        magnitude = np.abs(requested_phase)
        unit_phase = np.ones(self.num_elements, dtype=np.complex128)
        nonzero = magnitude > 0.0
        unit_phase[nonzero] = requested_phase[nonzero] / magnitude[nonzero]

        self.element_mode = element_mode.astype(int)
        self.amplitude = np.where(
            self.element_mode == 1,
            self.max_amplitude,
            1.0,
        )
        self.phase = np.angle(unit_phase)
        self.base_phase = self.phase.copy()
        self.group_phase_residual = np.zeros(self.num_groups, dtype=float)

        # Retain these fields for legacy diagnostics.  A value of -1 marks a
        # group whose elements no longer share one active/passive mode.
        self.group_mode = np.empty(self.num_groups, dtype=int)
        self.group_amplitude = np.ones(self.num_groups, dtype=float)
        for group_index in range(self.num_groups):
            group_elements = self.element_to_group == group_index
            modes = self.element_mode[group_elements]
            self.group_mode[group_index] = (
                int(modes[0]) if np.all(modes == modes[0]) else -1
            )
            if np.any(modes == 1):
                self.group_amplitude[group_index] = self.max_amplitude

        return self.ris_coeff.copy(), self.active_ris_coeff.copy()

    def _power_weights(self, h_sat_ris, W, active_to_ris):
        h_sat_ris = np.asarray(h_sat_ris, dtype=np.complex128)
        W = np.asarray(W, dtype=np.complex128)
        active_to_ris = _vector(active_to_ris, self.num_elements, "active_to_ris", dtype=np.complex128)
        if (
            h_sat_ris.ndim != 2
            or h_sat_ris.shape[0] != self.num_elements
            or not np.all(np.isfinite(h_sat_ris))
        ):
            raise ValueError("h_sat_ris must have shape (N, M)")
        if (
            W.ndim != 2
            or h_sat_ris.shape[1] != W.shape[0]
            or not np.all(np.isfinite(W))
        ):
            raise ValueError("W must have shape (M, U)")
        satellite_incident = np.sum(np.abs(h_sat_ris @ W) ** 2, axis=1)
        jammer_incident = self.jammer_power * np.abs(active_to_ris) ** 2
        return satellite_incident + jammer_incident + self.noise_power

    def get_active_ris_power(self, h_sat_ris, W, active_to_ris):
        """计算当前激活的 RIS 元素的总功率，考虑卫星和干扰信号的入射功率。"""
        weights = self._power_weights(h_sat_ris, W, active_to_ris)
        return float(np.sum(weights * np.abs(self.active_ris_coeff) ** 2))

    def enforce_power_constraint(self, h_sat_ris, W, active_to_ris):
        """确保当前激活的 RIS 元素满足最大功率约束，如果不满足，则按比例缩放幅度。"""
        self.last_power_projection_scale = 1.0
        weights = self._power_weights(h_sat_ris, W, active_to_ris)
        active = self.element_mode.astype(bool)
        self.amplitude[~active] = 1.0
        if not np.any(active):
            self.group_amplitude.fill(1.0)
            return 0.0

        active_weight = float(np.sum(weights[active]))
        if active_weight > self.max_power:
            self.amplitude[active] = 1.0
            self.last_power_projection_scale = 0.0
            raise RISPowerInfeasibleError(
                "selected active elements exceed ris_max_power even at unit gain"
            )

        if active_weight <= 0.0:
            active_gain = self.max_amplitude
        else:
            active_gain = max(
                1.0,
                min(
                    self.max_amplitude,
                    float(np.sqrt(self.max_power / active_weight)),
                ),
            )
        self.amplitude[active] = active_gain

        self.group_amplitude.fill(1.0)
        for group_index in range(self.num_groups):
            group_elements = self.element_to_group == group_index
            if np.any(active[group_elements]):
                self.group_amplitude[group_index] = active_gain

        if self.max_amplitude > 1.0:
            self.last_power_projection_scale = float(
                (active_gain - 1.0) / (self.max_amplitude - 1.0)
            )
        return self.get_active_ris_power(h_sat_ris, W, active_to_ris)

    def get_position(self):
        return self.position.copy()
