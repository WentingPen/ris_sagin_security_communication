import argparse
import math


BOLTZMANN_CONSTANT = 1.380649e-23


def thermal_noise_power(bandwidth_hz, noise_figure_db, temperature_k):
    return BOLTZMANN_CONSTANT * float(temperature_k) * float(bandwidth_hz) * 10.0 ** (float(noise_figure_db) / 10.0)


def validate_system_params(args):
    for name, value in vars(args).items():
        values = value if isinstance(value, (list, tuple)) else (value,)

        for item in values:
            if isinstance(item, (int, float)) and not math.isfinite(float(item)):
                raise ValueError(f"{name} must contain only finite values")

    positive = (
        "num_users",
        "num_sat_antennas",
        "num_ris_elements",
        "num_ris_groups",
        "sat_max_power",
        "ris_max_power",
        "uav_height",
        "ris_max_yaw_step_deg",
        "ris_max_pitch_step_deg",
        "carrier_freq",
        "c",
        "bandwidth_hz",
        "noise_temperature_k",
        "array_spacing_wavelengths",
        "sat_array_rows",
        "ris_array_rows",
        "ris_group_rows",
        "ris_group_cols",
        "ris_phase_residual_max_deg",
        "ground_array_rows",
        "g2g_path_loss_exponent",
        "max_hcpp_resamples",
        "max_episode_steps",
        "risk_grid_size",
        "risk_distance_scale",
        "risk_density_reference",
        "satellite_position_scale",
    )

    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    nonnegative = (
        "jammer_power",
        "ris_noise_power",
        "uav_max_step",
        "residual_si_factor",
        "self_interference_channel_gain",
        "bob_protect_radius",
        "eve_min_distance",
        "bg_density",
        "reward_outage_weight",
        "reward_margin_weight",
        "reward_sat_power_weight",
        "reward_ris_power_weight",
        "reward_move_weight",
        "reward_orientation_weight",
        "infeasible_ris_penalty",
        "ris_angle_gain_exponent",
        "risk_distance_weight",
        "risk_density_weight",
        "risk_sensing_weight",
        "risk_eavesdrop_weight",
        "risk_jamming_weight",
    )

    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")

    if args.ris_max_amplitude < 1.0:
        raise ValueError("ris_max_amplitude must be at least 1")

    if args.bob_noise_power <= 0 or args.eve_noise_power <= 0:
        raise ValueError("bob_noise_power and eve_noise_power must be positive")

    if min(args.bob_noise_figure_db, args.eve_noise_figure_db, args.ris_noise_figure_db) < 0.0:
        raise ValueError("noise figures must be non-negative")

    for lower_name, upper_name in (
        ("channel_gain_db_min", "channel_gain_db_max"),
        ("sinr_db_min", "sinr_db_max"),
        ("inr_db_min", "inr_db_max"),
    ):
        if getattr(args, lower_name) >= getattr(args, upper_name):
            raise ValueError(f"{lower_name} must be smaller than {upper_name}")

    if args.residual_si_factor > 1.0:
        raise ValueError("residual_si_factor must not exceed 1")

    if args.num_ris_elements % args.num_ris_groups != 0:
        raise ValueError("num_ris_elements must be divisible by num_ris_groups")

    if args.num_sat_antennas % args.sat_array_rows != 0:
        raise ValueError("num_sat_antennas must be divisible by sat_array_rows")

    if args.num_ris_elements % args.ris_array_rows != 0:
        raise ValueError("num_ris_elements must be divisible by ris_array_rows")

    ris_cols = args.num_ris_elements // args.ris_array_rows
    if args.ris_array_rows % args.ris_group_rows != 0:
        raise ValueError("ris_array_rows must be divisible by ris_group_rows")
    if ris_cols % args.ris_group_cols != 0:
        raise ValueError("RIS array columns must be divisible by ris_group_cols")
    expected_groups = (
        args.ris_array_rows // args.ris_group_rows
    ) * (
        ris_cols // args.ris_group_cols
    )
    if expected_groups != args.num_ris_groups:
        raise ValueError(
            "num_ris_groups does not match the configured physical group "
            f"layout: expected {expected_groups}, got {args.num_ris_groups}"
        )
    if args.ris_phase_residual_max_deg > 180.0:
        raise ValueError("ris_phase_residual_max_deg must not exceed 180 degrees")

    if args.ground_array_rows != 1:
        raise ValueError("ground_array_rows must be 1 for the current single-antenna ground nodes")

    if len(args.secrecy_rate_threshold) not in (1, args.num_users):
        raise ValueError("secrecy_rate_threshold must contain 1 or num_users values")

    if any(value <= 0 for value in args.secrecy_rate_threshold):
        raise ValueError("secrecy_rate_threshold must be positive")

    if len(args.peak_density) not in (1, args.num_users):
        raise ValueError("peak_density must contain 1 or num_users values")

    if any(value < 0 for value in args.peak_density):
        raise ValueError("peak_density must be non-negative")

    if len(args.eve_cluster_sigma) not in (1, args.num_users):
        raise ValueError("eve_cluster_sigma must contain 1 or num_users values")

    if any(value <= 0 for value in args.eve_cluster_sigma):
        raise ValueError("eve_cluster_sigma must be positive")

    if len(args.area_bounds) != 4:
        raise ValueError("area_bounds must contain four values")

    x_min, x_max, y_min, y_max = args.area_bounds
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("area_bounds must be [x_min, x_max, y_min, y_max]")

    if args.ris_array_plane != "yz":
        raise ValueError(
            "ris_array_plane is deprecated for dynamic RIS orientation; migrate to the local yz plane (ris_array_plane='yz')"
        )
    if len(args.ris_normal) != 3 or any(
        not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
        for value, expected in zip(args.ris_normal, (1.0, 0.0, 0.0))
    ):
        raise ValueError(
            "ris_normal is deprecated for dynamic RIS orientation; use the default [1, 0, 0] and set yaw/pitch instead"
        )

    if not -90.0 <= args.ris_pitch_min_deg < args.ris_pitch_max_deg <= 90.0:
        raise ValueError("RIS pitch limits must satisfy -90 <= ris_pitch_min_deg < ris_pitch_max_deg <= 90")
    if not args.ris_pitch_min_deg <= args.ris_initial_pitch_deg <= args.ris_pitch_max_deg:
        raise ValueError("ris_initial_pitch_deg must lie within the configured pitch limits")

    risk_weight_sum = (args.risk_distance_weight + args.risk_density_weight + args.risk_sensing_weight)

    if risk_weight_sum <= 0:
        raise ValueError("at least one risk weight must be positive")

    if args.risk_eavesdrop_weight + args.risk_jamming_weight <= 0:
        raise ValueError("at least one sensing-risk weight must be positive")

    if args.bg_density + sum(args.peak_density) <= 0:
        raise ValueError("the parent Eve intensity must be positive")

    if args.seed is not None and args.seed < 0:
        raise ValueError("seed must be non-negative")

    if args.risk_grid_size < 2:
        raise ValueError("risk_grid_size must be at least 2")

    if not 0.0 < args.jammer_beamwidth_deg <= 180.0:
        raise ValueError("jammer_beamwidth_deg must be in (0, 180]")

    if not 0.0 <= args.jammer_max_steering_offset_deg <= 180.0:
        raise ValueError("jammer_max_steering_offset_deg must be in [0, 180]")

    if args.jammer_max_attenuation_db < 0.0 or args.jammer_leakage_weight < 0.0:
        raise ValueError("jammer attenuation and leakage weight must be non-negative")

    if args.jammer_steering_samples < 1 or args.jammer_steering_samples % 2 == 0:
        raise ValueError("jammer_steering_samples must be a positive odd integer")

    if not 0.0 <= args.passive_eve_cancel_factor <= 1.0:
        raise ValueError("passive_eve_cancel_factor must be in [0, 1]")

    if not 0.0 <= args.active_eve_echo_cancel_factor <= 1.0:
        raise ValueError("active_eve_echo_cancel_factor must be in [0, 1]")

    return args


def get_system_params(argv=None):
    parser = argparse.ArgumentParser(description='System Parameters')

    # ==============================
    # System
    # ==============================
    parser.add_argument('--num_users', '--num_user', dest='num_users', type=int, default=1, help='Number of users')
    # 当前物理模型只支持各一个
    parser.add_argument("--num_satellites", type=int, choices=[1], default=1)
    parser.add_argument("--num_uavs", type=int, choices=[1], default=1)
    parser.add_argument("--num_ris", type=int, choices=[1], default=1)

    parser.add_argument("--num_sat_antennas", "--num_antennas", dest="num_sat_antennas", type=int, default=16)
    parser.add_argument("--num_ris_elements", type=int, default=64)
    parser.add_argument("--num_ris_groups", type=int, default=16)
    parser.add_argument("--ris_group_rows", type=int, default=2, help="Number of physical RIS rows in each control group")
    parser.add_argument("--ris_group_cols", type=int, default=2, help="Number of physical RIS columns in each control group")
    parser.add_argument("--ris_phase_residual_max_deg", type=float, default=45.0, help="Maximum absolute PPO phase residual for each RIS group")

    parser.add_argument("--sat_max_power", "--transmit_power", dest="sat_max_power", type=float, default=2.5, help="Satellite total transmit-power limit in W")
    parser.add_argument("--jammer_power", type=float, default=0, help="Active Eve transmit power before antenna gain in W")

    parser.add_argument("--ris_max_amplitude", type=float, default=12.0)
    parser.add_argument("--ris_max_power", type=float, default=1, help="Active-element output RF-power limit in W")
    parser.add_argument("--ris_noise_power", "--noise_power", dest="ris_noise_power", type=float, default=None, help="Optional active-RIS input-noise power override in W")
    parser.add_argument("--bob_noise_power", type=float, default=None, help="Optional Bob receiver-noise power override in W")
    parser.add_argument("--eve_noise_power", type=float, default=None, help="Optional Eve receiver-noise power override in W")

    parser.add_argument("--uav_height", type=float, default=100.0)
    parser.add_argument("--uav_max_step", "--ris_max_step", dest="uav_max_step", type=float, default=100.0)
    parser.add_argument("--ris_initial_yaw_deg", type=float, default=0.0)
    parser.add_argument("--ris_initial_pitch_deg", type=float, default=22.5)
    parser.add_argument("--ris_max_yaw_step_deg", type=float, default=10.0)
    parser.add_argument("--ris_max_pitch_step_deg", type=float, default=10.0)
    parser.add_argument("--ris_pitch_min_deg", type=float, default=-60.0)
    parser.add_argument("--ris_pitch_max_deg", type=float, default=60.0)
    parser.add_argument("--area_bounds", type=float, nargs=4, default=[-5000.0, 5000.0, -5000.0, 5000.0])
    parser.add_argument("--ground_height", type=float, default=0.0)

    parser.add_argument("--residual_si_factor", type=float, default=1e-10, help="Residual direct self-interference power fraction after cancellation")
    parser.add_argument("--self_interference_channel_gain", type=float, default=1.0, help="Direct self-interference channel power gain")

    # ==============================
    # Channel model
    # ==============================
    parser.add_argument("--carrier_freq", type=float, default=2.0e9, help="Carrier frequency in Hz")
    parser.add_argument("--c", type=float, default=299_792_458.0, help="Speed of light in m/s")
    parser.add_argument("--use_3d_array_response", type=int, choices=[0, 1], default=1)
    parser.add_argument("--bandwidth_hz", type=float, default=1e6, help="Modeled receiver bandwidth in Hz")
    parser.add_argument("--noise_temperature_k", type=float, default=290.0, help="Reference noise temperature in K")
    parser.add_argument("--bob_noise_figure_db", type=float, default=7.0, help="Bob receiver noise figure in dB")
    parser.add_argument("--eve_noise_figure_db", type=float, default=7.0, help="Eve receiver noise figure in dB")
    parser.add_argument("--ris_noise_figure_db", type=float, default=6.0, help="Equivalent active-RIS input noise figure in dB")

    parser.add_argument('--a2g_a', type=float, default=9.61, help='Environment coefficient a for A2G LoS probability')
    parser.add_argument('--a2g_b', type=float, default=0.16, help='Environment coefficient b for A2G LoS probability')
    parser.add_argument('--eta_los', type=float, default=1.0, help='Additional LoS attenuation in dB')
    parser.add_argument('--eta_nlos', type=float, default=20.0, help='Additional NLoS attenuation in dB')
    parser.add_argument('--rician_factor_a2g_db', type=float, default=10.0, help='A2G Rician K-factor in dB')
    parser.add_argument('--atmospheric_loss_db', type=float, default=2.0, help='Satellite-ground atmospheric attenuation in dB')
    parser.add_argument('--s2a_atmospheric_loss_db', type=float, default=2.0, help='Satellite-RIS atmospheric attenuation in dB')
    parser.add_argument('--sat_tx_gain_db', type=float, default=18.0, help='Satellite antenna element gain in dB')
    parser.add_argument('--ground_rx_gain_db', type=float, default=0.0, help='Ground receiver antenna gain in dB')
    parser.add_argument('--ris_link_gain_db', type=float, default=3.0, help='RIS element antenna gain in dB')
    parser.add_argument('--array_spacing_wavelengths', type=float, default=0.5, help='Antenna spacing normalized by wavelength')
    parser.add_argument('--sat_array_rows', type=int, default=4, help='Number of rows of satellite UPA')
    parser.add_argument('--ris_array_rows', type=int, default=16, help='Number of rows of RIS UPA')
    parser.add_argument('--ground_array_rows', type=int, default=1, help='Number of rows of ground antenna array')
    parser.add_argument('--sat_array_plane', type=str, default='xy', choices=['xy', 'xz', 'yz'], help='Satellite UPA plane')
    parser.add_argument('--ris_array_plane', type=str, default='yz', choices=['xy', 'xz', 'yz'], help='RIS UPA plane')
    parser.add_argument('--ground_array_plane', type=str, default='xy', choices=['xy', 'xz', 'yz'], help='Ground antenna array plane')
    parser.add_argument('--ris_normal', type=float, nargs=3, default=[1.0, 0.0, 0.0], help='RIS surface normal vector')
    parser.add_argument('--enforce_ris_half_space', type=int, choices=[0, 1], default=1, help='Whether to enforce RIS front-side visibility')
    parser.add_argument('--ris_angle_gain_exponent', type=float, default=1.0, help='Exponent q of the RIS power-domain angular gain')
    parser.add_argument('--g2g_reference_gain_db', type=float, default=-38.5, help='G2G reference channel power gain at 1 m in dB')
    parser.add_argument('--g2g_path_loss_exponent', type=float, default=3.0, help='G2G path-loss exponent')

    # ==============================
    # HCPP Eve process
    # ==============================
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    parser.add_argument("--bob_protect_radius", type=float, default=100.0, help="Minimum distance from Bob to any Eve")
    parser.add_argument("--eve_min_distance", type=float, default=50.0, help="Minimum distance between any two Eves")
    parser.add_argument("--bg_density", type=float, default=5e-8, help="Background Eve density in 1/m^2")
    parser.add_argument("--peak_density", type=float, nargs="+", default=[1e-5], help="Peak Eve density in 1/m^2")
    parser.add_argument("--eve_cluster_sigma", type=float, nargs="+", default=[150.0], help="Standard deviation of Eve cluster distribution in meters")
    parser.add_argument("--max_hcpp_resamples", type=int, default=100, help="Maximum number of resampling attempts for HCPP Eve process")

    # ==============================
    # Environment and reward
    # ==============================
    parser.add_argument("--max_episode_steps", type=int, default=200, help="Maximum number of steps per episode")
    parser.add_argument("--secrecy_rate_threshold", type=float, nargs="+", default=[0.1], help="Scalar or one threshold per user, bit/s/Hz")
    parser.add_argument("--channel_gain_db_min", type=float, default=-180.0, help="Fixed lower bound for channel-gain normalization")
    parser.add_argument("--channel_gain_db_max", type=float, default=-60.0, help="Fixed upper bound for channel-gain normalization")
    parser.add_argument("--sinr_db_min", type=float, default=-30.0, help="Fixed lower bound for SINR normalization")
    parser.add_argument("--sinr_db_max", type=float, default=30.0, help="Fixed upper bound for SINR normalization")
    parser.add_argument("--inr_db_min", type=float, default=-30.0, help="Fixed lower bound for jammer-INR normalization")
    parser.add_argument("--inr_db_max", type=float, default=60.0, help="Fixed upper bound for jammer-INR normalization")

    parser.add_argument("--reward_outage_weight", "--reward_sop_weight", dest="reward_outage_weight", type=float, default=1.0, help="Weight of the one-step secrecy-outage indicator")
    parser.add_argument("--reward_margin_weight", type=float, default=1.0, help="Weight of the one-step secrecy-rate margin")
    parser.add_argument("--reward_sat_power_weight", type=float, default=0.0, help="Weight of the one-step satellite power")
    parser.add_argument("--reward_ris_power_weight", type=float, default=0.01, help="Weight of the one-step RIS power")
    parser.add_argument("--reward_move_weight", type=float, default=0.01, help="Weight of the one-step movement")
    parser.add_argument("--reward_orientation_weight", type=float, default=0.01, help="Weight of the one-step RIS orientation movement")
    parser.add_argument("--infeasible_ris_penalty", type=float, default=1.0, help="Penalty for infeasible RIS configurations")


    # ==============================
    # Eve risk graph
    # ==============================
    parser.add_argument("--risk_grid_size", type=int, default=100, help="Number of grid points along each axis for risk graph")
    parser.add_argument("--risk_distance_scale", type=float, default=200.0, help="Exponential distance-decay scale in meters")
    parser.add_argument("--risk_distance_weight", type=float, default=0.4, help="Weight of the risk distance component")
    parser.add_argument("--risk_density_weight", type=float, default=0.3, help="Weight of the risk density component")
    parser.add_argument("--risk_sensing_weight", type=float, default=0.3, help="Weight of the risk sensing component")
    parser.add_argument("--risk_eavesdrop_weight", type=float, default=0.5, help="Weight of the risk eavesdropping component")
    parser.add_argument("--risk_jamming_weight", type=float, default=0.5, help="Weight of the risk jamming component")
    parser.add_argument("--risk_density_reference", type=float, default=1e-5, help="Reference density used to normalize HCPP risk features")
    parser.add_argument("--satellite_position_scale", type=float, default=1e6, help="Scale factor for satellite position")

    # ==============================
    # active eve parameters
    # ==============================   
    parser.add_argument("--jammer_main_gain_db", "--jammer_tx_gain_db", dest="jammer_main_gain_db", type=float, default=8.0, help="Active Eve/Jammer peak transmit gain in dB")
    parser.add_argument("--jammer_beamwidth_deg", type=float, default=60.0, help="Active Eve/Jammer full 3-dB beamwidth in degrees")
    parser.add_argument("--jammer_max_attenuation_db", type=float, default=25.0, help="Active Eve/Jammer maximum attenuation in dB")
    parser.add_argument("--jammer_max_steering_offset_deg", type=float, default=30.0, help="Active Eve/Jammer maximum steering offset in degrees")
    parser.add_argument("--jammer_steering_samples", type=int, default=31, help="Number of steering samples for the jammer")
    parser.add_argument("--jammer_leakage_weight", type=float, default=1.0, help="Weight of the jammer leakage component")
    parser.add_argument("--passive_eve_cancel_factor", type=float, default=1.0, help="Residual jammer-power fraction at a passive Eve after cancellation")
    parser.add_argument("--active_eve_echo_cancel_factor", type=float, default=0.05, help="Residual RIS-echo power fraction at the active Eve after cancellation")



    args = parser.parse_args(argv)

    if args.bob_noise_power is None:
        args.bob_noise_power = thermal_noise_power(args.bandwidth_hz, args.bob_noise_figure_db, args.noise_temperature_k)
    if args.eve_noise_power is None:
        args.eve_noise_power = thermal_noise_power(args.bandwidth_hz, args.eve_noise_figure_db, args.noise_temperature_k)
    if args.ris_noise_power is None:
        args.ris_noise_power = thermal_noise_power(args.bandwidth_hz, args.ris_noise_figure_db, args.noise_temperature_k)

    # 兼容当前各模块尚未统一的旧参数名。
    args.num_antennas = args.num_sat_antennas
    args.transmit_power = args.sat_max_power
    args.ris_max_step = args.uav_max_step
    args.noise_power = args.ris_noise_power
    args.reward_sop_weight = args.reward_outage_weight

    return validate_system_params(args)
