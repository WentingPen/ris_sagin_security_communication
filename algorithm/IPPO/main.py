import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import traceback


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import matplotlib
import torch

from algorithm.IPPO.Controller import Controller, IPPOConfig
from algorithm.ppo.file_io import replace_file
from algorithm.ppo.main import (
    LiveTrainingPlotter as MAPPOTrainingPlotter,
    _build_environment,
    _move_optimizer_state,
)
from system_model.params import get_system_params, validate_system_params


ALGORITHM = "IPPO"
ACTION_SCHEMA = "satellite_conditioned_ris_group_residual_v3"
CRITIC_OBSERVATION_SCHEMA = "public_vector_risk_nodes_v1"
MODEL_NAMES = (
    "satellite_actor", "haps_ris_actor", "satellite_critic", "haps_ris_critic",
)
EPISODE_FIELDS = [
    "episode", "env_steps", "policy_version", "return", "mean_reward",
    "length", "outage_rate", "num_eves", "mean_raw_security_gap",
    "mean_bob_capacity", "mean_max_eve_capacity", "max_bob_capacity",
    "max_eve_capacity", "mean_bob_jammer_inr", "ris_repair_rate",
]
UPDATE_FIELDS = [
    "update", "total_episodes", "total_env_steps", "mean_episode_return",
    "mean_outage_rate", "satellite_actor_loss", "haps_ris_actor_loss",
    "actor_loss", "critic_loss", "satellite_entropy", "haps_ris_entropy",
    "entropy", "satellite_approx_kl", "haps_ris_approx_kl", "approx_kl",
    "satellite_clip_fraction", "haps_ris_clip_fraction", "clip_fraction",
    "entropy_coef", "satellite_early_stop", "haps_ris_early_stop",
    "satellite_stop_kl", "haps_ris_stop_kl",
    "satellite_critic_loss", "haps_ris_critic_loss",
]


def _parse_training_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or resume the independent satellite-RIS IPPO controller",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--total-updates", "--total_updates", type=int, default=1000,
        help="Target total update count, including completed updates",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None,
        help="Defaults to CUDA when available; use cpu for isolated verification",
    )
    parser.add_argument(
        "--phase-only-training", "--phase_only_training", "--phase-only",
        action="store_true",
        help="Use analytic MRT and an all-passive RIS; learn only phase residuals",
    )
    parser.add_argument(
        "--phase-only-csi-seed", "--phase_only_csi_seed", type=int, default=None,
    )
    args, system_argv = parser.parse_known_args(argv)
    if args.total_updates <= 0:
        parser.error("--total-updates must be positive")
    if args.phase_only_csi_seed is not None and (
        args.phase_only_csi_seed < 0 or not args.phase_only_training
    ):
        parser.error("--phase-only-csi-seed requires phase-only training and a non-negative seed")
    if args.resume is not None and (
        system_argv or args.phase_only_training or args.phase_only_csi_seed is not None
    ):
        parser.error("a resumed run must reuse the checkpoint system and IPPO configuration")
    return args, system_argv


def _load_resume_checkpoint(path):
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("algorithm") != ALGORITHM:
        raise ValueError("only IPPO checkpoints can be resumed; MAPPO checkpoints are incompatible")
    if checkpoint.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("checkpoint uses an incompatible action schema")
    if checkpoint.get("critic_observation_schema") != CRITIC_OBSERVATION_SCHEMA:
        raise ValueError("checkpoint must use public-observation local critics")
    required = {
        "update", "total_episodes", "total_env_steps", "ppo_config", "system_params",
        *MODEL_NAMES, *(name + "_optimizer" for name in MODEL_NAMES),
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError("checkpoint is incomplete; missing keys: " + ", ".join(sorted(missing)))
    for name in ("update", "total_episodes", "total_env_steps"):
        if int(checkpoint[name]) < 0:
            raise ValueError(f"checkpoint {name} must be non-negative")
    params = validate_system_params(argparse.Namespace(**checkpoint["system_params"]))
    config = IPPOConfig(**checkpoint["ppo_config"])
    return checkpoint_path, checkpoint, params, config


def _restore_controller(controller, checkpoint):
    for name in MODEL_NAMES:
        getattr(controller, name).load_state_dict(checkpoint[name], strict=True)
        optimizer = getattr(controller, name + "_optimizer")
        optimizer.load_state_dict(checkpoint[name + "_optimizer"])
        _move_optimizer_state(optimizer, controller.device)
    controller.update_index = int(checkpoint["update"])
    controller.total_episodes = int(checkpoint["total_episodes"])
    controller.total_env_steps = int(checkpoint["total_env_steps"])


def _save_checkpoint(path, controller, system_params, resumed_from=None):
    state = {
        "algorithm": ALGORITHM,
        "action_schema": ACTION_SCHEMA,
        "critic_observation_schema": CRITIC_OBSERVATION_SCHEMA,
        "update": controller.update_index,
        "total_episodes": controller.total_episodes,
        "total_env_steps": controller.total_env_steps,
        "ppo_config": vars(controller.config),
        "system_params": vars(system_params),
        "resumed_from": None if resumed_from is None else str(resumed_from),
    }
    for name in MODEL_NAMES:
        state[name] = getattr(controller, name).state_dict()
        state[name + "_optimizer"] = getattr(controller, name + "_optimizer").state_dict()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    replace_file(
        temporary, path,
        fallback=path.with_name(f"{path.stem}_u{controller.update_index:04d}{path.suffix}"),
    )


class LiveTrainingPlotter(MAPPOTrainingPlotter):
    """Keep the MAPPO six-panel figure; also show both independent critics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.satellite_critic_losses = []
        self.haps_ris_critic_losses = []
        self.critic_line.set_label("Mean")
        self.satellite_critic_line, = self.critic_axis.plot(
            [], [], color="tab:blue", label="Satellite",
        )
        self.haps_ris_critic_line, = self.critic_axis.plot(
            [], [], color="tab:orange", label="HAPS-RIS",
        )
        self.actor_axis.set_title("Actor Loss per IPPO Update")
        self.critic_axis.set_title("Independent Critic Loss per IPPO Update")
        self.critic_axis.legend()
        self.figure.tight_layout()
        if self.output_path is not None:
            self.save()

    def on_update(self, metrics):
        self.satellite_critic_losses.append(metrics["satellite_critic_loss"])
        self.haps_ris_critic_losses.append(metrics["haps_ris_critic_loss"])
        episodes = [*self.loss_episode, metrics["total_episodes"]]
        self.satellite_critic_line.set_data(episodes, self.satellite_critic_losses)
        self.haps_ris_critic_line.set_data(episodes, self.haps_ris_critic_losses)
        super().on_update(metrics)


def main(argv=None):
    args, system_argv = _parse_training_args(argv)
    resume_path = None
    checkpoint = None
    if args.resume is None:
        params = get_system_params(system_argv)
        phase_only = args.phase_only_training
        fixed_seed = params.seed if args.phase_only_csi_seed is None else args.phase_only_csi_seed
        config = IPPOConfig(
            gamma=0.0 if phase_only else 0.99,
            gae_lambda=0.0 if phase_only else 0.95,
            entropy_coef=1e-4 if phase_only else 1e-3,
            entropy_coef_final=0.0 if phase_only else 1e-4,
            entropy_anneal_updates=args.total_updates,
            haps_ris_actor_lr=3e-5 if phase_only else 1e-5,
            phase_only_training=phase_only,
            fixed_csi_seed=fixed_seed if phase_only else None,
            phase_log_std_init=-2.0 if phase_only else -0.5,
        )
        seed = fixed_seed if phase_only else params.seed
    else:
        resume_path, checkpoint, params, config = _load_resume_checkpoint(args.resume)
        if args.total_updates <= int(checkpoint["update"]):
            raise ValueError("--total-updates must exceed the completed checkpoint updates")
        seed = (
            (params.seed if config.fixed_csi_seed is None else config.fixed_csi_seed)
            if config.phase_only_training
            else params.seed + int(checkpoint["total_episodes"])
        )
    env = _build_environment(params)
    controller = Controller(env, config=config, seed=seed, device=args.device)
    if checkpoint is not None:
        _restore_controller(controller, checkpoint)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.phase_only_training:
        run_name += "_phase_only"
    if checkpoint is not None:
        run_name += f"_resume_u{controller.update_index:04d}"
    run_dir = Path(__file__).resolve().parent / "result" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Training results will be saved to: {run_dir}", flush=True)
    print(
        f"IPPO device={controller.device}; independent public-observation critics; "
        f"RIS {env.uav_ris.num_groups} groups "
        f"({env.uav_ris.group_rows}x{env.uav_ris.group_cols}) -> "
        f"{env.uav_ris.num_elements} elements; "
        f"phase residuals +/-{config.phase_residual_max_deg:.1f} deg.",
        flush=True,
    )
    if config.phase_only_training:
        print(
            f"Phase-only: analytic MRT, all-passive RIS, replayed seed={config.fixed_csi_seed}.",
            flush=True,
        )
    if checkpoint is not None:
        print(
            f"Resuming {resume_path}; committed update={controller.update_index}, "
            f"episodes={controller.total_episodes}, env_steps={controller.total_env_steps}. "
            "The new result directory records only episodes after this boundary.",
            flush=True,
        )
    plotter = LiveTrainingPlotter(
        smooth_window=20, draw_every=1,
        output_path=run_dir / "training_curves.png", save_every=1,
    )
    print(f"Matplotlib backend={matplotlib.get_backend()}; PNG refreshed every episode.", flush=True)
    with (
        (run_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as episode_file,
        (run_dir / "update_metrics.csv").open("w", newline="", encoding="utf-8") as update_file,
    ):
        episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
        update_writer = csv.DictWriter(update_file, fieldnames=UPDATE_FIELDS)
        episode_writer.writeheader()
        update_writer.writeheader()
        episode_file.flush()
        update_file.flush()

        def handle_episode(event):
            episode_writer.writerow({name: event[name] for name in EPISODE_FIELDS})
            episode_file.flush()
            print(
                f"episode={event['episode']:6d} return={event['return']:.4f} "
                f"mean_reward={event['mean_reward']:.4f} SOP={event['outage_rate']:.4f} "
                f"gap={event['mean_raw_security_gap']:.6f} "
                f"max_bob_C={event['max_bob_capacity']:.6f} "
                f"max_eve_C={event['max_eve_capacity']:.6f}",
                flush=True,
            )
            plotter.on_episode(event)
            if event["episode"] % config.episodes_per_update == 0:
                print("Rollout batch complete; starting IPPO optimization...", flush=True)

        training_error = None
        try:
            for _ in range(controller.update_index, args.total_updates):
                metrics = controller.update(episode_callback=handle_episode)
                _save_checkpoint(run_dir / "checkpoint.pt", controller, params, resume_path)
                update_writer.writerow({name: metrics[name] for name in UPDATE_FIELDS})
                update_file.flush()
                print(
                    f"update={metrics['update']:4d} episodes={metrics['total_episodes']:6d} "
                    f"return={metrics['mean_episode_return']:.4f} "
                    f"SOP={metrics['mean_outage_rate']:.4f} "
                    f"sat_actor_loss={metrics['satellite_actor_loss']:.6f} "
                    f"haps_actor_loss={metrics['haps_ris_actor_loss']:.6f} "
                    f"critic_loss={metrics['critic_loss']:.6f} "
                    f"sat_critic_loss={metrics['satellite_critic_loss']:.6f} "
                    f"haps_critic_loss={metrics['haps_ris_critic_loss']:.6f} "
                    f"entropy_coef={metrics['entropy_coef']:.6f} "
                    f"sat_KL={metrics['satellite_approx_kl']:.6f} "
                    f"haps_KL={metrics['haps_ris_approx_kl']:.6f} "
                    f"KL_stop={int(metrics['satellite_early_stop'])}/"
                    f"{int(metrics['haps_ris_early_stop'])}",
                    flush=True,
                )
                plotter.on_update(metrics)
        except BaseException as error:
            training_error = error
            (run_dir / "training_error.log").write_text(
                traceback.format_exc(), encoding="utf-8",
            )
            raise
        finally:
            if training_error is None:
                plotter.finish()
            else:
                try:
                    plotter.finish()
                except Exception as finish_error:
                    print(f"Final plot could not be saved: {finish_error}", file=sys.stderr)
    return run_dir


if __name__ == "__main__":
    main()
