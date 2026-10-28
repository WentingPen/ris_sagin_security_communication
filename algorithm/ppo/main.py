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
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends import BackendFilter, backend_registry

from algorithm.ppo.Controller import Controller, PPOConfig
from system_model.params import get_system_params, validate_system_params
from system_model.system_env import SAGIN_ENV


ACTION_SCHEMA = "satellite_conditioned_ris_group_residual_v3"


def _parse_training_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or resume the satellite-RIS MAPPO controller",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from; checkpoint parameters are reused",
    )
    parser.add_argument(
        "--total-updates",
        "--total_updates",
        dest="total_updates",
        type=int,
        default=1000,
        help="Target total PPO update count, including completed updates",
    )
    parser.add_argument(
        "--phase-only-training",
        "--phase_only_training",
        "--phase-only",
        action="store_true",
        help=(
            "Use analytic satellite MRT, an all-passive RIS, and train only "
            "per-group phase residuals"
        ),
    )
    parser.add_argument(
        "--phase-only-csi-seed",
        "--phase_only_csi_seed",
        type=int,
        default=None,
        help=(
            "Episode seed replayed in phase-only training; defaults to the "
            "system --seed"
        ),
    )
    training_args, system_argv = parser.parse_known_args(argv)
    if training_args.total_updates <= 0:
        parser.error("--total-updates must be positive")
    if training_args.resume is not None and system_argv:
        parser.error(
            "system parameter overrides are not allowed with --resume; "
            "the checkpoint parameters must remain unchanged"
        )
    if training_args.phase_only_csi_seed is not None:
        if training_args.phase_only_csi_seed < 0:
            parser.error("--phase-only-csi-seed must be non-negative")
        if not training_args.phase_only_training:
            parser.error(
                "--phase-only-csi-seed requires --phase-only-training"
            )
    if training_args.resume is not None and (
        training_args.phase_only_training
        or training_args.phase_only_csi_seed is not None
    ):
        parser.error(
            "phase-only options cannot override a resumed checkpoint; "
            "the checkpoint PPO configuration is reused"
        )
    return training_args, system_argv


def _load_resume_checkpoint(path):
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    required_keys = {
        "action_schema",
        "update",
        "total_episodes",
        "total_env_steps",
        "satellite_actor",
        "haps_ris_actor",
        "critic",
        "satellite_actor_optimizer",
        "haps_ris_actor_optimizer",
        "critic_optimizer",
        "ppo_config",
        "system_params",
    }
    missing_keys = sorted(required_keys.difference(checkpoint))
    if missing_keys:
        raise ValueError(
            "resume checkpoint is incomplete; missing keys: "
            + ", ".join(missing_keys)
        )
    if checkpoint["action_schema"] != ACTION_SCHEMA:
        raise ValueError(
            "resume checkpoint uses an incompatible action schema: "
            f"{checkpoint['action_schema']!r}"
        )

    for name in ("update", "total_episodes", "total_env_steps"):
        if int(checkpoint[name]) < 0:
            raise ValueError(f"checkpoint {name} must be non-negative")

    system_params = validate_system_params(
        argparse.Namespace(**checkpoint["system_params"])
    )
    ppo_config = PPOConfig(**checkpoint["ppo_config"])
    return checkpoint_path, checkpoint, system_params, ppo_config


def _build_environment(system_params):
    x_min, x_max, y_min, y_max = system_params.area_bounds
    area_center = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)])
    sat_positions = np.array([[
        area_center[0], area_center[1], system_params.satellite_height
    ]])
    uav_positions = np.array([[
        system_params.ris_fixed_x,
        system_params.ris_fixed_y,
        system_params.uav_height,
    ]])
    user_positions = np.array([[
        100.0, 0.0, system_params.ground_height
    ]])
    return SAGIN_ENV(
        params=system_params,
        sat_positions=sat_positions,
        uav_positions=uav_positions,
        user_positions=user_positions,
    )


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _restore_controller(controller, checkpoint):
    controller.satellite_actor.load_state_dict(
        checkpoint["satellite_actor"], strict=True
    )
    controller.haps_ris_actor.load_state_dict(
        checkpoint["haps_ris_actor"], strict=True
    )
    controller.critic.load_state_dict(checkpoint["critic"], strict=True)
    controller.satellite_actor_optimizer.load_state_dict(
        checkpoint["satellite_actor_optimizer"]
    )
    controller.haps_ris_actor_optimizer.load_state_dict(
        checkpoint["haps_ris_actor_optimizer"]
    )
    controller.critic_optimizer.load_state_dict(
        checkpoint["critic_optimizer"]
    )
    for optimizer in (
        controller.satellite_actor_optimizer,
        controller.haps_ris_actor_optimizer,
        controller.critic_optimizer,
    ):
        _move_optimizer_state(optimizer, controller.device)

    controller.update_index = int(checkpoint["update"])
    controller.total_episodes = int(checkpoint["total_episodes"])
    controller.total_env_steps = int(checkpoint["total_env_steps"])


class LiveTrainingPlotter:
    def __init__(
        self,
        smooth_window=20,
        draw_every=1,
        output_path=None,
        save_every=1,
        interactive=None,
    ):
        self.smooth_window = int(smooth_window)
        self.draw_every = int(draw_every)
        self.save_every = int(save_every)
        if min(self.smooth_window, self.draw_every, self.save_every) <= 0:
            raise ValueError("smooth_window, draw_every, and save_every must be positive")
        self.output_path = None if output_path is None else Path(output_path)

        if interactive is None:
            interactive_backends = {
                name.lower()
                for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
            }
            interactive = matplotlib.get_backend().lower() in interactive_backends
        self.interactive = bool(interactive)

        self.episodes = []
        self.sops = []
        self.sop_mean = []
        self.returns = []
        self.return_mean = []
        self.max_bob_capacities = []
        self.max_bob_capacity_mean = []
        self.max_eve_capacities = []
        self.max_eve_capacity_mean = []
        self.loss_episode = []
        self.satellite_actor_losses = []
        self.haps_ris_actor_losses = []
        self.critic_losses = []

        if self.interactive:
            plt.ion()
        self.figure, axes = plt.subplots(3, 2, figsize=(14, 11))
        self.actor_axis, self.critic_axis = axes[0]
        self.sop_axis, self.reward_axis = axes[1]
        self.bob_capacity_axis, self.eve_capacity_axis = axes[2]

        self.satellite_actor_line, = self.actor_axis.plot(
            [], [], color="tab:blue", marker="o", markersize=3,
            label="Satellite",
        )
        self.haps_ris_actor_line, = self.actor_axis.plot(
            [], [], color="tab:orange", marker="o", markersize=3,
            label="HAPS-RIS",
        )
        self.critic_line, = self.critic_axis.plot(
            [], [], color="tab:purple", marker="o", markersize=3
        )
        self.sop_raw, = self.sop_axis.plot(
            [], [], alpha=0.3, color="tab:green", label="raw"
        )
        self.sop_avg, = self.sop_axis.plot(
            [], [], color="tab:green", label="moving average"
        )
        self.reward_raw, = self.reward_axis.plot(
            [], [], alpha=0.3, color="tab:red", label="raw"
        )
        self.reward_avg, = self.reward_axis.plot(
            [], [], color="tab:red", label="moving average"
        )
        self.bob_capacity_raw, = self.bob_capacity_axis.plot(
            [], [], alpha=0.3, color="tab:cyan", label="raw"
        )
        self.bob_capacity_avg, = self.bob_capacity_axis.plot(
            [], [], color="tab:cyan", label="moving average"
        )
        self.eve_capacity_raw, = self.eve_capacity_axis.plot(
            [], [], alpha=0.3, color="tab:pink", label="raw"
        )
        self.eve_capacity_avg, = self.eve_capacity_axis.plot(
            [], [], color="tab:pink", label="moving average"
        )

        self.actor_axis.set_title("Actor Loss per PPO Update")
        self.critic_axis.set_title("Critic Loss per PPO Update")
        self.sop_axis.set_title("Episode SOP")
        self.reward_axis.set_title("Episode Return")
        self.bob_capacity_axis.set_title("Mean Step-Max Bob Capacity per Episode")
        self.eve_capacity_axis.set_title("Mean Step-Max Eve Capacity per Episode")

        for axis in axes.ravel():
            axis.set_xlabel("Completed episodes")
            axis.grid(alpha=0.25)

        self.actor_axis.set_ylabel("Actor loss")
        self.critic_axis.set_ylabel("Critic loss")
        self.sop_axis.set_ylabel("SOP")
        self.sop_axis.set_ylim(-0.02, 1.02)
        self.reward_axis.set_ylabel("Return")
        self.bob_capacity_axis.set_ylabel("Capacity (bit/s/Hz)")
        self.eve_capacity_axis.set_ylabel("Capacity (bit/s/Hz)")
        waiting_message = "Waiting for the first PPO update"
        self.actor_waiting = self.actor_axis.text(
            0.5,
            0.5,
            waiting_message,
            ha="center",
            va="center",
            transform=self.actor_axis.transAxes,
            color="0.45",
        )
        self.critic_waiting = self.critic_axis.text(
            0.5,
            0.5,
            waiting_message,
            ha="center",
            va="center",
            transform=self.critic_axis.transAxes,
            color="0.45",
        )
        self.actor_axis.legend()
        self.sop_axis.legend()
        self.reward_axis.legend()
        self.bob_capacity_axis.legend()
        self.eve_capacity_axis.legend()
        self.figure.tight_layout()
        if self.interactive:
            plt.show(block=False)
            self._draw()
        if self.output_path is not None:
            self.save()

    def on_episode(self, event):
        self.episodes.append(event["episode"])
        self.sops.append(event["outage_rate"])
        self.returns.append(event["return"])
        self.max_bob_capacities.append(event["max_bob_capacity"])
        self.max_eve_capacities.append(event["max_eve_capacity"])
        self.sop_mean.append(
            float(np.mean(self.sops[-self.smooth_window:]))
        )
        self.return_mean.append(
            float(np.mean(self.returns[-self.smooth_window:]))
        )
        self.max_bob_capacity_mean.append(
            float(np.mean(self.max_bob_capacities[-self.smooth_window:]))
        )
        self.max_eve_capacity_mean.append(
            float(np.mean(self.max_eve_capacities[-self.smooth_window:]))
        )

        self.sop_raw.set_data(self.episodes, self.sops)
        self.sop_avg.set_data(self.episodes, self.sop_mean)
        self.reward_raw.set_data(self.episodes, self.returns)
        self.reward_avg.set_data(self.episodes, self.return_mean)
        self.bob_capacity_raw.set_data(self.episodes, self.max_bob_capacities)
        self.bob_capacity_avg.set_data(
            self.episodes, self.max_bob_capacity_mean
        )
        self.eve_capacity_raw.set_data(self.episodes, self.max_eve_capacities)
        self.eve_capacity_avg.set_data(
            self.episodes, self.max_eve_capacity_mean
        )
        self._rescale(self.sop_axis)
        self._rescale(self.reward_axis)
        self._rescale(self.bob_capacity_axis)
        self._rescale(self.eve_capacity_axis)

        if len(self.episodes) % self.draw_every == 0:
            self._draw()
        if (
            self.output_path is not None
            and len(self.episodes) % self.save_every == 0
        ):
            self.save()

    def on_update(self, metrics):
        self.loss_episode.append(metrics["total_episodes"])
        self.satellite_actor_losses.append(metrics["satellite_actor_loss"])
        self.haps_ris_actor_losses.append(metrics["haps_ris_actor_loss"])
        self.critic_losses.append(metrics["critic_loss"])
        self.actor_waiting.set_visible(False)
        self.critic_waiting.set_visible(False)
        self.satellite_actor_line.set_data(
            self.loss_episode, self.satellite_actor_losses
        )
        self.haps_ris_actor_line.set_data(
            self.loss_episode, self.haps_ris_actor_losses
        )
        self.critic_line.set_data(self.loss_episode, self.critic_losses)
        self._rescale(self.actor_axis)
        self._rescale(self.critic_axis)
        self._draw()
        if self.output_path is not None:
            self.save()

    @staticmethod
    def _rescale(axis):
        axis.relim()
        axis.autoscale_view()

    def _draw(self):
        if not self.interactive or not plt.fignum_exists(self.figure.number):
            return
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def save(self, path=None):
        path = self.output_path if path is None else Path(path)
        if path is None:
            raise ValueError("an output path is required to save the training figure")
        if not path.suffix:
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
        self.figure.savefig(temporary_path, dpi=150)
        temporary_path.replace(path)
        return path

    def finish(self, path=None):
        if path is not None or self.output_path is not None:
            self.save(path)
        if self.interactive:
            self._draw()
            plt.ioff()
        plt.close(self.figure)


def main(argv=None):
    training_args, system_argv = _parse_training_args(argv)
    resume_path = None
    resume_checkpoint = None
    if training_args.resume is None:
        system_params = get_system_params(system_argv)
        phase_only_training = bool(training_args.phase_only_training)
        fixed_csi_seed = (
            int(system_params.seed)
            if training_args.phase_only_csi_seed is None
            else int(training_args.phase_only_csi_seed)
        )
        ppo_config = PPOConfig(
            episodes_per_update=8,
            batch_size=256,
            ppo_epochs=2,
            gamma=0.0 if phase_only_training else 0.99,
            gae_lambda=0.0 if phase_only_training else 0.95,
            clip_coef=0.1,
            entropy_coef=1e-4 if phase_only_training else 1e-3,
            entropy_coef_final=0.0 if phase_only_training else 1e-4,
            entropy_anneal_updates=training_args.total_updates,
            satellite_target_kl=0.01,
            haps_ris_target_kl=0.02,
            actor_lr=1e-5,
            haps_ris_actor_lr=(
                3e-5 if phase_only_training else 1e-5
            ),
            critic_lr=1e-4,
            max_grad_norm=0.5,
            phase_only_training=phase_only_training,
            fixed_csi_seed=(
                fixed_csi_seed if phase_only_training else None
            ),
            phase_log_std_init=(
                -2.0 if phase_only_training else -0.5
            ),
            phase_residual_max_deg=30.0,
        )
        controller_seed = (
            fixed_csi_seed if phase_only_training else system_params.seed
        )
    else:
        (
            resume_path,
            resume_checkpoint,
            system_params,
            ppo_config,
        ) = _load_resume_checkpoint(training_args.resume)
        completed_updates = int(resume_checkpoint["update"])
        if training_args.total_updates <= completed_updates:
            raise ValueError(
                f"checkpoint already contains {completed_updates} updates; "
                "--total-updates must be larger"
            )
        if ppo_config.phase_only_training:
            controller_seed = (
                int(system_params.seed)
                if ppo_config.fixed_csi_seed is None
                else int(ppo_config.fixed_csi_seed)
            )
        else:
            controller_seed = (
                int(system_params.seed)
                + int(resume_checkpoint["total_episodes"])
            )

    env = _build_environment(system_params)
    controller = Controller(
        env=env,
        config=ppo_config,
        seed=controller_seed,
    )
    if resume_checkpoint is not None:
        _restore_controller(controller, resume_checkpoint)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if ppo_config.phase_only_training:
        run_name += "_phase_only"
    if resume_checkpoint is not None:
        run_name += f"_resume_u{controller.update_index:04d}"
    run_dir = (
        Path(__file__).resolve().parent
        / "result"
        / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    figure_path = run_dir / "training_curves.png"
    checkpoint_path = run_dir / "checkpoint.pt"
    episode_metrics_path = run_dir / "episode_metrics.csv"
    update_metrics_path = run_dir / "update_metrics.csv"
    print(f"Training results will be saved to: {run_dir}")
    print(
        f"RIS control: {env.uav_ris.num_groups} groups "
        f"({env.uav_ris.group_rows}x{env.uav_ris.group_cols}) -> "
        f"{env.uav_ris.num_elements} elements; analytic base + "
        f"group phase residuals (+/-{ppo_config.phase_residual_max_deg:.1f} deg).",
        flush=True,
    )
    if ppo_config.phase_only_training:
        print(
            "Phase-only curriculum enabled: analytic MRT, all-passive RIS, "
            f"phase residual limit=+/-{ppo_config.phase_residual_max_deg:.1f} deg, "
            f"replayed CSI seed={ppo_config.fixed_csi_seed}.",
            flush=True,
        )
    if resume_checkpoint is not None:
        print(
            f"Resuming from: {resume_path}\n"
            f"Committed boundary: update={controller.update_index}, "
            f"episode={controller.total_episodes}, "
            f"env_steps={controller.total_env_steps}\n"
            f"Target: {training_args.total_updates} total updates; "
            f"remaining={training_args.total_updates - controller.update_index}.\n"
            "Any episode CSV rows after the committed boundary belong to an "
            "incomplete rollout and are intentionally not restored.",
            flush=True,
        )
    plotter = LiveTrainingPlotter(
        smooth_window=20,
        draw_every=1,
        output_path=figure_path,
        save_every=1,
    )
    display_mode = "live window + PNG" if plotter.interactive else "PNG only"
    print(
        f"Matplotlib backend={matplotlib.get_backend()}; mode={display_mode}; "
        "training_curves.png is overwritten after every episode.",
        flush=True,
    )

    episode_fields = [
        "episode", "env_steps", "policy_version", "return", "mean_reward",
        "length", "outage_rate", "num_eves", "mean_raw_security_gap",
        "mean_bob_capacity", "mean_max_eve_capacity", "max_bob_capacity",
        "max_eve_capacity",
        "mean_bob_jammer_inr", "ris_repair_rate",
    ]
    update_fields = [
        "update", "total_episodes", "total_env_steps", "mean_episode_return",
        "mean_outage_rate", "satellite_actor_loss", "haps_ris_actor_loss",
        "actor_loss", "critic_loss", "satellite_entropy", "haps_ris_entropy",
        "entropy", "satellite_approx_kl", "haps_ris_approx_kl", "approx_kl",
        "satellite_clip_fraction", "haps_ris_clip_fraction", "clip_fraction",
        "entropy_coef", "satellite_early_stop", "haps_ris_early_stop",
        "satellite_stop_kl", "haps_ris_stop_kl",
    ]
    episode_file = episode_metrics_path.open("w", newline="", encoding="utf-8")
    update_file = update_metrics_path.open("w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
    update_writer = csv.DictWriter(update_file, fieldnames=update_fields)
    episode_writer.writeheader()
    update_writer.writeheader()
    episode_file.flush()
    update_file.flush()

    def handle_episode(event):
        episode_writer.writerow({name: event[name] for name in episode_fields})
        episode_file.flush()
        print(
            f"episode={event['episode']:6d} "
            f"return={event['return']:.4f} "
            f"mean_reward={event['mean_reward']:.4f} "
            f"SOP={event['outage_rate']:.4f} "
            f"gap={event['mean_raw_security_gap']:.6f} "
            f"max_bob_C={event['max_bob_capacity']:.6f} "
            f"max_eve_C={event['max_eve_capacity']:.6f}",
            flush=True,
        )
        plotter.on_episode(event)
        if event["episode"] % ppo_config.episodes_per_update == 0:
            print(
                "Rollout batch complete; starting PPO optimization...",
                flush=True,
            )

    training_error = None
    try:
        for _ in range(controller.update_index, training_args.total_updates):
            metrics = controller.update(
                episode_callback=handle_episode
            )

            temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "action_schema": ACTION_SCHEMA,
                    "update": controller.update_index,
                    "total_episodes": controller.total_episodes,
                    "total_env_steps": controller.total_env_steps,
                    "satellite_actor": controller.satellite_actor.state_dict(),
                    "haps_ris_actor": controller.haps_ris_actor.state_dict(),
                    "critic": controller.critic.state_dict(),
                    "satellite_actor_optimizer": (
                        controller.satellite_actor_optimizer.state_dict()
                    ),
                    "haps_ris_actor_optimizer": (
                        controller.haps_ris_actor_optimizer.state_dict()
                    ),
                    "critic_optimizer": controller.critic_optimizer.state_dict(),
                    "ppo_config": vars(ppo_config),
                    "system_params": vars(system_params),
                    "resumed_from": (
                        None if resume_path is None else str(resume_path)
                    ),
                },
                temporary_checkpoint,
            )
            temporary_checkpoint.replace(checkpoint_path)
            update_writer.writerow({name: metrics[name] for name in update_fields})
            update_file.flush()

            print(
                f"update={metrics['update']:4d} "
                f"episodes={metrics['total_episodes']:6d} "
                f"return={metrics['mean_episode_return']:.4f} "
                f"SOP={metrics['mean_outage_rate']:.4f} "
                f"sat_actor_loss={metrics['satellite_actor_loss']:.6f} "
                f"haps_actor_loss={metrics['haps_ris_actor_loss']:.6f} "
                f"critic_loss={metrics['critic_loss']:.6f} "
                f"entropy_coef={metrics['entropy_coef']:.6f} "
                f"sat_KL={metrics['satellite_approx_kl']:.6f} "
                f"haps_KL={metrics['haps_ris_approx_kl']:.6f} "
                f"KL_stop={int(metrics['satellite_early_stop'])}/"
                f"{int(metrics['haps_ris_early_stop'])} "
                f"stop_KL={metrics['satellite_stop_kl']:.6f}/"
                f"{metrics['haps_ris_stop_kl']:.6f}",
                flush=True,
            )
            plotter.on_update(metrics)

    except BaseException as error:
        training_error = error
        error_log_path = run_dir / "training_error.log"
        try:
            error_log_path.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            print(
                f"Training error traceback saved to: {error_log_path}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as log_error:
            print(
                f"Training failed and the traceback could not be saved: {log_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        episode_file.close()
        update_file.close()
        if training_error is None:
            plotter.finish()
        else:
            try:
                plotter.finish()
            except Exception as finish_error:
                print(
                    "Training failed and the final plot could not be saved: "
                    f"{finish_error}",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    main()
