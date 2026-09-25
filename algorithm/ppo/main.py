import csv
from datetime import datetime
from pathlib import Path
import sys


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
from system_model.params import get_system_params
from system_model.system_env import SAGIN_ENV


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
        self.returns = []
        self.return_mean = []
        self.sops = []
        self.sop_mean = []
        self.loss_episode = []
        self.actor_losses = []
        self.critic_losses = []

        if self.interactive:
            plt.ion()
        self.figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        self.reward_axis, self.sop_axis = axes[0]
        self.actor_axis, self.critic_axis = axes[1]

        self.reward_raw, = self.reward_axis.plot(
            [], [], alpha=0.3, color="tab:blue", label="raw"
        )
        self.reward_avg, = self.reward_axis.plot(
            [], [], color="tab:blue", label="moving average"
        )
        self.sop_raw, = self.sop_axis.plot(
            [], [], alpha=0.3, color="tab:red", label="raw"
        )
        self.sop_avg, = self.sop_axis.plot(
            [], [], color="tab:red", label="moving average"
        )
        self.actor_line, = self.actor_axis.plot(
            [], [], color="tab:green", marker="o", markersize=3
        )
        self.critic_line, = self.critic_axis.plot(
            [], [], color="tab:purple", marker="o", markersize=3
        )

        self.reward_axis.set_title("Episode Return")
        self.sop_axis.set_title("Episode SOP")
        self.actor_axis.set_title("Actor Loss per PPO Update")
        self.critic_axis.set_title("Critic Loss per PPO Update")

        for axis in axes.ravel():
            axis.set_xlabel("Completed episodes")
            axis.grid(alpha=0.25)

        self.reward_axis.set_ylabel("Return")
        self.sop_axis.set_ylabel("SOP")
        self.actor_axis.set_ylabel("Actor loss")
        self.critic_axis.set_ylabel("Critic loss")
        self.sop_axis.set_ylim(-0.02, 1.02)
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
        self.reward_axis.legend()
        self.sop_axis.legend()
        self.figure.tight_layout()
        if self.interactive:
            plt.show(block=False)
            self._draw()
        if self.output_path is not None:
            self.save()

    def on_episode(self, event):
        self.episodes.append(event["episode"])
        self.returns.append(event["return"])
        self.sops.append(event["outage_rate"])
        self.return_mean.append(
            float(np.mean(self.returns[-self.smooth_window:]))
        )
        self.sop_mean.append(
            float(np.mean(self.sops[-self.smooth_window:]))
        )

        self.reward_raw.set_data(self.episodes, self.returns)
        self.reward_avg.set_data(self.episodes, self.return_mean)
        self.sop_raw.set_data(self.episodes, self.sops)
        self.sop_avg.set_data(self.episodes, self.sop_mean)
        self._rescale(self.reward_axis)
        self._rescale(self.sop_axis, sop=True)

        if len(self.episodes) % self.draw_every == 0:
            self._draw()
        if (
            self.output_path is not None
            and len(self.episodes) % self.save_every == 0
        ):
            self.save()

    def on_update(self, metrics):
        self.loss_episode.append(metrics["total_episodes"])
        self.actor_losses.append(metrics["actor_loss"])
        self.critic_losses.append(metrics["critic_loss"])
        self.actor_waiting.set_visible(False)
        self.critic_waiting.set_visible(False)
        self.actor_line.set_data(self.loss_episode, self.actor_losses)
        self.critic_line.set_data(self.loss_episode, self.critic_losses)
        self._rescale(self.actor_axis)
        self._rescale(self.critic_axis)
        self._draw()
        if self.output_path is not None:
            self.save()

    @staticmethod
    def _rescale(axis, sop=False):
        axis.relim()
        axis.autoscale_view()
        if sop:
            axis.set_ylim(-0.02, 1.02)

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


def main():
    num_updates = 1000
    system_params = get_system_params()

    # Single-user example; replace these positions with the experiment setup.
    sat_positions = np.array([[500.0, 0.0, 6000.0]])
    uav_positions = np.array([[0.0, 0.0, system_params.uav_height]])
    user_positions = np.array([[100.0, 0.0, system_params.ground_height]])

    env = SAGIN_ENV(
        params=system_params,
        sat_positions=sat_positions,
        uav_positions=uav_positions,
        user_positions=user_positions,
    )
    ppo_config = PPOConfig(
        episodes_per_update=8,
        batch_size=400,
        ppo_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.1,
        entropy_coef=0.01,
        actor_lr=3e-5,
        critic_lr=3e-4,
        max_grad_norm=0.5,
    )
    controller = Controller(
        env=env,
        config=ppo_config,
        seed=system_params.seed,
    )
    run_dir = (
        Path(__file__).resolve().parent
        / "result"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    figure_path = run_dir / "training_curves.png"
    checkpoint_path = run_dir / "checkpoint.pt"
    episode_metrics_path = run_dir / "episode_metrics.csv"
    update_metrics_path = run_dir / "update_metrics.csv"
    print(f"Training results will be saved to: {run_dir}")
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
        "mean_bob_capacity", "mean_max_eve_capacity",
        "mean_bob_jammer_inr", "ris_repair_rate",
    ]
    update_fields = [
        "update", "total_episodes", "total_env_steps", "mean_episode_return",
        "mean_outage_rate", "actor_loss", "critic_loss", "entropy",
        "approx_kl", "clip_fraction",
    ]
    episode_file = episode_metrics_path.open("w", newline="", encoding="utf-8")
    update_file = update_metrics_path.open("w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
    update_writer = csv.DictWriter(update_file, fieldnames=update_fields)
    episode_writer.writeheader()
    update_writer.writeheader()

    def handle_episode(event):
        plotter.on_episode(event)
        episode_writer.writerow({name: event[name] for name in episode_fields})
        episode_file.flush()
        print(
            f"episode={event['episode']:6d} "
            f"return={event['return']:.4f} "
            f"mean_reward={event['mean_reward']:.4f} "
            f"SOP={event['outage_rate']:.4f} "
            f"gap={event['mean_raw_security_gap']:.6f}",
            flush=True,
        )

    try:
        for _ in range(num_updates):
            metrics = controller.update(
                episode_callback=handle_episode
            )
            plotter.on_update(metrics)

            temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "action_schema": "mrt_ris_group_residual_v1",
                    "update": controller.update_index,
                    "total_episodes": controller.total_episodes,
                    "total_env_steps": controller.total_env_steps,
                    "actor": controller.actor.state_dict(),
                    "critic": controller.critic.state_dict(),
                    "actor_optimizer": controller.actor_optimizer.state_dict(),
                    "critic_optimizer": controller.critic_optimizer.state_dict(),
                    "ppo_config": vars(ppo_config),
                    "system_params": vars(system_params),
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
                f"actor_loss={metrics['actor_loss']:.6f} "
                f"critic_loss={metrics['critic_loss']:.6f} "
                f"KL={metrics['approx_kl']:.6f}"
            )

    finally:
        episode_file.close()
        update_file.close()
        plotter.finish()


if __name__ == "__main__":
    main()
