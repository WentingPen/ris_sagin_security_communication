"""Reuse MAPPO's four performance panels and replace the two RL-loss panels."""

from pathlib import Path

from algorithm.ppo.main import LiveTrainingPlotter


class AOPlotter(LiveTrainingPlotter):
    def __init__(self, output_path):
        super().__init__(output_path=None, interactive=False)
        self.output_path = Path(output_path)
        self.actor_axis.set_title("AO Secrecy Margin per Batch")
        self.actor_axis.set_ylabel("Margin (bit/s/Hz)")
        self.satellite_actor_line.set_label("Before AO")
        self.haps_ris_actor_line.set_label("After AO")
        self.actor_axis.legend()
        self.critic_axis.set_title("Mean AO Solve Time per Slot")
        self.critic_axis.set_ylabel("Seconds")
        self.actor_waiting.set_text("Waiting for the first AO batch")
        self.critic_waiting.set_text("Waiting for the first AO batch")
        self.initial_objectives, self.final_objectives, self.solve_times = [], [], []
        self.figure.tight_layout()
        self.save()

    def on_update(self, metrics):
        self.loss_episode.append(metrics["total_episodes"])
        self.initial_objectives.append(metrics["mean_initial_objective"])
        self.final_objectives.append(metrics["mean_final_objective"])
        self.solve_times.append(metrics["mean_solve_seconds"])
        self.actor_waiting.set_visible(False)
        self.critic_waiting.set_visible(False)
        self.satellite_actor_line.set_data(self.loss_episode, self.initial_objectives)
        self.haps_ris_actor_line.set_data(self.loss_episode, self.final_objectives)
        self.critic_line.set_data(self.loss_episode, self.solve_times)
        self._rescale(self.actor_axis)
        self._rescale(self.critic_axis)
        self.save()
