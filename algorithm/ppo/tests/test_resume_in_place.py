import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from algorithm.ppo.main import (
    LiveTrainingPlotter,
    _discard_uncommitted_metric_rows,
    _load_committed_metric_rows,
    _parse_training_args,
)


class ResumeInPlaceTests(unittest.TestCase):
    def test_resume_in_place_requires_checkpoint(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            _parse_training_args(["--resume-in-place"])

    def test_uncommitted_metric_rows_are_discarded(self):
        fields = ["episode", "value"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as metric_file:
                writer = csv.DictWriter(metric_file, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {"episode": 1, "value": "committed"},
                    {"episode": 2, "value": "committed"},
                    {"episode": 3, "value": "partial"},
                ])

            rows, discarded_count = _load_committed_metric_rows(
                path, fields, "episode", committed_index=2
            )

            self.assertEqual([row["episode"] for row in rows], ["1", "2"])
            self.assertEqual(discarded_count, 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 4)

            _discard_uncommitted_metric_rows(
                path, fields, rows, discarded_count
            )

            with path.open("r", newline="", encoding="utf-8") as metric_file:
                saved_rows = list(csv.DictReader(metric_file))
            self.assertEqual(saved_rows, rows)

    def test_validation_failure_does_not_modify_another_metric_file(self):
        fields = ["episode", "value"]
        with tempfile.TemporaryDirectory() as directory:
            episode_path = Path(directory) / "episode_metrics.csv"
            update_path = Path(directory) / "update_metrics.csv"
            episode_path.write_text(
                "episode,value\n1,committed\n2,partial\n", encoding="utf-8"
            )
            update_path.write_text("wrong,header\n1,value\n", encoding="utf-8")
            original_episode_bytes = episode_path.read_bytes()

            _load_committed_metric_rows(
                episode_path, fields, "episode", committed_index=1
            )
            with self.assertRaisesRegex(ValueError, "header does not match"):
                _load_committed_metric_rows(
                    update_path, ["update", "value"], "update", committed_index=1
                )

            self.assertEqual(episode_path.read_bytes(), original_episode_bytes)

    def test_plot_history_is_restored(self):
        plotter = LiveTrainingPlotter(output_path=None, interactive=False)
        try:
            plotter.load_history(
                [
                    {
                        "episode": "1",
                        "outage_rate": "1.0",
                        "return": "-2.0",
                        "max_bob_capacity": "0.1",
                        "max_eve_capacity": "0.2",
                    },
                    {
                        "episode": "2",
                        "outage_rate": "0.5",
                        "return": "-1.0",
                        "max_bob_capacity": "0.3",
                        "max_eve_capacity": "0.1",
                    },
                ],
                [{
                    "total_episodes": "2",
                    "satellite_actor_loss": "0.01",
                    "haps_ris_actor_loss": "0.02",
                    "critic_loss": "0.03",
                }],
            )
            self.assertEqual(plotter.episodes, [1, 2])
            self.assertEqual(plotter.sop_mean, [1.0, 0.75])
            self.assertEqual(plotter.loss_episode, [2])
        finally:
            plotter.finish()


if __name__ == "__main__":
    unittest.main()
