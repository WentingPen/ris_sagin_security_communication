import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from algorithm.ppo.main import _parse_training_args, main


class TrainingArgsTests(unittest.TestCase):
    def test_episode_batch_option_preserves_system_arguments(self):
        for option in ("--episodes-per-update", "--episodes_per_update"):
            with self.subTest(option=option):
                args, system_argv = _parse_training_args([
                    option, "16", "--total-updates", "500",
                    "--jammer_power", "0.1",
                ])
                self.assertEqual(args.episodes_per_update, 16)
                self.assertEqual(args.total_updates, 500)
                self.assertEqual(system_argv, ["--jammer_power", "0.1"])

    def test_episode_batch_must_be_a_positive_integer(self):
        for value in ("0", "-1", "1.5"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                _parse_training_args(["--episodes-per-update", value])
            self.assertEqual(error.exception.code, 2)

    def test_resume_rejects_episode_batch_override(self):
        with (
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as error,
        ):
            _parse_training_args([
                "--resume", "checkpoint.pt", "--episodes-per-update", "16",
            ])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("checkpoint PPO configuration is reused", stderr.getvalue())

    def test_resume_without_override_keeps_checkpoint_configuration(self):
        args, system_argv = _parse_training_args(["--resume", "checkpoint.pt"])
        self.assertEqual(args.resume, Path("checkpoint.pt"))
        self.assertIsNone(args.episodes_per_update)
        self.assertEqual(system_argv, [])

    def test_new_training_receives_episode_batch_and_matching_anneal_length(self):
        for argv, episodes, updates in (
            ([], 8, 1000),
            (["--episodes-per-update", "16", "--total-updates", "500"], 16, 500),
        ):
            with (
                self.subTest(argv=argv),
                patch("algorithm.ppo.main._build_environment", return_value=object()),
                patch(
                    "algorithm.ppo.main.Controller",
                    side_effect=RuntimeError("stop before training"),
                ) as controller,
                self.assertRaisesRegex(RuntimeError, "stop before training"),
            ):
                main(argv)
            config = controller.call_args.kwargs["config"]
            self.assertEqual(config.episodes_per_update, episodes)
            self.assertEqual(config.entropy_anneal_updates, updates)
            self.assertEqual(config.batch_size, 256)
            self.assertEqual(config.ppo_epochs, 2)


if __name__ == "__main__":
    unittest.main()
