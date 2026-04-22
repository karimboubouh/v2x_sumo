import os
import sys
import tempfile
import unittest
from unittest import mock

import logger


class LoggingArtifactsTests(unittest.TestCase):
    def tearDown(self):
        logger.stop_file_logging()
        logger.set_level("info")

    def test_parse_args_accepts_save_logs_flag(self):
        try:
            from parser import parse_args
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")
        with mock.patch.object(sys, "argv", ["prog", "--save-logs"]):
            args = parse_args()
        self.assertTrue(args.save_logs)

    def test_parse_args_defaults_save_logs_to_false(self):
        try:
            from parser import parse_args
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")
        with mock.patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        self.assertFalse(args.save_logs)

    def test_logger_file_sink_writes_plain_filtered_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "run.log")
            logger.set_level("info")
            logger.start_file_logging(log_path)
            logger.log("hello world", "info")
            logger.log("details continue")
            logger.log("hidden debug", "debug")
            logger.stop_file_logging()

            with open(log_path, "r", encoding="utf-8") as fh:
                lines = [line.rstrip("\n") for line in fh]

        self.assertEqual(lines[0], "[INFO] hello world")
        self.assertEqual(lines[1], f"{' ' * 10}details continue")
        self.assertEqual(len(lines), 2)
        self.assertNotIn("\033", "\n".join(lines))

    def test_prepare_and_save_experiment_reuse_same_folder_and_log_path(self):
        try:
            from dl.experiment import prepare_experiment_dir, save_experiment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                "scenario": "demo",
                "algorithm": "DANTE",
                "dataset": "MNIST",
                "model": "DNN",
                "num_vehicles": 4,
            }
            prepared = prepare_experiment_dir(metadata, out_root=tmpdir)
            with open(prepared["log_path"], "w", encoding="utf-8") as fh:
                fh.write("[INFO] started\n")

            experiment = {
                "metadata": {
                    **metadata,
                    "experiment_id": prepared["experiment_id"],
                },
                "config": {},
                "train_history": [],
                "eval_history": [],
                "test_history": [],
                "reward_history": [],
                "summary": {},
            }
            saved = save_experiment(experiment, out_root=tmpdir)

            self.assertEqual(saved["experiment_dir"], prepared["experiment_dir"])
            self.assertEqual(saved["log_path"], prepared["log_path"])
            self.assertTrue(os.path.isfile(os.path.join(prepared["experiment_dir"], "experiment.pkl")))
            self.assertTrue(os.path.isfile(prepared["log_path"]))

    def test_save_experiment_rejects_invalid_local_only_links(self):
        try:
            from dl.experiment import save_experiment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = {
                "metadata": {"algorithm": "Local Only"},
                "config": {
                    "ALGORITHM": "LocalOnly",
                    "MAX_SIDELINK_NEIGHBORS": 0,
                    "MAX_INTERNET_NEIGHBORS": 0,
                    "MAX_COLLAB_NEIGHBORS": 0,
                },
                "train_history": [{"round": 1, "total_links": 2}],
                "eval_history": [],
                "test_history": [],
                "reward_history": [],
                "summary": {},
            }
            with self.assertRaises(ValueError):
                save_experiment(experiment, out_root=tmpdir)


if __name__ == "__main__":
    unittest.main()
