import builtins
import inspect
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import config
from event_stream import EventStream
from runtime_state import FrameSnapshot, LatestFrameBuffer
from runtime_tuning import resolve_runtime_tuning

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ParserImportTests(unittest.TestCase):
    def test_parse_args_does_not_import_algorithms(self):
        from parser import parse_args

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "algorithms" or name.startswith("algorithms."):
                raise AssertionError("parse_args imported algorithms before runtime tuning")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(sys, "argv", ["prog"]), mock.patch(
            "builtins.__import__",
            side_effect=guarded_import,
        ):
            args = parse_args()

        self.assertFalse(args.dl)


class RuntimeTuningTests(unittest.TestCase):
    def test_dashboard_mode_caps_training_workers(self):
        args = SimpleNamespace(
            headless=False,
            train_workers=None,
            torch_threads=None,
            torch_interop_threads=None,
        )
        with mock.patch.object(config, "N_TRAIN_WORKERS", 8), mock.patch(
            "os.cpu_count",
            return_value=10,
        ):
            tuning = resolve_runtime_tuning(args)

        self.assertEqual(tuning.train_workers, 4)
        self.assertEqual(tuning.torch_threads, config.TORCH_NUM_THREADS)
        self.assertEqual(tuning.torch_interop_threads, config.TORCH_NUM_INTEROP_THREADS)

    def test_headless_mode_keeps_configured_training_workers(self):
        args = SimpleNamespace(
            headless=True,
            train_workers=None,
            torch_threads=None,
            torch_interop_threads=None,
        )
        with mock.patch.object(config, "N_TRAIN_WORKERS", 8):
            tuning = resolve_runtime_tuning(args)

        self.assertEqual(tuning.train_workers, 8)

    def test_cli_worker_override_wins(self):
        args = SimpleNamespace(
            headless=False,
            train_workers=2,
            torch_threads=3,
            torch_interop_threads=1,
        )
        tuning = resolve_runtime_tuning(args)

        self.assertEqual(tuning.train_workers, 2)
        self.assertEqual(tuning.torch_threads, 3)
        self.assertEqual(tuning.torch_interop_threads, 1)


class LatestFrameBufferTests(unittest.TestCase):
    def test_latest_frame_overwrites_stale_frame(self):
        buffer = LatestFrameBuffer()

        v1 = buffer.publish(FrameSnapshot(sim_time=1.0, step_count=1))
        v2 = buffer.publish(FrameSnapshot(sim_time=2.0, step_count=2))
        latest_version, latest_frame = buffer.latest()

        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(latest_version, 2)
        self.assertEqual(latest_frame.sim_time, 2.0)
        self.assertEqual(latest_frame.step_count, 2)


class EventStreamTests(unittest.TestCase):
    def test_depth_tracks_bounded_backlog(self):
        stream = EventStream(max_events=2)

        stream.publish(1.0, "status", "a")
        stream.publish(2.0, "status", "b")
        stream.publish(3.0, "status", "c")

        self.assertEqual(stream.depth(), 2)
        drained = stream.drain(max_items=10)
        self.assertEqual([event.text for event in drained], ["b", "c"])
        self.assertEqual(stream.depth(), 0)


class DLEnvironmentSnapshotTests(unittest.TestCase):
    def test_step_does_not_build_progress_snapshot(self):
        try:
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        source = inspect.getsource(DLEnvironment.step)

        self.assertNotIn("get_progress_snapshot()", source)
        self.assertNotIn('"training_status"', source)


class MapZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        cls._app = QApplication.instance() or QApplication(["test"])

    def test_scale_view_preserves_anchor_viewport_position(self):
        try:
            from PySide6.QtCore import QPoint
            from dashboard import theme
            from dashboard.map_view import MapWidget
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        theme.init("dark")
        widget = MapWidget(
            net_bounds=(0.0, 0.0, 1000.0, 1000.0),
            edge_shapes=[[(0.0, 0.0), (1000.0, 1000.0)]],
            scenario_name="test",
        )
        widget.resize(500, 400)
        widget.show()
        self._app.processEvents()
        widget._fit_initial()

        anchor = QPoint(173, 121)
        widget._scale_view(1.12, widget.viewport().rect().center())
        widget._scale_view(1.12, widget.viewport().rect().center())
        scene_before = widget.mapToScene(anchor)
        widget._scale_view(1.12, anchor)
        viewport_after = widget.mapFromScene(scene_before)

        self.assertLessEqual(abs(viewport_after.x() - anchor.x()), 1)
        self.assertLessEqual(abs(viewport_after.y() - anchor.y()), 1)
        widget.close()


if __name__ == "__main__":
    unittest.main()
