import builtins
import inspect
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import config
from event_stream import EventStream
from runtime_state import (
    DPLSnapshot,
    FrameSnapshot,
    LatestFrameBuffer,
    LatestMobilityBuffer,
    MobilitySnapshot,
)
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

    def test_latest_mobility_buffer_wait_returns_newest_snapshot_only(self):
        buffer = LatestMobilityBuffer()

        buffer.publish(MobilitySnapshot(sim_time=1.0, step_count=1))
        version = buffer.publish(MobilitySnapshot(sim_time=2.0, step_count=2))
        latest_version, latest_snapshot = buffer.wait_for_newer(0, timeout=0.01)

        self.assertEqual(latest_version, version)
        self.assertEqual(latest_snapshot.step_count, 2)

        same_version, same_snapshot = buffer.wait_for_newer(version, timeout=0.01)
        self.assertEqual(same_version, version)
        self.assertIs(same_snapshot, latest_snapshot)


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

    def test_initial_training_round_is_deferred_when_requested(self):
        try:
            from dl.env import DLEnvironment
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise unittest.SkipTest(f"optional dependency missing: {exc}")

        init_source = inspect.getsource(DLEnvironment.__init__)
        initial_round_source = inspect.getsource(DLEnvironment.run_initial_training_round)

        self.assertIn("run_initial_round", inspect.signature(DLEnvironment).parameters)
        self.assertIn("if run_initial_round:", init_source)
        self.assertNotIn("futures_wait(futs)", init_source)
        self.assertIn("futures_wait(futs)", initial_round_source)


class DecoupledRuntimeTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "headless": True,
            "dl": True,
            "dl_algorithm": "DPFL",
            "dl_dataset": "CIFAR10",
            "dl_model": "CNN",
            "num_vehicles": 1,
            "rounds": 20,
            "target_acc": 2.0,
            "speed": 1.0,
            "ui_fps": 60,
            "comm_range": 350.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_mobility_step_does_not_call_dpl_step(self):
        from simulation_runtime import SimulationRuntime

        class FakeSumo:
            def __init__(self):
                self.sim_time = 0.0

            def step(self, headless=False):
                del headless
                self.sim_time += 1.0
                return {
                    "mv_0": SimpleNamespace(
                        vehicle_id="mv_0",
                        x=self.sim_time,
                        y=0.0,
                        speed=1.0,
                        angle=90.0,
                        edge_id="",
                    )
                }

            def get_sim_time(self):
                return self.sim_time

        runtime = SimulationRuntime(self._args(), {}, EventStream(), lambda sim_time: {})
        runtime.sumo = FakeSumo()
        runtime.comm = SimpleNamespace(
            update=lambda vehicle_states, sim_time: None,
            get_active_links=lambda: [],
        )
        runtime.dl_env = SimpleNamespace(
            step=mock.Mock(side_effect=AssertionError("mobility must not call DPL"))
        )

        runtime._advance_mobility_step()
        runtime._advance_mobility_step()

        self.assertEqual(runtime.step_count, 2)
        self.assertEqual(runtime.sim_time, 2.0)
        runtime.dl_env.step.assert_not_called()

    def test_frame_composition_uses_fresh_mobility_with_stale_dpl(self):
        from simulation_runtime import SimulationRuntime

        runtime = SimulationRuntime(self._args(headless=False), {}, EventStream(), lambda sim_time: {})
        now = time.perf_counter()
        mobility = MobilitySnapshot(
            vehicle_states={"mv_0": SimpleNamespace(x=5.0, y=0.0, speed=1.0, angle=90.0)},
            comm_links=["comm"],
            sim_time=5.0,
            step_count=5,
            sim_hz=1.0,
            wall_time=now,
        )
        dpl_snapshot = DPLSnapshot(
            training_status={"enabled": True, "active_trainers": 1},
            render_links=["dpl"],
            vehicle_overlays={"mv_0": {"training_active": True}},
            lifecycle="running",
            processed_step_count=2,
            latest_step_count=2,
            wall_time=now - 2.0,
        )

        frame = runtime._compose_frame(mobility, dpl_snapshot, now=now)

        self.assertEqual(frame.step_count, 5)
        self.assertEqual(frame.render_links, ["comm", "dpl"])
        self.assertEqual(frame.log_links, ["comm"])
        self.assertEqual(frame.training_status["dpl_lag_steps"], 3)
        self.assertGreaterEqual(frame.training_status["dpl_status_age_s"], 2.0)

    def test_empty_dpl_links_do_not_hide_physical_comm_links(self):
        from simulation_runtime import SimulationRuntime

        runtime = SimulationRuntime(self._args(headless=False), {}, EventStream(), lambda sim_time: {})
        now = time.perf_counter()
        mobility = MobilitySnapshot(
            vehicle_states={"mv_0": SimpleNamespace(x=5.0, y=0.0, speed=1.0, angle=90.0)},
            comm_links=["comm"],
            sim_time=5.0,
            step_count=5,
            sim_hz=1.0,
            wall_time=now,
        )
        dpl_snapshot = DPLSnapshot(
            training_status={"enabled": True},
            render_links=[],
            lifecycle="initial_training",
            processed_step_count=0,
            latest_step_count=5,
            wall_time=now,
        )

        frame = runtime._compose_frame(mobility, dpl_snapshot, now=now)

        self.assertEqual(frame.render_links, ["comm"])
        self.assertEqual(frame.log_links, ["comm"])
        self.assertEqual(frame.training_status["dpl_lifecycle"], "initial_training")


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

    def test_visual_extrapolation_is_bounded_and_does_not_mutate_source_state(self):
        try:
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
        state = SimpleNamespace(vehicle_id="mv_0", x=0.0, y=0.0, speed=10.0, angle=90.0, edge_id="")
        states = {"mv_0": state}

        rendered = widget._extrapolated_vehicle_states(states, source_wall_time=10.0, now=15.0)

        self.assertIsNot(rendered["mv_0"], state)
        self.assertEqual(state.x, 0.0)
        self.assertEqual(state.y, 0.0)
        self.assertAlmostEqual(rendered["mv_0"].x, 10.0, places=5)
        self.assertAlmostEqual(rendered["mv_0"].y, 0.0, places=5)
        widget.close()


if __name__ == "__main__":
    unittest.main()
