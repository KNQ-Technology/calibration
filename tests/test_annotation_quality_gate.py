"""Headless checks for the warning/confirmation path before measurement.

Importing annotation_client prepares the GUI environment (and may install
dependencies), so execute only its production save-and-run method from the
source AST.  The worker is recorded but never started in these tests.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"


def _load_save_and_run_measurement(namespace):
    module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient"
    )
    method = next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_and_run_measurement"
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method.name]


class _Status:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


class _LogText:
    def __init__(self):
        self.entries = []

    def insert(self, _position, text):
        self.entries.append(text)

    def configure(self, **_kwargs):
        pass


class _App:
    def __init__(self, *, boundary_mode=False, keypoints_complete=True, missing_step=None):
        self.events = []
        self.boundary_mode = boundary_mode
        self.boundary_enabled_var = SimpleNamespace(get=lambda: self.boundary_mode)
        self.keypoints_complete = keypoints_complete
        self.missing_step = missing_step
        self.boundary_edges = {"top": [], "right": [], "bottom": [], "left": []}
        self.boundary_source_name = "frame.mp4"
        self.keypoint_source_name = "frame.mp4"
        self.marker_groups = [[(100, 100)]]
        self.pitch_config = object()
        self.image_w = 1280
        self.image_h = 720
        self.keypoints = ([SimpleNamespace(vertex_index=0, img_coord=(100, 100))]
                          if keypoints_complete else [])
        self.source_path = Path("frame.mp4")
        self.media_kind = "video"
        self.video_path_var = SimpleNamespace(get=lambda: str(self.source_path))
        self._measurement_running = False
        self.camera_name = "camera1"
        self.status = _Status()
        self.log_text = _LogText()

    def _step_save_path(self, step):
        if step == "field_boundary":
            return SimpleNamespace(exists=lambda: self.boundary_mode)
        return Path(f"/unused/{step}.json")

    def _build_boundary_data(self):
        return {"enabled": True, "source": self.boundary_source_name}

    def _read_video_frame(self, _path, _index):
        return SimpleNamespace(shape=(self.image_h, self.image_w, 3))

    def _step_complete(self, step):
        self.events.append(("step_complete", step))
        if step == "keypoints":
            return self.keypoints_complete
        return step != self.missing_step

    def _select_step_by_key(self, key):
        self.events.append(("select_step", key))

    def save_all_default(self):
        self.events.append(("save",))
        return True

    def _open_measurement_log_window(self):
        self.events.append(("open_log",))
        return object(), self.log_text

    def _run_measurement_worker(self, *_args):
        raise AssertionError("A headless flow test must not run the measurement worker")


def _exercise(*, issues=(), confirm=True, boundary_mode=False,
              keypoints_complete=True, missing_step=None):
    app = _App(boundary_mode=boundary_mode, keypoints_complete=keypoints_complete,
               missing_step=missing_step)
    prompts = []
    inspection_calls = []

    def inspect_quality(**kwargs):
        inspection_calls.append(kwargs)
        return [SimpleNamespace(message=issue) for issue in issues]

    def askokcancel(title, message):
        prompts.append((title, message))
        return confirm

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            app.events.append(("thread_created", daemon))
            self.target = target
            self.args = args

        def start(self):
            app.events.append(("thread_started",))

    namespace = {
        "Path": Path,
        "messagebox": SimpleNamespace(
            askokcancel=askokcancel,
            showwarning=lambda title, message: app.events.append(("warning", title, message)),
        ),
        "inspect_calibration_quality": inspect_quality,
        "tk": SimpleNamespace(END="end"),
        "threading": SimpleNamespace(Thread=FakeThread),
    }
    method = _load_save_and_run_measurement(namespace)
    method(app)
    return app, prompts, inspection_calls


class AnnotationQualityGateTests(unittest.TestCase):
    def test_quality_warning_cancel_does_not_save_or_start(self):
        app, prompts, calls = _exercise(issues=("关键点对应关系异常",), confirm=False)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(prompts), 1)
        self.assertIn("关键点对应关系异常", prompts[0][1])
        self.assertNotIn(("save",), app.events)
        self.assertNotIn(("thread_started",), app.events)
        self.assertNotIn(("open_log",), app.events)

    def test_quality_warning_confirm_continues_and_starts_worker(self):
        app, prompts, _calls = _exercise(issues=("关键点对应关系异常",), confirm=True)

        self.assertEqual(len(prompts), 1)
        self.assertIn(("save",), app.events)
        self.assertIn(("thread_started",), app.events)
        self.assertLess(app.events.index(("save",)), app.events.index(("thread_started",)))
        self.assertTrue(any("关键点对应关系异常" in text for text in app.log_text.entries))

    def test_healthy_annotations_continue_without_confirmation(self):
        app, prompts, _calls = _exercise()

        self.assertEqual(prompts, [])
        self.assertIn(("save",), app.events)
        self.assertIn(("thread_started",), app.events)

    def test_valid_four_edge_mode_does_not_require_keypoints(self):
        app, prompts, calls = _exercise(boundary_mode=True, keypoints_complete=False)

        self.assertEqual(prompts, [])
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0]["field_boundary"])
        self.assertEqual(calls[0]["keypoints"], [])
        self.assertNotIn(("step_complete", "keypoints"), app.events)
        self.assertIn(("save",), app.events)
        self.assertIn(("thread_started",), app.events)

    def test_legacy_mode_requires_both_line_annotation_steps(self):
        for key, title in (("manual_h", "横向直线标注"),
                           ("manual_v", "纵向直线标注")):
            with self.subTest(step=key):
                app, prompts, inspection_calls = _exercise(missing_step=key)
                self.assertEqual(prompts, [])
                self.assertEqual(inspection_calls, [])
                self.assertIn(("select_step", key), app.events)
                self.assertTrue(any(event[0] == "warning" and title in event[2]
                                    for event in app.events))
                self.assertNotIn(("save",), app.events)
                self.assertNotIn(("thread_started",), app.events)


if __name__ == "__main__":
    unittest.main()
