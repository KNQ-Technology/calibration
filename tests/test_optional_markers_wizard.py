"""Headless regressions for trainings without equipment annotations."""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, Tuple
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _ReachedSave(Exception):
    pass


def _load_method(name, *, with_writer=False):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    functions = [method]
    if with_writer:
        functions.insert(0, next(node for node in tree.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name == "_save_annotation_file"))
    namespace = {
        "Path": Path,
        "Any": Any, "Dict": Dict, "Tuple": Tuple, "json": json,
        "inspect_calibration_quality": lambda **_kwargs: [],
        "messagebox": SimpleNamespace(showwarning=lambda *_args: None),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace if with_writer else namespace[name]


class OptionalMarkersWizardTests(unittest.TestCase):
    def test_marker_step_is_explicitly_optional(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        step_list = next(node.value for node in tree.body
                         if isinstance(node, ast.AnnAssign)
                         and isinstance(node.target, ast.Name)
                         and node.target.id == "STEP_DEFS")
        marker_step = next(node for node in step_list.elts
                           if isinstance(node, ast.Call)
                           and node.args and isinstance(node.args[0], ast.Constant)
                           and node.args[0].value == "markers")
        self.assertTrue(any(keyword.arg == "optional"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                            for keyword in marker_step.keywords))

    def test_empty_marker_placeholder_does_not_count_as_completed(self):
        is_complete = _load_method("_step_complete")
        app = SimpleNamespace(marker_groups=[[]])
        self.assertFalse(is_complete(app, "markers"))
        app.marker_groups = [[], [(3, 4)]]
        self.assertTrue(is_complete(app, "markers"))

    def test_empty_marker_groups_only_update_status(self):
        warn = _load_method("_warn_unsaved_marker_groups")
        app = SimpleNamespace(marker_groups=[[]], status=_Variable("已保存"))
        warn(app)
        self.assertIn("无器材训练", app.status.get())
        app.marker_groups = [[(1, 2)], []]
        app.status.set("已保存")
        warn(app)
        self.assertIn("空白器材组 2 已忽略", app.status.get())

    def test_measurement_preflight_allows_empty_and_partial_marker_groups(self):
        start_measurement = _load_method("save_and_run_measurement")
        for groups in ([[]], [[(1, 2)], []]):
            with self.subTest(groups=groups):
                app = SimpleNamespace(
                    _measurement_running=False,
                    video_path_var=_Variable("/example/video.mp4"),
                    _read_video_frame=lambda *_args: SimpleNamespace(shape=(720, 1280, 3)),
                    media_kind="video",
                    boundary_enabled_var=_Variable(True),
                    _build_boundary_data=lambda: {"enabled": True},
                    pitch_config=object(), image_h=720, image_w=1280,
                    keypoints=[], source_path=None, keypoint_source_name="",
                    boundary_source_name="", marker_groups=groups,
                    _confirm_measurement_retry=lambda: True,
                    save_all_default=lambda: (_ for _ in ()).throw(_ReachedSave()),
                )
                with self.assertRaises(_ReachedSave):
                    start_measurement(app)

    def test_single_step_save_invalidates_old_report_only_when_markers_change(self):
        namespace = _load_method("save_current_step", with_writer=True)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            markers = output / "markers.json"
            results = tuple(output / name for name in (
                "coordinates.json", "measurement_report.txt", "visualization.jpg"))
            for path in results:
                path.write_text("previous report", encoding="utf-8")
            app = SimpleNamespace(
                _require_applied_pitch_config=lambda: True,
                current_step=SimpleNamespace(key="markers"),
                _step_save_path=lambda _key: markers,
                _build_step_data=lambda _key: {"source": "markers", "groups": []},
                _marker_result_paths=lambda: results,
                _write_config_yaml=lambda: None,
                _refresh_step_list=lambda: None,
                _warn_unsaved_marker_groups=lambda: None,
                _refresh_finish_summary=lambda: None,
                status=_Variable(""),
            )
            namespace["save_current_step"](app)
            self.assertEqual(json.loads(markers.read_text(encoding="utf-8"))["groups"], [])
            self.assertTrue(all(not path.exists() for path in results))
            for path in results:
                path.write_text("fresh report", encoding="utf-8")
            namespace["save_current_step"](app)
            self.assertTrue(all(path.exists() for path in results))

    def test_save_all_invalidates_old_report_only_when_markers_change(self):
        namespace = _load_method("save_all_default", with_writer=True)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            markers = output / "markers.json"
            results = tuple(output / name for name in (
                "coordinates.json", "measurement_report.txt", "visualization.jpg"))
            for path in results:
                path.write_text("previous report", encoding="utf-8")

            def build_data(key):
                if key == "markers":
                    return {"source": "markers", "groups": []}
                raise ValueError("optional step skipped")

            app = SimpleNamespace(
                _require_applied_pitch_config=lambda: True,
                _step_save_path=lambda key: output / ("markers.json" if key == "markers" else f"{key}.json"),
                boundary_enabled_var=_Variable(False),
                _write_config_yaml=lambda **_kwargs: None,
                _build_step_data=build_data,
                _marker_result_paths=lambda: results,
                _refresh_step_list=lambda: None,
                _warn_unsaved_marker_groups=lambda: None,
                _refresh_finish_summary=lambda: None,
                status=_Variable(""),
            )
            self.assertTrue(namespace["save_all_default"](app))
            self.assertEqual(json.loads(markers.read_text(encoding="utf-8"))["groups"], [])
            self.assertTrue(all(not path.exists() for path in results))
            for path in results:
                path.write_text("fresh report", encoding="utf-8")
            self.assertTrue(namespace["save_all_default"](app))
            self.assertTrue(all(path.exists() for path in results))


if __name__ == "__main__":
    unittest.main()
