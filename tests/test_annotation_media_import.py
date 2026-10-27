"""Headless regressions for switching the annotation wizard to new media.

The GUI module performs environment setup at import time, so only its selected
methods are executed from the AST.  In particular, these tests never create a
Tk window or touch the user's calibration files.
"""

import ast
from datetime import datetime
import os
from pathlib import Path
import re
from types import MethodType, SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"


def _load_methods(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    methods = [node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names), f"missing media methods: {set(names) - {m.name for m in methods}}"
    namespace = {
        "Path": Path, "datetime": datetime, "os": os, "re": re, "uuid4": uuid4,
        "BOUNDARY_KEYS": ("top", "right", "bottom", "left"),
        "BOUNDARY_UI_ORDER": ("left", "top", "right", "bottom"),
        "DEFAULT_EQUIPMENT_TYPE": "default",
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _app(namespace, *method_names, **attributes):
    instance = SimpleNamespace(**attributes)
    for name in method_names:
        setattr(instance, name, MethodType(namespace[name], instance))
    return instance


class AnnotationMediaImportTests(unittest.TestCase):
    def test_restart_reopens_saved_image_instead_of_previous_video(self):
        ns = _load_methods("_load_default_frame")
        expected = object()
        image = Path("/new/task-frame.jpg")
        app = _app(ns, "_load_default_frame",
                   config_data={"annotation_media_path": str(image)},
                   media_kind="image", frame_index=_Var(0),
                   _default_video_path=lambda: Path("/old/video.mp4"),
                   _read_image=lambda path: expected if path == image else None,
                   _read_video_frame=lambda *_args: self.fail("old video was reopened"))
        self.assertIs(app._load_default_frame(), expected)
        self.assertEqual(app.source_path, image)

    def test_video_dialog_cancel_does_not_replace_current_source(self):
        ns = _load_methods("open_video")
        ns["filedialog"] = SimpleNamespace(askopenfilename=lambda **_kw: "")
        old_source = Path("/previous/video.mp4")
        app = _app(ns, "open_video", project_dir=Path("/project"),
                   source_path=old_source, video_path_var=_Var(str(old_source)),
                   frame_index=_Var(45),
                   _activate_new_media=lambda *_args: self.fail("cancel activated new media"))
        app.open_video()
        self.assertEqual(app.source_path, old_source)
        self.assertEqual(app.video_path_var.get(), str(old_source))
        self.assertEqual(app.frame_index.get(), 45)

    def test_video_decode_failure_does_not_replace_current_source(self):
        ns = _load_methods("open_video")
        ns["filedialog"] = SimpleNamespace(
            askopenfilename=lambda **_kw: "/new/broken.mp4")
        errors = []
        ns["messagebox"] = SimpleNamespace(showerror=lambda *args: errors.append(args))
        old_source = Path("/previous/video.mp4")
        app = _app(ns, "open_video", project_dir=Path("/project"),
                   source_path=old_source, video_path_var=_Var(str(old_source)),
                   frame_index=_Var(45),
                   _read_video_frame=lambda *_args: None,
                   _activate_new_media=lambda *_args: self.fail("bad video activated"))
        app.open_video()
        self.assertEqual(app.source_path, old_source)
        self.assertEqual(app.video_path_var.get(), str(old_source))
        self.assertEqual(app.frame_index.get(), 45)
        self.assertTrue(errors)

    def test_new_video_uses_first_frame_then_activates_new_session(self):
        ns = _load_methods("open_video")
        ns["filedialog"] = SimpleNamespace(
            askopenfilename=lambda **_kw: "/new/training.mp4")
        first_frame = object()
        decoded = []
        activated = []
        app = _app(ns, "open_video", project_dir=Path("/project"),
                   source_path=Path("/previous/video.mp4"), frame_index=_Var(45),
                   _read_video_frame=lambda path, index:
                   (decoded.append((Path(path), index)) or first_frame),
                   _activate_new_media=lambda path, frame, kind:
                   (activated.append((Path(path), frame, kind)) or True))
        app.open_video()
        self.assertEqual(decoded, [(Path("/new/training.mp4"), 0)])
        self.assertEqual(activated, [(Path("/new/training.mp4"), first_frame, "video")])

    def test_image_decode_failure_does_not_replace_current_source(self):
        ns = _load_methods("open_image")
        ns["filedialog"] = SimpleNamespace(
            askopenfilename=lambda **_kw: "/new/broken.jpg")
        errors = []
        ns["messagebox"] = SimpleNamespace(showerror=lambda *args: errors.append(args))
        old_source = Path("/previous/video.mp4")
        app = _app(ns, "open_image", project_dir=Path("/project"),
                   source_path=old_source, video_path_var=_Var(str(old_source)),
                   frame_index=_Var(45),
                   _read_image=lambda *_args: None,
                   _activate_new_media=lambda *_args: self.fail("bad image activated"))
        app.open_image()
        self.assertEqual(app.source_path, old_source)
        self.assertEqual(app.video_path_var.get(), str(old_source))
        self.assertEqual(app.frame_index.get(), 45)
        self.assertTrue(errors)

    def test_new_image_activates_new_session(self):
        ns = _load_methods("open_image")
        ns["filedialog"] = SimpleNamespace(
            askopenfilename=lambda **_kw: "/new/frame.jpg")
        frame = object()
        activated = []
        app = _app(ns, "open_image", project_dir=Path("/project"),
                   source_path=Path("/previous/video.mp4"),
                   _read_image=lambda *_args: frame,
                   _activate_new_media=lambda path, decoded, kind:
                   (activated.append((Path(path), decoded, kind)) or True))
        app.open_image()
        self.assertEqual(activated, [(Path("/new/frame.jpg"), frame, "image")])

    def test_image_without_associated_video_does_not_start_measurement(self):
        ns = _load_methods("save_and_run_measurement")
        warnings = []
        ns["messagebox"] = SimpleNamespace(showwarning=lambda *args: warnings.append(args))
        app = _app(ns, "save_and_run_measurement",
                   media_kind="image", source_path=Path("/new/frame.jpg"),
                   _measurement_running=False, video_path_var=_Var(""),
                   _confirm_measurement_retry=lambda: True,
                   boundary_enabled_var=_Var(True),
                   _select_step_by_key=lambda key: self.assertEqual(key, "setup"),
                   _build_boundary_data=lambda: self.fail("image reached calibration preflight"),
                   save_all_default=lambda: self.fail("image started a video measurement"))
        app.save_and_run_measurement()
        self.assertTrue(warnings)
        self.assertIn("视频", " ".join(map(str, warnings[0])))

    def test_new_media_state_reset_clears_all_old_annotations(self):
        ns = _load_methods("_reset_annotation_state")
        old_point = (100, 200)
        app = _app(ns, "_reset_annotation_state",
                   keypoints=[old_point], keypoint_source_name="old.mp4",
                   _keypoint_geometry_warning="old warning",
                   manual_h_groups=[[old_point]], manual_v_groups=[[old_point]],
                   boundary_edges={key: [old_point] for key in
                                   ("top", "right", "bottom", "left")},
                   boundary_source_name="old.mp4", boundary_preview_lines=True,
                   train_area_points=[old_point], train_area_region=[[1, 2], [3, 4], [5, 6]],
                   train_area_tokens=_Var("1, 2, V1"),
                   _saved_preview_paths={"train_area": Path("/old/train_area_preview.jpg")},
                   marker_groups=[[old_point]], marker_equipment_types=["old"],
                   marker_group_closed=[True], marker_group_count_var=_Var("1"),
                   marker_equipment_type_var=_Var("old"), marker_layout_var=_Var("ring"),
                   _config_source_indices=[3], _active_marker_type_index=1,
                   boundary_edge_var=_Var("left"), boundary_enabled_var=_Var(False),
                   boundary_status_var=_Var("old status"), selected_vertex=_Var(2),
                   current_group=_Var(1), frame_index=_Var(45),
                   current_step_index=7)
        app._reset_annotation_state()
        self.assertFalse(app.keypoints)
        self.assertFalse(app.keypoint_source_name)
        self.assertFalse(app._keypoint_geometry_warning)
        self.assertFalse(any(app.manual_h_groups))
        self.assertFalse(any(app.manual_v_groups))
        self.assertFalse(any(app.boundary_edges.values()))
        self.assertFalse(app.boundary_source_name)
        self.assertFalse(app.boundary_preview_lines)
        self.assertFalse(app.train_area_points)
        self.assertFalse(app.train_area_region)
        self.assertFalse(app.train_area_tokens.get())
        self.assertFalse(app._saved_preview_paths)
        self.assertFalse(any(app.marker_groups))
        self.assertEqual(app.frame_index.get(), 0)
        self.assertEqual(app.current_step_index, 0)
        self.assertTrue(app.boundary_enabled_var.get())

    @staticmethod
    def _media_app(namespace, root, old_source, *, old_kind="video"):
        """Minimal real-file fake for _activate_new_media without Tk."""
        return _app(
            namespace, "_activate_new_media", project_dir=root,
            camera_name="camera1", source_path=old_source,
            store_base_root="store", store_root="store",
            store_root_var=_Var("store"), video_path_var=_Var(str(old_source)),
            frame_index=_Var(45), media_kind=old_kind,
            media_identity={"path": str(old_source.resolve()), "size": 1, "mtime_ns": 1},
            isolated_calibration_session=False, _measurement_running=False,
            selected_pitch_config=_Var("QingYuanConfiguration"),
            _media_file_identity=lambda path: {
                "path": str(Path(path).resolve()), "size": 1, "mtime_ns": 1},
            _load_config=lambda: {},
            _reset_annotation_state=lambda: None,
            set_frame=lambda _frame: None,
            _sync_media_controls=lambda: None,
            _select_step=lambda _index: None,
            _refresh_step_list=lambda: None,
            _refresh_finish_summary=lambda: None,
            _refresh_store_dir_hint=lambda: None,
            status=_Var(""),
        )

    def test_reopening_same_source_keeps_annotations_and_store_root(self):
        ns = _load_methods("_activate_new_media")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_source = root / "old.mp4"
            old_source.write_bytes(b"video")
            app = self._media_app(ns, root, old_source)
            app.keypoints = [(100, 200)]
            app._reset_annotation_state = lambda: self.fail("same source reset annotations")
            app.set_frame = lambda _frame: None
            ns["messagebox"] = SimpleNamespace(
                askokcancel=lambda *_args: self.fail("same source asked for new session"))
            self.assertTrue(app._activate_new_media(old_source, object(), "video"))
            self.assertEqual(app.keypoints, [(100, 200)])
            self.assertEqual(app.store_root, "store")
            self.assertEqual(app.frame_index.get(), 0)

    def test_new_session_isolated_but_old_store_remains(self):
        """New-media output must not alias or remove the old camera's output."""
        ns = _load_methods("_activate_new_media")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_store = root / "store" / "camera1"
            old_store.mkdir(parents=True)
            old_file = old_store / "field_boundary.json"
            old_file.write_text("old annotation", encoding="utf-8")
            old_manual = root / "store" / "manual" / "camera1"
            old_manual.mkdir(parents=True)
            old_report = old_manual / "measurement_report.txt"
            old_report.write_text("old report", encoding="utf-8")
            (root / "config.yaml").write_text("old task configuration", encoding="utf-8")
            # The app fake receives only the public state expected by the
            # importer; additional UI callbacks can be harmless no-ops.
            app = self._media_app(ns, root, root / "old.mp4")
            ns["messagebox"] = SimpleNamespace(
                askokcancel=lambda *_args: True,
                showwarning=lambda *_args: self.fail("unexpected warning"),
                showerror=lambda *_args: self.fail("unexpected error"),
            )
            ns["yaml"] = SimpleNamespace(
                safe_dump=lambda _config, handle, **_kw: handle.write("test config"))
            # Do not assume the exact session folder name, only that it differs.
            accepted = app._activate_new_media(root / "new.mp4", object(), "video")
            self.assertTrue(accepted)
            self.assertNotEqual(Path(app.store_root), Path("store"))
            self.assertTrue(old_file.exists())
            self.assertEqual(old_file.read_text(encoding="utf-8"), "old annotation")
            self.assertTrue(old_report.exists())
            self.assertEqual(old_report.read_text(encoding="utf-8"), "old report")
            self.assertEqual(
                (root / app.store_root / "previous_config.yaml").read_text(encoding="utf-8"),
                "old task configuration",
            )

    def test_new_image_clears_measurement_video_config(self):
        ns = _load_methods("_activate_new_media")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_source = root / "old.mp4"
            old_source.write_bytes(b"video")
            image = root / "new.jpg"
            image.write_bytes(b"image")
            app = self._media_app(ns, root, old_source)
            ns["messagebox"] = SimpleNamespace(
                askokcancel=lambda *_args: True,
                showwarning=lambda *_args: self.fail("unexpected warning"),
                showerror=lambda *_args: self.fail("unexpected error"),
            )
            saved = {}
            ns["yaml"] = SimpleNamespace(
                safe_dump=lambda config, handle, **_kw:
                (saved.update(config), handle.write("test config")))
            self.assertTrue(app._activate_new_media(image, object(), "image"))
            self.assertEqual(app.media_kind, "image")
            self.assertEqual(app.video_path_var.get(), "")
            self.assertEqual(saved["video_name"], "")
            self.assertEqual(saved["annotation_media_kind"], "image")


if __name__ == "__main__":
    unittest.main()
