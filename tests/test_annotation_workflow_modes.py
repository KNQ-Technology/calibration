"""Headless regression checks for the two calibration wizard workflows.

The GUI module has import-time environment setup, so these tests execute only
the relevant production definitions and methods extracted from its AST.
"""

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from typing import List
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"
FOUR_EDGE_KEYS = ["setup", "field_boundary", "train_area", "markers", "finish"]
LEGACY_KEYS = [
    "setup", "field_boundary", "keypoints", "manual_h", "manual_v",
    "train_area", "markers", "finish",
]


def _load_workflow(*method_names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    definitions = [
        node for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "StepDef")
        or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.target.id == "STEP_DEFS")
        or (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id in {"STEP_KEYS", "LEGACY_STEP_KEYS"}
                    for target in node.targets))
    ]
    methods = [node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in method_names]
    assert len(methods) == len(method_names), "A required wizard method is missing"
    namespace = {"dataclass": dataclass, "datetime": datetime,
                 "List": List, "Path": Path,
                 "BOUNDARY_KEYS": ("top", "right", "bottom", "left")}
    exec(compile(ast.Module(body=definitions + methods, type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    return namespace


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Widget:
    def __init__(self):
        self.disabled = False
        self.config = {}

    def state(self, spec):
        self.disabled = "disabled" in spec

    def configure(self, **kwargs):
        self.config.update(kwargs)


class _StepListbox:
    def __init__(self):
        self.items = []
        self.config = {}

    def delete(self, _first, _last):
        self.items.clear()

    def configure(self, **kwargs):
        self.config.update(kwargs)

    def insert(self, _position, label):
        self.items.append(label)

    def itemconfigure(self, _index, **_kwargs):
        pass

    def selection_clear(self, _first, _last):
        pass

    def selection_set(self, _index):
        pass


def _wizard(enabled, methods):
    namespace = _load_workflow(*methods)
    app = SimpleNamespace(
        boundary_enabled_var=_Var(enabled),
        boundary_edges={key: [] for key in ("top", "right", "bottom", "left")},
    )
    for name in methods:
        setattr(app, name, MethodType(namespace[name], app))
    return app, namespace


class AnnotationWorkflowModeTests(unittest.TestCase):
    def test_disabled_boundary_step_does_not_draw_old_edges(self):
        app, _ = _wizard(False, ("_draw_annotations",))
        app.current_step = SimpleNamespace(key="field_boundary")
        app._draw_annotations()  # no canvas or boundary data should be needed

    def test_invalid_boundary_is_archived_only_after_confirmation(self):
        app, namespace = _wizard(False, ("_archive_invalid_boundary_for_legacy",))
        app.status = _Var("")
        approved = False
        namespace["messagebox"] = SimpleNamespace(
            askyesno=lambda *_args: approved,
            showwarning=lambda *_args: self.fail("unexpected backup failure"),
        )
        with TemporaryDirectory() as directory:
            boundary = Path(directory) / "field_boundary.json"
            boundary.write_text("original invalid data", encoding="utf-8")
            self.assertFalse(app._archive_invalid_boundary_for_legacy(
                boundary, ValueError("mismatch")))
            self.assertTrue(boundary.exists())

            approved = True
            self.assertTrue(app._archive_invalid_boundary_for_legacy(
                boundary, ValueError("mismatch")))
            self.assertFalse(boundary.exists())
            backups = list(Path(directory).glob("field_boundary.disabled-backup-*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "original invalid data")

    def test_default_mode_has_five_steps_in_requested_order(self):
        app, namespace = _wizard(True, ("_active_steps",))
        self.assertEqual([step.key for step in app._active_steps()], FOUR_EDGE_KEYS)
        self.assertEqual(len(namespace["STEP_DEFS"]), 8)
        self.assertEqual([step.key for step in namespace["STEP_DEFS"]], LEGACY_KEYS)
        self.assertFalse(next(step for step in app._active_steps()
                              if step.key == "field_boundary").optional)

    def test_step_list_rebuilds_numbering_when_mode_changes(self):
        app, namespace = _wizard(True, ("_active_steps", "_refresh_step_list"))
        namespace["tk"] = SimpleNamespace(END="end")
        app.current_step_index = 1
        app.step_listbox = _StepListbox()
        app._step_complete = lambda _key: False
        app._refresh_cache_panel = lambda: None

        app._refresh_step_list()
        self.assertEqual(app.step_listbox.config["height"], 5)
        self.assertIn("2. 四条边线标定", app.step_listbox.items[1])
        self.assertIn("3. 训练区域", app.step_listbox.items[2])
        self.assertIn("5. 保存与测量", app.step_listbox.items[4])

        app.boundary_enabled_var.set(False)
        app._refresh_step_list()
        self.assertEqual(app.step_listbox.config["height"], 8)
        self.assertIn("3. 关键点标注", app.step_listbox.items[2])
        self.assertIn("4. 横向直线标注", app.step_listbox.items[3])
        self.assertIn("5. 纵向直线标注", app.step_listbox.items[4])
        self.assertIn("8. 保存与测量", app.step_listbox.items[7])

    def test_disabling_four_edges_reveals_all_eight_required_legacy_steps(self):
        app, _ = _wizard(True, ("_active_steps", "_step_complete",
                                "_first_incomplete_step"))
        app.boundary_enabled_var.set(False)
        self.assertEqual([step.key for step in app._active_steps()], LEGACY_KEYS)
        self.assertTrue(app._step_complete("field_boundary"))
        app._step_complete = lambda key: key in {"setup", "field_boundary"}
        self.assertEqual(app._first_incomplete_step(), 2)  # keypoints
        app._step_complete = lambda key: key in {"setup", "field_boundary", "keypoints"}
        self.assertEqual(app._first_incomplete_step(), 3)  # horizontal lines
        app._step_complete = lambda key: key in {
            "setup", "field_boundary", "keypoints", "manual_h"}
        self.assertEqual(app._first_incomplete_step(), 4)  # vertical lines

    def test_four_edge_step_needs_all_four_edges_only_when_enabled(self):
        app, _ = _wizard(True, ("_step_complete",))
        self.assertFalse(app._step_complete("field_boundary"))
        app.boundary_edges = {key: [(0, 0), (1, 1)] for key in app.boundary_edges}
        self.assertTrue(app._step_complete("field_boundary"))
        app.boundary_enabled_var.set(False)
        app.boundary_edges["top"] = []
        self.assertTrue(app._step_complete("field_boundary"))

    def test_disabling_four_edges_disables_annotation_and_canvas_actions(self):
        app, _ = _wizard(True, ("_sync_boundary_controls",))
        app.current_step = SimpleNamespace(key="field_boundary")
        app.boundary_edit_widgets = [_Widget() for _ in range(7)]
        app.undo_button = _Widget()
        app.clear_step_button = _Widget()
        app.preview_step_button = _Widget()
        app.save_step_button = _Widget()

        app.boundary_enabled_var.set(False)
        app._sync_boundary_controls()
        for widget in app.boundary_edit_widgets + [
            app.undo_button, app.clear_step_button, app.preview_step_button,
        ]:
            self.assertTrue(widget.disabled)
        self.assertEqual(app.save_step_button.config.get("text"), "保存旧模式")

        app.boundary_enabled_var.set(True)
        app._sync_boundary_controls()
        for widget in app.boundary_edit_widgets + [
            app.undo_button, app.clear_step_button, app.preview_step_button,
        ]:
            self.assertFalse(widget.disabled)


if __name__ == "__main__":
    unittest.main()
