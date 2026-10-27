"""The optional equipment editor must display and behave as zero groups."""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from annotation_client import AnnotationClient


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self):
        self.disabled = False

    def state(self, changes):
        if "disabled" in changes:
            self.disabled = True
        if "!disabled" in changes:
            self.disabled = False


def marker_editor(groups):
    widgets = (Widget(), Widget(), Widget(), Widget())
    app = SimpleNamespace(
        current_step=SimpleNamespace(key="markers"),
        marker_groups=[list(group) for group in groups],
        marker_equipment_types=["未指定"] * len(groups),
        marker_group_closed=[False] * len(groups),
        marker_group_count_var=Variable(str(len(groups))),
        current_group=Variable(1 if groups else 0),
        marker_equipment_type_var=Variable("未指定"),
        marker_layout_var=Variable("line"),
        marker_current_group_spin=widgets[0],
        marker_equipment_type_combo=widgets[1],
        marker_layout_buttons=widgets[2:],
        _active_marker_type_index=0 if groups else -1,
        _refresh_step_list=lambda: None,
        _redraw=lambda: None,
        status=Variable(""),
    )
    app._active_group_store = lambda: app.marker_groups
    app._sync_marker_group_controls = lambda: AnnotationClient._sync_marker_group_controls(app)
    return app, widgets


class MarkerDefaultZeroTests(unittest.TestCase):
    def test_initialization_no_longer_creates_placeholder_group(self):
        source = Path(__file__).resolve().parents[1] / "annotation_client.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        owner = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
        init = next(node for node in owner.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        text = ast.get_source_segment(source.read_text(encoding="utf-8"), init)
        self.assertIn("self.marker_groups: List[List[Point]] = []", text)
        self.assertIn("self.current_group = tk.IntVar(value=0)", text)

    def test_zero_disables_group_specific_controls_and_add_enables_them(self):
        app, widgets = marker_editor([])
        AnnotationClient._sync_marker_group_controls(app)
        self.assertEqual(app.marker_group_count_var.get(), "0")
        self.assertEqual(app.current_group.get(), 0)
        self.assertEqual(app._active_marker_type_index, -1)
        self.assertTrue(all(widget.disabled for widget in widgets))

        AnnotationClient.new_group(app)
        self.assertEqual(app.marker_groups, [[]])
        self.assertEqual(app.current_group.get(), 1)
        self.assertEqual(app.marker_group_count_var.get(), "1")
        self.assertTrue(all(not widget.disabled for widget in widgets))

    def test_setting_count_back_to_zero_clears_group_after_confirmation(self):
        app, widgets = marker_editor([[(12, 34)]])
        app.marker_equipment_types = ["标志桶"]
        app.marker_group_count_var.set("0")
        with patch("annotation_client.messagebox.askyesno", return_value=True) as confirm:
            AnnotationClient.apply_marker_group_count(app)
        confirm.assert_called_once()
        self.assertEqual(app.marker_groups, [])
        self.assertEqual(app.marker_equipment_types, [])
        self.assertEqual(app.marker_group_closed, [])
        self.assertEqual(app.current_group.get(), 0)
        self.assertTrue(all(widget.disabled for widget in widgets))
        self.assertIn("0 组器材", app.status.get())

    def test_empty_canvas_click_does_not_create_implicit_group(self):
        app, _widgets = marker_editor([])
        app.canvas_to_image = lambda *_args: (12, 34)
        app.ensure_group = lambda: self.fail("zero groups created implicitly")
        AnnotationClient.on_canvas_click(app, SimpleNamespace(x=12, y=34))
        self.assertEqual(app.marker_groups, [])
        self.assertIn("新增一组器材", app.status.get())

    def test_saved_markers_restore_modes_but_old_empty_config_starts_zero(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "markers.json"
            app, _widgets = marker_editor([])
            app.config_data = {"GROUP_CONFIGS": [{"group": 0, "is_closed": True}]}
            app.camera_name = "camera1"
            app.boundary_enabled_var = Variable(True)
            app._step_save_path = lambda key: path if key == "markers" else root / f"{key}.json"
            app.pitch_config = object()
            app.image_w, app.image_h = 1280, 720
            app.train_area_tokens = Variable("")

            AnnotationClient._load_existing_annotations(app)
            self.assertEqual(app.marker_groups, [])
            self.assertEqual(app.marker_group_count_var.get(), "0")

            path.write_text(json.dumps({"groups": [{"group_id": 0,
                "equipment_type": "标志桶", "points": [{"id": 0, "img_coord": [12, 34]}]}]}),
                encoding="utf-8")
            AnnotationClient._load_existing_annotations(app)
            self.assertEqual(app.marker_groups, [[(12, 34)]])
            self.assertEqual(app.marker_group_count_var.get(), "1")
            self.assertEqual(app.marker_group_closed, [True])
            self.assertEqual(app._config_source_indices, [0])


if __name__ == "__main__":
    unittest.main()
