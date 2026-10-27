"""Headless checks for the four-edge wizard's presentation order.

The annotation client performs environment setup when imported, so these
tests execute only the relevant definitions extracted from its AST.
"""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, Optional
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"


def _definitions(*method_names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    constants = [node for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in {
                     "BOUNDARY_KEYS", "BOUNDARY_UI_ORDER", "BOUNDARY_LABELS",
                 } for target in node.targets)]
    methods = [node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in method_names]
    assert len(methods) == len(method_names)
    namespace = {"Any": Any, "Dict": Dict, "Optional": Optional, "Path": Path,
                 "tk": SimpleNamespace(END="end")}
    exec(compile(ast.Module(body=constants + methods, type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    return tree, owner, namespace


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Listbox:
    def __init__(self):
        self.items = []

    def delete(self, *_args):
        self.items.clear()

    def insert(self, _position, text):
        self.items.append(text)


class BoundaryUiOrderTests(unittest.TestCase):
    def test_display_order_does_not_change_saved_geometry_keys(self):
        tree, owner, namespace = _definitions()
        self.assertEqual(namespace["BOUNDARY_KEYS"], ("top", "right", "bottom", "left"))
        self.assertEqual(namespace["BOUNDARY_UI_ORDER"], ("left", "top", "right", "bottom"))
        self.assertEqual(set(namespace["BOUNDARY_UI_ORDER"]), set(namespace["BOUNDARY_KEYS"]))
        self.assertTrue(all("（" not in label for label in namespace["BOUNDARY_LABELS"].values()))

        panel = next(node for node in owner.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_build_boundary_panel")
        panel_source = ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), panel)
        self.assertIn("enumerate(BOUNDARY_UI_ORDER)", panel_source)
        self.assertIn('text="启用四边标定"', panel_source)
        self.assertNotIn("关闭后使用旧模式", panel_source)

        client_source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("tk.StringVar(value=BOUNDARY_UI_ORDER[0])", client_source)
        self.assertIn("self.boundary_edge_var.set(BOUNDARY_UI_ORDER[0])", client_source)
        self.assertIn("依次选择左、上、右、下边线", client_source)

    def test_incomplete_edge_message_follows_marking_order(self):
        _, _, namespace = _definitions("_build_boundary_data")
        with TemporaryDirectory() as directory:
            app = SimpleNamespace(
                boundary_enabled_var=_Variable(True),
                boundary_edges={key: [] for key in namespace["BOUNDARY_KEYS"]},
                _step_save_path=lambda _key: Path(directory) / "field_boundary.json",
            )
            with self.assertRaisesRegex(
                ValueError, "未完成：左端线、上边线、右端线、下边线"
            ):
                namespace["_build_boundary_data"](app)

    def test_annotation_list_follows_marking_order(self):
        _, _, namespace = _definitions("_refresh_list")
        listbox = _Listbox()
        app = SimpleNamespace(
            annotation_list=listbox,
            current_step=SimpleNamespace(key="field_boundary"),
            boundary_edges={key: [] for key in namespace["BOUNDARY_KEYS"]},
        )
        namespace["_refresh_list"](app)
        self.assertEqual(listbox.items, [
            "左端线：0/2", "上边线：0/2", "右端线：0/2", "下边线：0/2",
        ])

    def test_second_point_selects_next_unfinished_edge_and_wraps(self):
        _, _, namespace = _definitions(
            "_next_unfinished_boundary_edge", "on_canvas_click")
        edges = {key: [] for key in namespace["BOUNDARY_KEYS"]}
        app = SimpleNamespace(
            canvas_to_image=lambda x, y: (x, y),
            current_step=SimpleNamespace(key="field_boundary"),
            boundary_enabled_var=_Variable(True),
            boundary_edge_var=_Variable("left"),
            boundary_edges=edges,
            boundary_preview_lines=True,
            status=_Variable(""),
            _refresh_boundary_status=lambda: None,
            _refresh_step_list=lambda: None,
            _redraw=lambda: None,
        )
        app._next_unfinished_boundary_edge = lambda current: (
            namespace["_next_unfinished_boundary_edge"](app, current))
        click = namespace["on_canvas_click"]

        click(app, SimpleNamespace(x=10, y=20))
        self.assertEqual(app.boundary_edge_var.get(), "left")
        click(app, SimpleNamespace(x=30, y=40))
        self.assertEqual(app.boundary_edge_var.get(), "top")
        self.assertEqual(len(edges["left"]), 2)
        self.assertIn("已自动切换到上边线", app.status.get())
        self.assertFalse(app.boundary_preview_lines)

        # A manually chosen later edge still advances in left/top/right/bottom
        # order, skipping complete edges and wrapping to any remaining one.
        app.boundary_edge_var.set("right")
        click(app, SimpleNamespace(x=50, y=60))
        click(app, SimpleNamespace(x=70, y=80))
        self.assertEqual(app.boundary_edge_var.get(), "bottom")
        click(app, SimpleNamespace(x=90, y=100))
        click(app, SimpleNamespace(x=110, y=120))
        self.assertEqual(app.boundary_edge_var.get(), "top")
        click(app, SimpleNamespace(x=130, y=140))
        click(app, SimpleNamespace(x=150, y=160))
        self.assertEqual(app.boundary_edge_var.get(), "top")
        self.assertIn("四条边线均已标满", app.status.get())

    def test_completed_edge_can_be_selected_cleared_and_marked_again(self):
        _, _, namespace = _definitions(
            "_next_unfinished_boundary_edge", "on_canvas_click",
            "clear_selected_boundary_edge")
        edges = {key: [] for key in namespace["BOUNDARY_KEYS"]}
        edges["left"] = [(10, 20), (30, 40)]
        app = SimpleNamespace(
            canvas_to_image=lambda x, y: (x, y),
            current_step=SimpleNamespace(key="field_boundary"),
            boundary_enabled_var=_Variable(True),
            boundary_edge_var=_Variable("left"),
            boundary_edges=edges,
            boundary_preview_lines=False,
            status=_Variable(""),
            _refresh_boundary_status=lambda: None,
            _refresh_step_list=lambda: None,
            _redraw=lambda: None,
        )
        app._next_unfinished_boundary_edge = lambda current: (
            namespace["_next_unfinished_boundary_edge"](app, current))
        click = namespace["on_canvas_click"]

        click(app, SimpleNamespace(x=50, y=60))
        self.assertEqual(edges["left"], [(10, 20), (30, 40)])
        self.assertEqual(app.boundary_edge_var.get(), "left")
        self.assertIn("先点“清除选中边线”", app.status.get())

        namespace["clear_selected_boundary_edge"](app)
        click(app, SimpleNamespace(x=50, y=60))
        click(app, SimpleNamespace(x=70, y=80))
        self.assertEqual(edges["left"], [(50, 60), (70, 80)])
        self.assertEqual(app.boundary_edge_var.get(), "top")

    def test_undo_after_auto_advance_reverts_previous_edge(self):
        _, _, namespace = _definitions(
            "_next_unfinished_boundary_edge", "on_canvas_click", "undo",
            "_on_boundary_edge_selected")
        edges = {key: [] for key in namespace["BOUNDARY_KEYS"]}
        app = SimpleNamespace(
            canvas_to_image=lambda x, y: (x, y),
            current_step=SimpleNamespace(key="field_boundary"),
            boundary_enabled_var=_Variable(True),
            boundary_edge_var=_Variable("left"),
            boundary_edges=edges,
            boundary_preview_lines=False,
            _boundary_last_advanced_from=None,
            status=_Variable(""),
            _refresh_boundary_status=lambda: None,
            _refresh_step_list=lambda: None,
            _redraw=lambda: None,
        )
        app._next_unfinished_boundary_edge = lambda current: (
            namespace["_next_unfinished_boundary_edge"](app, current))
        click = namespace["on_canvas_click"]
        click(app, SimpleNamespace(x=10, y=20))
        click(app, SimpleNamespace(x=30, y=40))
        self.assertEqual(app.boundary_edge_var.get(), "top")

        namespace["undo"](app)
        self.assertEqual(edges["left"], [(10, 20)])
        self.assertEqual(app.boundary_edge_var.get(), "left")

        click(app, SimpleNamespace(x=30, y=40))
        self.assertEqual(app.boundary_edge_var.get(), "top")
        app.boundary_edge_var.set("right")
        namespace["_on_boundary_edge_selected"](app)
        namespace["undo"](app)
        self.assertEqual(edges["left"], [(10, 20), (30, 40)])
        self.assertEqual(app.boundary_edge_var.get(), "right")


if __name__ == "__main__":
    unittest.main()
