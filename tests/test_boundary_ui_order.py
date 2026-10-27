"""Headless checks for the four-edge wizard's presentation order.

The annotation client performs environment setup when imported, so these
tests execute only the relevant definitions extracted from its AST.
"""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict
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
    namespace = {"Any": Any, "Dict": Dict, "Path": Path,
                 "tk": SimpleNamespace(END="end")}
    exec(compile(ast.Module(body=constants + methods, type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    return tree, owner, namespace


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


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


if __name__ == "__main__":
    unittest.main()
