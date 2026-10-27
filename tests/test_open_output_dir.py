"""Headless checks for the result-folder buttons' destinations and layout."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"


def _load_directory_method(name, opened, messages):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {
        "open_path_in_file_manager": opened.append,
        "messagebox": SimpleNamespace(showinfo=lambda *args: messages.append(args)),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def _load_save_preview(chosen_path):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == "save_preview")
    namespace = {
        "Path": Path,
        "filedialog": SimpleNamespace(asksaveasfilename=lambda **_kwargs: chosen_path),
        "messagebox": SimpleNamespace(showinfo=lambda *_args: None),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["save_preview"]


def _load_step_button_visibility():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_sync_step_result_button")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_sync_step_result_button"]


def _button_grid_position(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    builder = next(node for node in owner.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_build_side_panels")
    for node in ast.walk(builder):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "grid" and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self" and node.func.value.attr == name):
            return {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords
                    if keyword.arg in ("row", "column", "columnspan", "sticky")}
    raise AssertionError(f"missing grid placement for {name}")


class OpenOutputDirTests(unittest.TestCase):
    def test_result_button_sits_below_preview_and_spans_both_columns(self):
        preview = _button_grid_position("preview_step_button")
        result = _button_grid_position("open_step_result_button")
        self.assertEqual(result, {
            "row": preview["row"] + 1, "column": 0,
            "columnspan": 2, "sticky": "ew",
        })

    def test_result_button_appears_for_boundary_area_and_markers(self):
        sync = _load_step_button_visibility()
        calls = []
        button = SimpleNamespace(grid=lambda: calls.append("show"),
                                 grid_remove=lambda: calls.append("hide"))
        app = SimpleNamespace(current_step=SimpleNamespace(key="train_area"),
                              open_step_result_button=button)
        for key in ("train_area", "markers", "field_boundary", "finish"):
            app.current_step.key = key
            sync(app)
        self.assertEqual(calls, ["show", "show", "show", "hide"])

    def test_opens_camera_calibration_dir_even_when_manual_dir_exists(self):
        opened, messages = [], []
        open_output_dir = _load_directory_method("open_output_dir", opened, messages)
        with TemporaryDirectory() as directory:
            camera_dir = Path(directory) / "camera1"
            manual_dir = Path(directory) / "manual" / "camera1"
            camera_dir.mkdir()
            manual_dir.mkdir(parents=True)
            app = SimpleNamespace(_store_dir=lambda: camera_dir,
                                  _manual_dir=lambda: manual_dir)
            open_output_dir(app)
            self.assertEqual(opened, [str(camera_dir)])
            self.assertEqual(messages, [])

    def test_does_not_fall_back_to_manual_dir(self):
        opened, messages = [], []
        open_output_dir = _load_directory_method("open_output_dir", opened, messages)
        with TemporaryDirectory() as directory:
            camera_dir = Path(directory) / "camera1"
            manual_dir = Path(directory) / "manual" / "camera1"
            manual_dir.mkdir(parents=True)
            app = SimpleNamespace(_store_dir=lambda: camera_dir,
                                  _manual_dir=lambda: manual_dir)
            open_output_dir(app)
            self.assertEqual(opened, [])
            self.assertEqual(len(messages), 1)
            self.assertIn("相机标定结果目录尚不存在", messages[0][1])

    def test_measurement_button_opens_manual_camera_dir(self):
        opened, messages = [], []
        open_measurement_dir = _load_directory_method("open_measurement_dir", opened, messages)
        with TemporaryDirectory() as directory:
            camera_dir = Path(directory) / "camera1"
            manual_dir = Path(directory) / "manual" / "camera1"
            camera_dir.mkdir()
            manual_dir.mkdir(parents=True)
            app = SimpleNamespace(_store_dir=lambda: camera_dir,
                                  _manual_dir=lambda: manual_dir)
            open_measurement_dir(app)
            self.assertEqual(opened, [str(manual_dir)])
            self.assertEqual(messages, [])

    def test_measurement_button_does_not_open_camera_dir_when_manual_missing(self):
        opened, messages = [], []
        open_measurement_dir = _load_directory_method("open_measurement_dir", opened, messages)
        with TemporaryDirectory() as directory:
            camera_dir = Path(directory) / "camera1"
            manual_dir = Path(directory) / "manual" / "camera1"
            camera_dir.mkdir()
            app = SimpleNamespace(_store_dir=lambda: camera_dir,
                                  _manual_dir=lambda: manual_dir)
            open_measurement_dir(app)
            self.assertEqual(opened, [])
            self.assertEqual(len(messages), 1)
            self.assertIn("器材测量目录尚不存在", messages[0][1])

    def test_step_buttons_open_their_respective_default_preview_folders(self):
        opened, messages = [], []
        open_step_result_dir = _load_directory_method("open_step_result_dir", opened, messages)
        with TemporaryDirectory() as directory:
            camera_dir = Path(directory) / "camera1"
            manual_dir = Path(directory) / "manual" / "camera1"
            camera_dir.mkdir()
            manual_dir.mkdir(parents=True)
            paths = {"field_boundary": camera_dir / "field_boundary.json",
                     "train_area": camera_dir / "train_area.json",
                     "markers": manual_dir / "markers.json"}
            app = SimpleNamespace(_saved_preview_paths={}, _step_save_path=paths.get)
            open_step_result_dir(app, "field_boundary")
            open_step_result_dir(app, "train_area")
            open_step_result_dir(app, "markers")
            self.assertEqual(opened, [str(camera_dir), str(camera_dir), str(manual_dir)])
            self.assertEqual(messages, [])

    def test_step_button_opens_custom_preview_location_after_save(self):
        opened, messages = [], []
        open_step_result_dir = _load_directory_method("open_step_result_dir", opened, messages)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            custom_dir = root / "custom"
            custom_dir.mkdir()
            app = SimpleNamespace(
                _saved_preview_paths={"train_area": custom_dir / "area.png"},
                _step_save_path=lambda _key: root / "camera1" / "train_area.json",
            )
            open_step_result_dir(app, "train_area")
            self.assertEqual(opened, [str(custom_dir)])
            self.assertEqual(messages, [])

    def test_step_button_explains_missing_preview_folder(self):
        opened, messages = [], []
        open_step_result_dir = _load_directory_method("open_step_result_dir", opened, messages)
        with TemporaryDirectory() as directory:
            app = SimpleNamespace(
                _saved_preview_paths={},
                _step_save_path=lambda _key: Path(directory) / "camera1" / "train_area.json",
            )
            open_step_result_dir(app, "train_area")
            self.assertEqual(opened, [])
            self.assertIn("请先生成并保存本步预览图", messages[0][1])

    def test_preview_save_remembers_custom_result_folder(self):
        writes = []
        with TemporaryDirectory() as directory:
            root = Path(directory)
            custom_file = root / "custom" / "区域.png"
            custom_file.parent.mkdir()
            save_preview = _load_save_preview(str(custom_file))
            status = SimpleNamespace(set=lambda value: None)
            app = SimpleNamespace(
                current_step=SimpleNamespace(key="train_area"),
                _default_preview_path=lambda: root / "camera1" / "train_area_preview.jpg",
                _render_preview_image=lambda: "rendered",
                _write_image=lambda path, image: writes.append((path, image)),
                _saved_preview_paths={}, status=status,
            )
            save_preview(app)
            self.assertEqual(writes, [(custom_file, "rendered")])
            self.assertEqual(app._saved_preview_paths["train_area"], custom_file)


if __name__ == "__main__":
    unittest.main()
