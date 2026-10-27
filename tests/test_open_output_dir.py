"""Headless checks for the two result-folder buttons' destinations."""

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


class OpenOutputDirTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
