"""Check that isolated media sessions cannot load a previous camera's files."""

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


def _camera_method(name):
    source_path = Path(__file__).resolve().parents[1] / "camera.py"
    source = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    owner = next(
        node for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "Camera"
    )
    method = next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"Path": Path, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[name]


setup_search_paths = _camera_method("_setup_search_paths")
find_calibration_file = _camera_method("_find_calibration_file")


class StrictCalibrationDirTests(unittest.TestCase):
    def test_isolated_session_does_not_fall_back_to_old_camera_files(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                old_dir = Path("store/camera1")
                old_dir.mkdir(parents=True)
                old_file = old_dir / "field_boundary.json"
                old_file.write_text("{}", encoding="utf-8")
                isolated_dir = Path("store/_media_sessions/new-video/camera1")
                isolated_dir.mkdir(parents=True)
                camera = SimpleNamespace(
                    store_dir=isolated_dir,
                    name="camera1",
                    id="camera1",
                    mcs_name=None,
                    strict_calibration_dir=True,
                )

                setup_search_paths(camera)
                self.assertEqual(camera.search_paths, [isolated_dir])
                self.assertIsNone(find_calibration_file(camera, "field_boundary.json"))

                camera.strict_calibration_dir = False
                setup_search_paths(camera)
                self.assertEqual(
                    find_calibration_file(camera, "field_boundary.json"), old_file
                )
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
