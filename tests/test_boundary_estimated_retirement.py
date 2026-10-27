"""Keep new four-edge saves free of the retired estimate flag."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import unittest

from field_boundary import compute_boundary_homography, validate_boundary_data


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"
EDGE_NAMES = ("top", "right", "bottom", "left")


class DemoPitch:
    length = 10000
    width = 6000


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def build_boundary_data_method():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    owner = next(node for node in module.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_build_boundary_data")
    namespace = {
        "Any": Any, "Dict": Dict,
        "BOUNDARY_KEYS": EDGE_NAMES,
        "BOUNDARY_LABELS": dict(zip(EDGE_NAMES, EDGE_NAMES)),
        "validate_boundary_data": validate_boundary_data,
        "compute_boundary_homography": compute_boundary_homography,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method.name]


class BoundaryEstimateRetirementTests(unittest.TestCase):
    def test_new_save_omits_estimated_and_legacy_data_still_computes(self):
        app = SimpleNamespace(
            boundary_edges={
                "top": [(200, 150), (1050, 180)],
                "right": [(1050, 180), (1160, 650)],
                "bottom": [(100, 650), (1160, 650)],
                "left": [(200, 150), (100, 650)],
            },
            boundary_enabled_var=Value(True),
            camera_name="camera1", pitch_config=DemoPitch(),
            image_w=1280, image_h=720,
            source_path=Path("frame.mp4"), frame_index=Value(0),
        )
        app._step_save_path = lambda _key: SimpleNamespace(exists=lambda: False)
        data = build_boundary_data_method()(app)
        for edge in data["edges"].values():
            self.assertEqual(list(edge), ["points"])
        legacy = {**data, "edges": {**data["edges"],
                   "bottom": {**data["edges"]["bottom"], "estimated": True}}}
        clean = validate_boundary_data(legacy, camera_id="camera1", config=DemoPitch())
        compute_boundary_homography(clean)


if __name__ == "__main__":
    unittest.main()
