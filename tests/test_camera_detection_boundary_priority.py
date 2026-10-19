"""Exercise calibration dispatch and detection filtering without GPU imports.

``camera.py`` imports optional tracking and inference stacks at module import
time.  Extracting the relevant methods and dispatch branch keeps these tests
independent of those stacks while still executing production code.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest

def _camera_source():
    source_path = Path(__file__).resolve().parents[1] / "camera.py"
    source = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return source_path, source


def _load_filter_method(class_name, method_name):
    source_path, source = _camera_source()
    owner = next(
        node for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    namespace = {"np": np, "Tuple": Tuple}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[method.name]


filter_detections_in_coverage = _load_filter_method(
    "CameraDetection", "filter_detections_in_coverage"
)
filter_points_in_coverage = _load_filter_method("Camera", "filter_points_in_coverage")


def _run_calibration_branch(camera, compute_boundary_homography):
    """Execute Camera.__init__'s actual four-edge/legacy dispatch in isolation."""
    source_path, source = _camera_source()
    owner = next(node for node in source.body
                 if isinstance(node, ast.ClassDef) and node.name == "Camera")
    init = next(node for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    branch = next(node for node in init.body
                  if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "self.field_boundary_data is not None")
    namespace = {
        "self": camera,
        "np": np,
        "Path": Path,
        "compute_boundary_homography": compute_boundary_homography,
    }
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(source_path), "exec"), namespace)


def _detections(*boxes):
    return np.asarray([list(box) + [0.9, 0.9, 0] for box in boxes], dtype=np.float32)


def test_four_edge_mode_filters_by_mapped_pitch_coordinates_not_raw_pixel_quad():
    class Camera:
        field_boundary_quad = np.asarray(
            [[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32
        )
        config = SimpleNamespace(length=100, width=100)
        filter_points_in_coverage = filter_points_in_coverage

        def pixel_to_pitch(self, points):
            # Deliberately disagree with the saved raw-pixel quadrilateral:
            # an in-polygon foot projects out, while an out-of-polygon foot
            # projects in.  The pitch range must be the only final decision.
            return np.asarray([[150, 50], [50, 50], [0, 0]], dtype=np.float32)

    detection = SimpleNamespace(
        camera=Camera(),
        raw_detections=_detections(
            (10, 60, 30, 80),  # foot (20, 80): inside
            (85, 60, 105, 80),  # foot (95, 80): outside
            (0, 0, 20, 10),  # foot (10, 10): on the line
        ),
    )

    positions, mask = filter_detections_in_coverage(detection)

    np.testing.assert_array_equal(detection.pixel_positions, [[20, 80], [95, 80], [10, 10]])
    np.testing.assert_array_equal(positions, [[150, 50], [50, 50], [0, 0]])
    assert mask.tolist() == [False, True, True]
    assert detection.mask is mask


def test_four_edge_calibration_skips_both_legacy_keypoint_solvers():
    calls = []

    def compute_boundary(data):
        calls.append("four_edges")
        assert data == {"source": ""}
        return np.eye(3), np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])

    def forbidden_legacy():
        raise AssertionError("启用四边标定时不得求解旧关键点/畸变模式")

    camera = SimpleNamespace(
        field_boundary_data={"source": ""},
        field_boundary_path=Path("store/camera1/field_boundary.json"),
        video_path="video.mp4",
        id="camera1",
        enable_distortion_correction=True,
        _calibrate=forbidden_legacy,
        _calibrate_with_distortion=forbidden_legacy,
    )

    _run_calibration_branch(camera, compute_boundary)

    assert calls == ["four_edges"]
    np.testing.assert_array_equal(camera.homography_matrix, np.eye(3))
    assert camera.field_boundary_quad.shape == (4, 2)
    assert camera.enable_distortion_correction is False


@pytest.mark.parametrize("boundary_state", ["missing", "none"])
def test_no_manual_boundary_preserves_original_pitch_rectangle_filter(boundary_state):
    class Camera:
        def __init__(self):
            self.pitch_filter_calls = 0

        def pixel_to_pitch(self, points):
            return points + np.array([1, 2], dtype=np.float32)

        def filter_points_in_coverage(self, points):
            self.pitch_filter_calls += 1
            mask = points[:, 0] < 50
            return points[mask], mask

    camera = Camera()
    if boundary_state == "none":
        camera.field_boundary_quad = None
    detection = SimpleNamespace(
        camera=camera,
        raw_detections=_detections((10, 60, 30, 80), (85, 60, 105, 80)),
    )

    positions, mask = filter_detections_in_coverage(detection)

    np.testing.assert_array_equal(positions, [[21, 82], [96, 82]])
    assert mask.tolist() == [True, False]
    assert camera.pitch_filter_calls == 1


def test_no_detections_does_not_call_either_boundary_mode():
    detection = SimpleNamespace(camera=None, raw_detections=np.empty((0, 7)))

    positions, mask = filter_detections_in_coverage(detection)

    assert positions.shape == (0, 2)
    assert detection.pixel_positions.shape == (0, 2)
    assert mask.dtype == bool and mask.shape == (0,)
