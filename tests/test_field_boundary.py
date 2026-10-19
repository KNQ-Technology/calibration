"""Focused safety tests for the opt-in four-boundary test build."""

import json
import sys
import types
from copy import deepcopy

import cv2
import numpy as np
import pytest

from field_boundary import (
    BoundaryCalibrationError,
    compute_boundary_homography,
    load_boundary,
    point_in_polygon,
    points_in_polygon,
    validate_boundary_data,
)


class DemoPitch:
    length = 10000
    width = 6000


@pytest.fixture
def boundary():
    corners = {
        "tl": [200, 150],
        "tr": [1050, 180],
        "br": [1160, 650],
        "bl": [100, 650],
    }
    return {
        "schema_version": 1,
        "camera_id": "camera1",
        "config_type": "DemoPitch",
        "pitch_size_cm": [10000, 6000],
        "image_size": [1280, 720],
        "edges": {
            "top": {"points": [corners["tl"], corners["tr"]]},
            "right": {"points": [corners["tr"], corners["br"]]},
            "bottom": {"points": [corners["bl"], corners["br"]], "estimated": True},
            "left": {"points": [corners["tl"], corners["bl"]]},
        },
    }


def test_load_and_compute_four_edges(tmp_path, boundary):
    path = tmp_path / "field_boundary.json"
    assert load_boundary(path, "camera1", DemoPitch(), (1280, 720)) is None
    path.write_text(json.dumps(boundary), encoding="utf-8")
    data = load_boundary(path, "camera1", DemoPitch(), (1280, 720))
    assert data["enabled"] is True
    assert data["edges"]["bottom"]["estimated"] is True
    H, raw_quad = compute_boundary_homography(data)
    expected_quad = np.array([[200, 150], [1050, 180], [1160, 650], [100, 650]])
    expected_pitch = np.array([[0, 0], [10000, 0], [10000, 6000], [0, 6000]])
    np.testing.assert_allclose(raw_quad, expected_quad, atol=1e-4)
    actual = cv2.perspectiveTransform(raw_quad.reshape(-1, 1, 2), H).reshape(-1, 2)
    np.testing.assert_allclose(actual, expected_pitch, atol=0.01)


@pytest.mark.parametrize("change, message", [
    ({"camera_id": "camera2"}, "相机"),
    ({"config_type": "OtherPitch"}, "场地"),
    ({"pitch_size_cm": [9000, 6000]}, "长宽"),
    ({"image_size": [640, 360]}, "画面尺寸"),
    ({"schema_version": 2}, "版本"),
])
def test_stale_annotation_is_not_silent_legacy_fallback(tmp_path, boundary, change, message):
    data = {**boundary, **change}
    path = tmp_path / "field_boundary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BoundaryCalibrationError, match=message):
        load_boundary(path, "camera1", DemoPitch(), (1280, 720))


def test_disabled_annotation_is_explicit(boundary):
    boundary["enabled"] = False
    clean = validate_boundary_data(
        boundary, camera_id="camera1", config=DemoPitch(), image_size=(1280, 720)
    )
    assert clean["enabled"] is False
    with pytest.raises(BoundaryCalibrationError, match="已禁用"):
        compute_boundary_homography(clean)


def test_missing_or_outside_edge_is_rejected(boundary):
    del boundary["edges"]["left"]
    with pytest.raises(BoundaryCalibrationError, match="四条边"):
        validate_boundary_data(boundary)
    boundary["edges"]["left"] = {"points": [[200, 150], [100, 720]]}
    with pytest.raises(BoundaryCalibrationError, match="超出"):
        validate_boundary_data(boundary)


def test_near_parallel_and_crossed_edges_are_rejected(boundary):
    parallel = deepcopy(boundary)
    parallel["edges"]["left"]["points"] = [[100, 160], [1000, 192]]
    with pytest.raises(BoundaryCalibrationError, match="平行"):
        compute_boundary_homography(parallel)

    crossed = deepcopy(boundary)
    crossed["edges"]["right"]["points"] = [[100, 650], [200, 150]]
    with pytest.raises(BoundaryCalibrationError):
        compute_boundary_homography(crossed)


def test_opposite_camera_orientation_can_mirror_pitch_axes(boundary):
    # A camera on the other side of a field reverses the visual left/right
    # order without making the world-coordinate edge labels invalid.
    corners = {
        "tl": [1100, 150], "tr": [200, 180],
        "br": [100, 650], "bl": [1160, 650],
    }
    boundary["edges"] = {
        "top": {"points": [corners["tl"], corners["tr"]]},
        "right": {"points": [corners["tr"], corners["br"]]},
        "bottom": {"points": [corners["bl"], corners["br"]]},
        "left": {"points": [corners["tl"], corners["bl"]]},
    }
    matrix, quad = compute_boundary_homography(boundary)
    mapped = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    np.testing.assert_allclose(
        mapped, [[0, 0], [10000, 0], [10000, 6000], [0, 6000]], atol=0.01)


def test_far_extrapolated_intersection_is_rejected(boundary):
    # The right and top edges nearly meet far outside the image.
    boundary["edges"]["right"]["points"] = [[1150, 195], [1151, 197]]
    with pytest.raises(BoundaryCalibrationError):
        compute_boundary_homography(boundary)


def test_distortion_path_matches_camera_projection_with_zero_coefficients(boundary, monkeypatch):
    # The real correction module imports SciPy at module import time.  The
    # geometry unit test only needs to verify that this path feeds corrected
    # endpoints into the homography, so avoid a SciPy test dependency here.
    correction = types.ModuleType("de_distortion")
    correction.undistort_points = lambda points, *_args, **_kwargs: points
    monkeypatch.setitem(sys.modules, "de_distortion", correction)
    H, raw_quad = compute_boundary_homography(
        boundary,
        distortion_params=np.zeros(5),
        distortion_info={"center": (640.0, 360.0), "enable_k3": True},
    )
    expected = np.array([[0, 0], [10000, 0], [10000, 6000], [0, 6000]])
    mapped = cv2.perspectiveTransform(raw_quad.reshape(-1, 1, 2), H).reshape(-1, 2)
    np.testing.assert_allclose(mapped, expected, atol=0.01)


def test_polygon_filter_includes_boundary_and_excludes_nan(boundary):
    _, raw_quad = compute_boundary_homography(boundary)
    sample = np.array([[600, 400], [200, 150], [0, 0], [np.nan, 100]])
    assert points_in_polygon(sample, raw_quad).tolist() == [True, True, False, False]
    assert point_in_polygon([600, 400], raw_quad)
