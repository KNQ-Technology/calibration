"""Optional four-edge pitch calibration for the isolated test build.

Image edges are named in pitch-coordinate order: ``top`` is y=0,
``right`` is x=length, ``bottom`` is y=width, and ``left`` is x=0.
The saved points are in the *original* image.  Camera.pixel_to_pitch()
undistorts pixels before applying its homography, so the homography here is
calculated from intersections of the undistorted edge endpoints when lens
correction is enabled.  The returned quadrilateral remains in raw pixels for
filtering and drawing on the original frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


EDGE_NAMES = ("top", "right", "bottom", "left")
SCHEMA_VERSION = 1
MAX_FILE_BYTES = 1024 * 1024


class BoundaryCalibrationError(ValueError):
    """A manual boundary cannot safely define a pitch transform."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BoundaryCalibrationError(f"{label} 必须是数字")
    result = float(value)
    if not math.isfinite(result):
        raise BoundaryCalibrationError(f"{label} 必须是有限数字")
    return result


def _pair(value: Any, label: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise BoundaryCalibrationError(f"{label} 必须包含两个坐标")
    return (_number(value[0], f"{label}[0]"), _number(value[1], f"{label}[1]"))


def validate_boundary_data(
    data: Any,
    *,
    camera_id: Optional[str] = None,
    config: Any = None,
    image_size: Optional[Sequence[int]] = None,
) -> dict:
    """Validate a v1 annotation; never silently reinterpret stale geometry.

    ``config`` is the existing SoccerPitchConfiguration (or another pitch
    preset exposing ``length`` and ``width``).  ``image_size`` is (width,
    height), not the NumPy frame's (height, width).
    """
    if not isinstance(data, dict):
        raise BoundaryCalibrationError("边线标定文件必须是 JSON 对象")
    version = data.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise BoundaryCalibrationError(f"不支持的边线标定版本: {version!r}")
    saved_camera = data.get("camera_id")
    if not isinstance(saved_camera, str) or not saved_camera:
        raise BoundaryCalibrationError("camera_id 必须是非空字符串")
    if camera_id is not None and saved_camera != str(camera_id):
        raise BoundaryCalibrationError(
            f"标定属于相机 {saved_camera}，当前相机为 {camera_id}"
        )
    saved_type = data.get("config_type")
    if not isinstance(saved_type, str) or not saved_type:
        raise BoundaryCalibrationError("config_type 必须是非空字符串")
    if config is not None and saved_type != type(config).__name__:
        raise BoundaryCalibrationError(
            f"标定场地为 {saved_type}，当前场地为 {type(config).__name__}"
        )

    pitch_size = _pair(data.get("pitch_size_cm"), "pitch_size_cm")
    if pitch_size[0] <= 0 or pitch_size[1] <= 0:
        raise BoundaryCalibrationError("场地长宽必须大于零")
    if config is not None:
        expected = (float(config.length), float(config.width))
        if any(abs(actual - wanted) > 1e-6 for actual, wanted in zip(pitch_size, expected)):
            raise BoundaryCalibrationError("场地长宽已改变，请重新标定四条边线")

    saved_size = data.get("image_size")
    if (not isinstance(saved_size, (list, tuple)) or len(saved_size) != 2 or
            any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in saved_size)):
        raise BoundaryCalibrationError("image_size 必须是正整数 [宽, 高]")
    width, height = saved_size
    if image_size is not None and tuple(saved_size) != tuple(image_size):
        raise BoundaryCalibrationError(
            f"标定画面尺寸为 {width}×{height}，当前为 {image_size[0]}×{image_size[1]}"
        )

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise BoundaryCalibrationError("enabled 必须是布尔值")
    edges = data.get("edges")
    if not isinstance(edges, dict) or set(edges) != set(EDGE_NAMES):
        raise BoundaryCalibrationError("edges 必须恰好包含 top/right/bottom/left 四条边")

    normalized = dict(data)
    normalized["enabled"] = enabled
    normalized["pitch_size_cm"] = list(pitch_size)
    normalized["image_size"] = [width, height]
    normalized_edges = {}
    min_segment = max(4.0, min(width, height) * 0.005)
    for name in EDGE_NAMES:
        edge = edges[name]
        if not isinstance(edge, dict):
            raise BoundaryCalibrationError(f"{name} 边线格式错误")
        points = edge.get("points")
        if not isinstance(points, (list, tuple)) or len(points) != 2:
            raise BoundaryCalibrationError(f"{name} 边线需要恰好两个像素点")
        p0 = _pair(points[0], f"{name}.points[0]")
        p1 = _pair(points[1], f"{name}.points[1]")
        for x, y in (p0, p1):
            if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                raise BoundaryCalibrationError(f"{name} 边线点超出 {width}×{height} 原图")
        if math.dist(p0, p1) < min_segment:
            raise BoundaryCalibrationError(f"{name} 边线太短，无法稳定计算交点")
        estimated = edge.get("estimated", False)
        if not isinstance(estimated, bool):
            raise BoundaryCalibrationError(f"{name}.estimated 必须是布尔值")
        normalized_edges[name] = {**edge, "points": [list(p0), list(p1)], "estimated": estimated}
    normalized["edges"] = normalized_edges
    return normalized


def load_boundary(
    path: str | Path,
    camera_id: str,
    config: Any,
    image_size: Sequence[int],
) -> Optional[dict]:
    """Load a boundary or return None only when it has not been created.

    A present but invalid annotation raises BoundaryCalibrationError.  A
    disabled annotation is returned with ``enabled=False`` so the caller can
    explicitly select the legacy calibration path.
    """
    annotation_path = Path(path)
    try:
        if annotation_path.stat().st_size > MAX_FILE_BYTES:
            raise BoundaryCalibrationError("边线标定文件过大")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryCalibrationError(f"无法读取边线标定文件: {annotation_path}") from exc
    return validate_boundary_data(
        data, camera_id=camera_id, config=config, image_size=image_size
    )


def _line_intersection(a: np.ndarray, b: np.ndarray, label: str) -> np.ndarray:
    """Intersection of two 2-point *infinite* lines."""
    start_a, end_a = a
    start_b, end_b = b
    va = end_a - start_a
    vb = end_b - start_b
    length_product = np.linalg.norm(va) * np.linalg.norm(vb)
    denominator = float(va[0] * vb[1] - va[1] * vb[0])
    if length_product == 0 or abs(denominator) / length_product < 0.01:
        raise BoundaryCalibrationError(f"{label} 相邻边线接近平行，交点不可靠")
    offset = start_b - start_a
    t = float(offset[0] * vb[1] - offset[1] * vb[0]) / denominator
    intersection = start_a + t * va
    if not np.isfinite(intersection).all():
        raise BoundaryCalibrationError(f"{label} 边线交点不是有限坐标")
    return intersection


def _quad_from_edges(edges: Mapping[str, np.ndarray], image_size: Sequence[int]) -> np.ndarray:
    quad = np.stack([
        _line_intersection(edges["top"], edges["left"], "左上"),
        _line_intersection(edges["top"], edges["right"], "右上"),
        _line_intersection(edges["bottom"], edges["right"], "右下"),
        _line_intersection(edges["bottom"], edges["left"], "左下"),
    ])
    width, height = image_size
    diagonal = math.hypot(width, height)
    # An unseen corner may lie outside the frame; do not require it to be
    # visible, but reject an intersection several frame widths away.
    excess_x = np.maximum(np.maximum(-quad[:, 0], quad[:, 0] - (width - 1)), 0)
    excess_y = np.maximum(np.maximum(-quad[:, 1], quad[:, 1] - (height - 1)), 0)
    if np.any(np.hypot(excess_x, excess_y) > 2 * diagonal):
        raise BoundaryCalibrationError("边线交点离画面过远，外推不可靠")

    sides = np.roll(quad, -1, axis=0) - quad
    if np.any(np.linalg.norm(sides, axis=1) < max(5.0, min(width, height) * 0.01)):
        raise BoundaryCalibrationError("边线交点过近，场地四边形退化")
    next_sides = np.roll(sides, -1, axis=0)
    signed_turns = sides[:, 0] * next_sides[:, 1] - sides[:, 1] * next_sides[:, 0]
    if not (np.all(signed_turns > 1e-6) or np.all(signed_turns < -1e-6)):
        raise BoundaryCalibrationError("四条边线顺序错误或互相交叉，应形成凸四边形")
    area = float(cv2.contourArea(quad.astype(np.float32)))
    if area < max(64.0, width * height * 0.001):
        raise BoundaryCalibrationError("场地四边形面积太小，无法稳定标定")
    return quad.astype(np.float32)


def compute_boundary_homography(
    data: Mapping[str, Any],
    distortion_params: Any = None,
    distortion_info: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (undistorted-pixel-to-pitch H, raw TL/TR/BR/BL quadrilateral)."""
    clean = validate_boundary_data(data)
    if not clean["enabled"]:
        raise BoundaryCalibrationError("边线标定已禁用，请使用原关键点标定流程")
    raw_edges = {
        name: np.asarray(clean["edges"][name]["points"], dtype=np.float64)
        for name in EDGE_NAMES
    }
    raw_quad = _quad_from_edges(raw_edges, clean["image_size"])
    if distortion_params is not None:
        from de_distortion import undistort_points

        info = distortion_info or {}
        center = info.get("center", (
            clean["image_size"][0] / 2, clean["image_size"][1] / 2
        ))
        combined = np.concatenate([raw_edges[name] for name in EDGE_NAMES]).astype(np.float32)
        corrected = np.asarray(undistort_points(
            combined,
            distortion_params,
            center,
            enable_k3=info.get("enable_k3", True),
        ), dtype=np.float64)
        if corrected.shape != (8, 2) or not np.isfinite(corrected).all():
            raise BoundaryCalibrationError("去畸变后的边线点无效")
        working_edges = {
            name: corrected[2 * i:2 * i + 2] for i, name in enumerate(EDGE_NAMES)
        }
    else:
        working_edges = raw_edges
    source_quad = _quad_from_edges(working_edges, clean["image_size"])
    length, width = clean["pitch_size_cm"]
    destination = np.asarray(
        [[0, 0], [length, 0], [length, width], [0, width]], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(source_quad, destination)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise BoundaryCalibrationError("四边线无法生成稳定的场地坐标映射")
    projected = cv2.perspectiveTransform(source_quad.reshape(-1, 1, 2), H).reshape(-1, 2)
    if np.max(np.linalg.norm(projected - destination, axis=1)) > max(length, width) * 0.001:
        raise BoundaryCalibrationError("四边线映射残差过大，请检查边线顺序")
    return H, raw_quad


def points_in_polygon(points: Any, polygon: Any) -> np.ndarray:
    """Boolean mask for original-image points inside or on a polygon."""
    sample = np.asarray(points, dtype=np.float64)
    contour = np.asarray(polygon, dtype=np.float32)
    if sample.ndim != 2 or sample.shape[1] != 2:
        raise ValueError("points 必须为 (N, 2) 数组")
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        raise ValueError("polygon 必须有至少三个二维点")
    if not np.isfinite(contour).all():
        raise ValueError("polygon 含无效坐标")
    finite = np.isfinite(sample).all(axis=1)
    result = np.zeros(len(sample), dtype=bool)
    for index in np.flatnonzero(finite):
        x, y = sample[index]
        result[index] = cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
    return result


def point_in_polygon(point: Sequence[float], polygon: Any) -> bool:
    """Single-point convenience wrapper for points_in_polygon()."""
    return bool(points_in_polygon([point], polygon)[0])
