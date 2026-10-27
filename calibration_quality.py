"""Non-blocking quality checks for a pitch annotation before measurement.

These checks are deliberately separate from the existing structural validators:
an unusual camera angle, lens distortion, or a partially visible field can
trigger a warning without making the annotation unusable.  Callers should show
the issues to the operator and let them explicitly continue.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from field_boundary import compute_boundary_homography


@dataclass(frozen=True)
class CalibrationQualityIssue:
    code: str
    message: str


def _project_pitch_to_image(pitch_points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(matrix)
    return cv2.perspectiveTransform(
        pitch_points.astype(np.float32).reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)


def _residual_issue(
    pitch_points: np.ndarray,
    image_points: np.ndarray,
    matrix: np.ndarray,
    image_size: tuple[int, int],
    labels: Sequence[str],
    *,
    code: str,
    description: str,
) -> CalibrationQualityIssue | None:
    """Compare independent marked pixels with model points in the same units.

    Pixel error is used instead of pitch-centimetre error: distant points can
    have very different world-space uncertainty under strong perspective.
    The generous, image-diagonal-scaled limit leaves room for lens distortion
    when this lightweight preflight has not yet run the full correction step.
    """
    try:
        projected = _project_pitch_to_image(pitch_points, matrix)
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return CalibrationQualityIssue(code, f"{description}无法稳定回投到画面，请检查标点对应关系。")
    if projected.shape != image_points.shape or not np.isfinite(projected).all():
        return CalibrationQualityIssue(code, f"{description}出现无效的投影坐标，请检查标点对应关系。")

    errors = np.linalg.norm(projected.astype(np.float64) - image_points, axis=1)
    diagonal = math.hypot(*image_size)
    limit = max(24.0, diagonal * 0.025)
    high_count = int(np.count_nonzero(errors > limit))
    median = float(np.median(errors))
    # A small number of imprecise clicks should not alarm the operator.  A
    # quarter of the annotations disagreeing with one mapping is substantial.
    # A single badly paired point matters even when the median stays low.
    extreme_limit = max(100.0, diagonal * 0.05)
    if (median <= limit and
            high_count < max(3, math.ceil(len(errors) * 0.25)) and
            float(np.max(errors)) <= extreme_limit):
        return None
    worst = np.argsort(errors)[::-1][:3]
    named = "、".join(labels[index] for index in worst)
    return CalibrationQualityIssue(
        code,
        f"{description}与标点偏差较大：{high_count}/{len(errors)} 个点超过 "
        f"{limit:.0f} 像素，中位偏差 {median:.0f} 像素；请优先核对 {named}。"
        "此项为画面预检，镜头畸变也可能影响数值。",
    )


def _keypoint_arrays(
    pitch_config: Any,
    keypoints: Sequence[tuple[int, Sequence[float]]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vertices = pitch_config.vertices
    labels = pitch_config.labels
    pitch_points = []
    image_points = []
    point_labels = []
    for index, image in keypoints:
        index = int(index)
        if index < 0 or index >= len(vertices) or len(image) != 2:
            raise ValueError("关键点编号或坐标不在当前场地配置中")
        pitch_points.append(vertices[index])
        image_points.append(image)
        point_labels.append(str(labels[index]) if index < len(labels) else str(index + 1))
    pitch = np.asarray(pitch_points, dtype=np.float32).reshape(-1, 2)
    pixel = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(pitch).all() or not np.isfinite(pixel).all():
        raise ValueError("关键点包含非有限坐标")
    return pitch, pixel, point_labels


def inspect_calibration_quality(
    pitch_config: Any,
    image_size: tuple[int, int],
    keypoints: Sequence[tuple[int, Sequence[float]]] = (),
    *,
    field_boundary: Mapping[str, Any] | None = None,
    source_name: str | None = None,
    keypoint_source_name: str | None = None,
) -> list[CalibrationQualityIssue]:
    """Return advisory issues for the *active* calibration mode only.

    ``image_size`` is ``(width, height)`` and each keypoint is
    ``(vertex_index, (image_x, image_y))``.  An enabled four-edge annotation
    takes precedence, exactly as it does in Camera.  Nothing is written and a
    returned issue must never, by itself, prevent visualization or measurement.
    """
    issues: list[CalibrationQualityIssue] = []
    width, height = image_size
    if width <= 0 or height <= 0:
        return [CalibrationQualityIssue("image_size", "画面尺寸异常，无法完成标定质量预检。")]

    boundary_active = field_boundary is not None and field_boundary.get("enabled", True)
    if boundary_active:
        saved_source = field_boundary.get("source")
        if source_name and saved_source and Path(str(source_name)).name != Path(str(saved_source)).name:
            issues.append(CalibrationQualityIssue(
                "boundary_source",
                f"四边标定来自 {Path(str(saved_source)).name}，当前画面为 "
                f"{Path(str(source_name)).name}；请确认机位和画面构图未变化。",
            ))
        try:
            matrix, quad = compute_boundary_homography(field_boundary)
        except (ValueError, TypeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
            # Structural errors remain the responsibility of the existing
            # validator.  This function only reports them; it never mutates.
            issues.append(CalibrationQualityIssue(
                "boundary_geometry", f"四边标定无法完成质量预检：{exc}"
            ))
            return issues
        exterior = np.maximum(np.maximum(-quad, quad - [width - 1, height - 1]), 0)
        if np.max(np.linalg.norm(exterior, axis=1)) > math.hypot(width, height) * 0.5:
            issues.append(CalibrationQualityIssue(
                "boundary_extrapolation",
                "四条边线的延长线交点远离画面，投影对标点误差可能非常敏感；请检查边线方向。",
            ))
    else:
        matrix = None

    if not keypoints:
        return issues
    try:
        pitch_points, image_points, labels = _keypoint_arrays(pitch_config, keypoints)
    except (ValueError, TypeError, IndexError) as exc:
        issues.append(CalibrationQualityIssue("keypoint_format", f"关键点质量预检失败：{exc}"))
        return issues

    if keypoint_source_name and source_name and (
        Path(str(keypoint_source_name)).name != Path(str(source_name)).name
    ):
        issues.append(CalibrationQualityIssue(
            "keypoint_source",
            f"关键点取自 {Path(str(keypoint_source_name)).name}，当前画面为 "
            f"{Path(str(source_name)).name}；请确认机位和画面构图未变化。",
        ))

    outside = np.flatnonzero(
        (image_points[:, 0] < 0) | (image_points[:, 0] >= width) |
        (image_points[:, 1] < 0) | (image_points[:, 1] >= height)
    )
    if len(outside):
        named = "、".join(labels[index] for index in outside[:5])
        issues.append(CalibrationQualityIssue(
            "keypoint_outside_image",
            f"{len(outside)} 个关键点在 {width}×{height} 画面之外（{named}），请核对输入画面的尺寸。",
        ))

    indices = [int(item[0]) for item in keypoints]
    duplicate_labels = [
        str(pitch_config.labels[index]) for index in sorted(set(indices))
        if indices.count(index) > 1
    ]
    if duplicate_labels:
        issues.append(CalibrationQualityIssue(
            "duplicate_keypoint_id",
            f"关键点编号重复：{'、'.join(duplicate_labels[:5])}；一个场地位置应只对应一个画面标点。",
        ))

    if len(keypoints) >= 2:
        separations = np.linalg.norm(
            image_points[:, None, :] - image_points[None, :, :], axis=2
        )
        world_separations = np.linalg.norm(
            pitch_points[:, None, :] - pitch_points[None, :, :], axis=2
        )
        conflicts = np.argwhere(np.triu(
            (separations < 3.0) & (world_separations > 100.0), k=1
        ))
        if len(conflicts):
            first, second = conflicts[0]
            issues.append(CalibrationQualityIssue(
                "duplicate_pixel",
                f"{labels[first]} 与 {labels[second]} 等不同场地位置被标在几乎同一个画面位置；"
                "请核对点号。",
            ))

    if boundary_active:
        # The four marked boundary lines fit their own four intersections
        # exactly.  Only independently marked keypoints can test accuracy.
        boundary_source = field_boundary.get("source")
        same_source = not (
            keypoint_source_name and boundary_source and
            Path(str(keypoint_source_name)).name != Path(str(boundary_source)).name
        )
        if len(keypoints) >= 4 and same_source:
            issue = _residual_issue(
                pitch_points, image_points, matrix, image_size, labels,
                code="boundary_keypoint_mismatch", description="四边标定投影",
            )
            if issue:
                issues.append(issue)
        elif not same_source:
            issues.append(CalibrationQualityIssue(
                "boundary_crosscheck_unavailable",
                "四边标定与关键点取自不同画面，已跳过误差比较；请核对同一机位是否移动。",
            ))
        return issues

    if len(keypoints) < 6:
        issues.append(CalibrationQualityIssue(
            "few_keypoints",
            f"当前只有 {len(keypoints)} 个关键点；少量点可以被单应矩阵精确拟合，"
            "无法可靠发现点号对应错误。建议补标分布在场地不同位置的点。",
        ))
    if len(keypoints) < 4:
        return issues

    hull_area = float(cv2.contourArea(cv2.convexHull(pitch_points)))
    pitch_area = float(pitch_config.length) * float(pitch_config.width)
    if pitch_area > 0 and hull_area / pitch_area < 0.01:
        issues.append(CalibrationQualityIssue(
            "poor_coverage",
            "关键点过于集中或接近同一条场地线，映射到未标注区域可能明显偏移。",
        ))

    try:
        # Camera's legacy path uses an all-point least-squares homography.
        # This is a fast preview on raw pixels, not the final distortion-
        # corrected result.  Compare against that same fitting policy.
        matrix, _mask = cv2.findHomography(image_points, pitch_points, method=0)
    except cv2.error:
        matrix = None
    if matrix is None or not np.isfinite(matrix).all():
        issues.append(CalibrationQualityIssue(
            "unstable_homography", "关键点无法拟合稳定的场地映射，请检查点号和分布。",
        ))
        return issues

    issue = _residual_issue(
        pitch_points, image_points, matrix, image_size, labels,
        code="keypoint_mismatch", description="关键点拟合结果",
    )
    if issue:
        issues.append(issue)
    return issues
