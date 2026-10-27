"""Render a numbered pitch reference for the selected calibration layout.

This uses only the annotation client's required image dependencies.  The
measurement renderer also depends on optional ``supervision``; opening a
reference diagram should not require that package to be installed.
"""

from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image

import soccer


def _is_half_pitch(config: Any) -> bool:
    return isinstance(config, (
        soccer.HalfPitchConfiguration,
        soccer.HalfQingYuanConfiguration,
        soccer.HalfWuChuanConfiguration,
        soccer.HalfWuChuanYiZhongConfiguration,
    ))


def resolve_reference_configuration(
    selected_name: str, applied_config: Any
) -> Tuple[Any, bool]:
    """Return the selected layout and whether it still needs to be applied."""
    config_type = getattr(soccer, selected_name, None)
    if not selected_name.endswith("Configuration") or not isinstance(config_type, type):
        raise ValueError(f"未知场地配置：{selected_name}")
    if type(applied_config).__name__ == selected_name:
        return applied_config, False
    config = config_type()
    if not all(hasattr(config, name) for name in ("length", "width", "vertices", "edges", "labels")):
        raise ValueError(f"无法绘制场地配置：{selected_name}")
    return config, True


def orientation_hint(config: Any) -> str:
    if _is_half_pitch(config):
        if getattr(config, "is_right_half", False):
            return "右半场：中线在左、球门在右。边线名称按场地坐标，不按视频画面。"
        return "左半场：球门在左、中线在右。边线名称按场地坐标，不按视频画面。"
    return "全场：两端为球门。边线名称按场地坐标，不按视频画面。"


def render_numbered_pitch(config: Any, *, show_keypoints: bool = True) -> Image.Image:
    """Draw the configured pitch and its actual, numbered calibration points."""
    padding = 48
    scale = min(860.0 / config.length, 700.0 / config.width)
    field_length = int(config.length * scale)
    field_width = int(config.width * scale)
    image = np.empty((field_width + 2 * padding, field_length + 2 * padding, 3), dtype=np.uint8)
    image[:] = (34, 139, 34)  # BGR grass green

    is_half = _is_half_pitch(config)
    is_right = is_half and getattr(config, "is_right_half", False)
    # Draw the same base geometry as the left half, then rotate it.  Point
    # labels are drawn afterward in their actual right-half coordinates.
    vertices = list(config.vertices)
    base_vertices = (
        [(config.length - x, config.width - y) for x, y in vertices]
        if is_right else vertices
    )

    def pixel(point):
        return (int(point[0] * scale) + padding, int(point[1] * scale) + padding)

    for start, end in config.edges:
        cv2.line(image, pixel(base_vertices[start - 1]),
                 pixel(base_vertices[end - 1]), (255, 255, 255), 3, cv2.LINE_AA)

    centre_y = field_width // 2 + padding
    circle_radius = max(1, int(config.centre_circle_radius * scale))
    if is_half:
        cv2.ellipse(image, (field_length + padding, centre_y),
                    (circle_radius, circle_radius), 0, 90, 270,
                    (255, 255, 255), 3, cv2.LINE_AA)
    else:
        cv2.circle(image, (field_length // 2 + padding, centre_y),
                   circle_radius, (255, 255, 255), 3, cv2.LINE_AA)
    penalty_x = int(config.penalty_spot_distance * scale) + padding
    cv2.circle(image, (penalty_x, centre_y), 5, (255, 255, 255), -1, cv2.LINE_AA)
    if not is_half:
        cv2.circle(image, (field_length - penalty_x + 2 * padding, centre_y),
                   5, (255, 255, 255), -1, cv2.LINE_AA)
    if is_right:
        image = cv2.rotate(image, cv2.ROTATE_180)

    if show_keypoints:
        for point, label, color_hex in zip(vertices, config.labels, config.colors):
            x, y = pixel(point)
            small = label.startswith(("T", "B"))
            radius = 4 if small else 7
            font_scale = 0.33 if small else 0.48
            color = tuple(int(color_hex[index:index + 2], 16) for index in (5, 3, 1))
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            text_x = x + radius + 5
            if text_x + text_width + 4 >= image.shape[1]:
                text_x = x - radius - text_width - 5
            text_y = y + text_height // 2
            text_y = min(max(text_height + 3, text_y), image.shape[0] - baseline - 3)
            cv2.rectangle(image, (text_x - 3, text_y - text_height - 3),
                          (text_x + text_width + 3, text_y + baseline + 3),
                          (15, 35, 15), -1)
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
