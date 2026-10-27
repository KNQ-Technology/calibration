"""Right-half diagrams and calibration coordinates mirror the left half."""

import unittest

import cv2
import numpy as np
import supervision as sv

import soccer
from tools.pitch_annotation_tool import draw_pitch, draw_points_on_pitch


HALF_PAIRS = (
    (soccer.HalfPitchConfiguration, soccer.RightHalfPitchConfiguration),
    (soccer.HalfGaoWuConfiguration, soccer.RightHalfGaoWuConfiguration),
    (soccer.HalfEliteU12Configuration, soccer.RightHalfEliteU12Configuration),
    (soccer.HalfEliteU15Configuration, soccer.RightHalfEliteU15Configuration),
    (soccer.HalfPuGongYingU11Configuration, soccer.RightHalfPuGongYingU11Configuration),
)


class RightHalfPitchTests(unittest.TestCase):
    def test_right_half_vertices_are_rotated_without_changing_indices(self):
        for left_type, right_type in HALF_PAIRS:
            with self.subTest(config=right_type.__name__):
                left, right = left_type(), right_type()
                self.assertEqual((left.length, left.width), (right.length, right.width))
                self.assertEqual(left.edges, right.edges)
                self.assertEqual(left.labels, right.labels)
                self.assertEqual(
                    right.vertices,
                    [(left.length - x, left.width - y) for x, y in left.vertices],
                )

    def test_right_half_pitch_image_is_exact_180_degree_rotation(self):
        for left_type, right_type in HALF_PAIRS:
            with self.subTest(config=right_type.__name__):
                left = draw_pitch(left_type(), scale=0.05, padding=20,
                                  line_thickness=2, point_radius=3)
                right = draw_pitch(right_type(), scale=0.05, padding=20,
                                   line_thickness=2, point_radius=3)
                np.testing.assert_array_equal(right, cv2.rotate(left, cv2.ROTATE_180))

    def test_marker_overlay_uses_right_half_coordinates(self):
        config = soccer.RightHalfGaoWuConfiguration()
        point = np.array([config.vertices[8]], dtype=float)  # 点球点
        scale, padding = 0.1, 20
        image = draw_points_on_pitch(
            config, point, pitch=draw_pitch(config, scale=scale, padding=padding),
            scale=scale, padding=padding, show_numbers=False, radius=5, thickness=1,
        )
        pixel_x = int(point[0, 0] * scale) + padding
        pixel_y = int(point[0, 1] * scale) + padding
        self.assertGreater(point[0, 0], config.length / 2)
        np.testing.assert_array_equal(image[pixel_y, pixel_x], sv.Color.RED.as_bgr())


if __name__ == "__main__":
    unittest.main()
