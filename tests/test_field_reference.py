"""The reference window follows the chosen pitch rather than static images."""

import ast
from pathlib import Path
import unittest

import numpy as np

import soccer
from field_reference import (
    orientation_hint,
    render_numbered_pitch,
    resolve_reference_configuration,
)


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"


class FieldReferenceTests(unittest.TestCase):
    def test_u15_right_selection_previews_right_layout_even_before_apply(self):
        applied = soccer.HalfEliteU15Configuration()
        config, not_applied = resolve_reference_configuration(
            "RightHalfEliteU15Configuration", applied)
        self.assertIsInstance(config, soccer.RightHalfEliteU15Configuration)
        self.assertTrue(not_applied)
        self.assertEqual(config.vertices[0], (config.length, config.width))
        self.assertIn("球门在右", orientation_hint(config))

        same_config, not_applied = resolve_reference_configuration(
            "HalfEliteU15Configuration", applied)
        self.assertIs(same_config, applied)
        self.assertFalse(not_applied)

    def test_right_half_reference_field_is_left_half_rotated(self):
        left = np.asarray(render_numbered_pitch(
            soccer.HalfEliteU15Configuration(), show_keypoints=False))
        right = np.asarray(render_numbered_pitch(
            soccer.RightHalfEliteU15Configuration(), show_keypoints=False))
        np.testing.assert_array_equal(right, np.rot90(left, 2))

    def test_right_half_first_number_is_at_right_bottom(self):
        config = soccer.RightHalfEliteU15Configuration()
        image = np.asarray(render_numbered_pitch(config))
        scale = min(860.0 / config.length, 700.0 / config.width)
        x = int(config.vertices[0][0] * scale) + 48
        y = int(config.vertices[0][1] * scale) + 48
        self.assertGreater(x, image.shape[1] / 2)
        self.assertGreater(y, image.shape[0] / 2)
        self.assertEqual(tuple(image[y, x]), (255, 20, 147))  # 01 点的粉色

    def test_hidden_legacy_layout_can_still_be_previewed(self):
        config, not_applied = resolve_reference_configuration(
            "HalfWuChuanConfiguration", soccer.HalfWuChuanConfiguration())
        self.assertFalse(not_applied)
        self.assertGreater(render_numbered_pitch(config).width, 0)

    def test_reference_button_no_longer_uses_wuchuan_static_image(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SOURCE))
        owner = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
        method = next(node for node in owner.body
                      if isinstance(node, ast.FunctionDef) and node.name == "open_field_reference")
        body = ast.get_source_segment(source, method)
        self.assertIn("self.selected_pitch_config.get()", body)
        self.assertIn("render_numbered_pitch(config)", body)
        self.assertNotIn("01_吴川半场.png", body)
        self.assertNotIn("_reference_images", body)


if __name__ == "__main__":
    unittest.main()
