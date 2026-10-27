"""Advisory-only calibration checks for the desktop annotation workflow."""

import unittest

import cv2
import numpy as np

from calibration_quality import inspect_calibration_quality


class DemoPitch:
    length = 10000
    width = 6000
    vertices = [
        (0, 0), (10000, 0), (10000, 6000), (0, 6000),
        (2500, 0), (5000, 0), (7500, 0),
        (2500, 6000), (5000, 6000), (7500, 6000),
        (2500, 3000), (7500, 3000),
    ]
    labels = [f"P{i + 1}" for i in range(len(vertices))]


def sample_annotations():
    pitch = DemoPitch()
    src = np.asarray([[0, 0], [10000, 0], [10000, 6000], [0, 6000]], np.float32)
    dst = np.asarray([[120, 100], [1120, 135], [1020, 690], [75, 680]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    pixels = cv2.perspectiveTransform(
        np.asarray(pitch.vertices, np.float32).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    return [(i, tuple(point)) for i, point in enumerate(pixels)]


def sample_boundary():
    return {
        "schema_version": 1,
        "camera_id": "camera1",
        "config_type": "DemoPitch",
        "pitch_size_cm": [10000, 6000],
        "image_size": [1280, 720],
        "source": "same.mp4",
        "enabled": True,
        "edges": {
            "top": {"points": [[120, 100], [1120, 135]]},
            "right": {"points": [[1120, 135], [1020, 690]]},
            "bottom": {"points": [[75, 680], [1020, 690]]},
            "left": {"points": [[120, 100], [75, 680]]},
        },
    }


def codes(issues):
    return {issue.code for issue in issues}


class CalibrationQualityTests(unittest.TestCase):
    def test_clean_projective_keypoints_have_no_warning(self):
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), sample_annotations(),
        )
        self.assertEqual(issues, [])

    def test_wrong_correspondences_warn_but_do_not_raise(self):
        points = sample_annotations()
        for index in (1, 2, 4, 5, 6):
            vertex, image = points[index]
            points[index] = (vertex, (image[0] - 320, image[1] + 210))
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), points,
        )
        self.assertIn("keypoint_mismatch", codes(issues))
        issue = next(issue for issue in issues if issue.code == "keypoint_mismatch")
        self.assertIn("像素", issue.message)

    def test_one_severely_misplaced_keypoint_also_warns(self):
        points = sample_annotations()
        vertex, image = points[4]
        points[4] = (vertex, (image[0] - 300, image[1] + 200))
        issues = inspect_calibration_quality(DemoPitch(), (1280, 720), points)
        self.assertIn("keypoint_mismatch", codes(issues))

    def test_four_exactly_fit_points_warn_about_lack_of_independent_check(self):
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), sample_annotations()[:4],
        )
        self.assertIn("few_keypoints", codes(issues))

    def test_collinear_annotations_warn_about_coverage(self):
        annotations = sample_annotations()
        selected = [annotations[index] for index in (0, 1, 4, 5, 6)]
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), selected,
        )
        self.assertIn("poor_coverage", codes(issues))

    def test_outside_and_overlapping_different_points_are_reported(self):
        annotations = sample_annotations()
        annotations[1] = (annotations[1][0], (-8, 50))
        annotations[2] = (annotations[2][0], annotations[0][1])
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), annotations,
        )
        self.assertIn("keypoint_outside_image", codes(issues))
        self.assertIn("duplicate_pixel", codes(issues))

    def test_boundary_not_judged_by_its_own_exact_four_corner_fit(self):
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), field_boundary=sample_boundary(),
            source_name="same.mp4",
        )
        self.assertEqual(issues, [])

    def test_legacy_estimated_boundary_does_not_require_confirmation(self):
        boundary = sample_boundary()
        boundary["edges"]["bottom"]["estimated"] = True
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), field_boundary=boundary,
        )
        self.assertNotIn("boundary_unverified", codes(issues))

    def test_boundary_checks_independent_keypoints_and_source(self):
        boundary = sample_boundary()
        annotations = sample_annotations()
        vertex, _image = annotations[0]
        annotations[0] = (vertex, (850, 650))
        vertex, _image = annotations[4]
        annotations[4] = (vertex, (700, 620))
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), annotations,
            field_boundary=boundary, source_name="other.mp4",
        )
        self.assertIn("boundary_source", codes(issues))
        self.assertIn("boundary_keypoint_mismatch", codes(issues))

    def test_boundary_skips_crosscheck_for_stale_keypoints(self):
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), sample_annotations(),
            field_boundary=sample_boundary(), source_name="same.mp4",
            keypoint_source_name="old.mp4",
        )
        self.assertIn("keypoint_source", codes(issues))
        self.assertNotIn("boundary_keypoint_mismatch", codes(issues))

    def test_disabled_boundary_checks_legacy_keypoints_instead(self):
        boundary = sample_boundary()
        boundary["enabled"] = False
        issues = inspect_calibration_quality(
            DemoPitch(), (1280, 720), sample_annotations()[:4],
            field_boundary=boundary,
        )
        self.assertIn("few_keypoints", codes(issues))


if __name__ == "__main__":
    unittest.main()
