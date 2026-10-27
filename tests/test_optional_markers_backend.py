"""Zero-equipment training must still produce the ordinary measurement outputs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import measure_marker_position as measurement_module
from marker_group_model import (
    build_marker_data,
    effective_marker_group_count,
    measurement_group_source_indices,
    resize_group_closed_modes,
    resize_marker_groups,
    sync_measurement_group_configs,
)
from soccer import SoccerPitchConfiguration


class OptionalMarkersBackendTests(unittest.TestCase):
    def test_planned_groups_can_resize_from_and_back_to_zero(self):
        self.assertEqual(resize_marker_groups([], [], 0), ([], []))
        self.assertEqual(resize_group_closed_modes([], 0), [])
        groups, equipment_types = resize_marker_groups([], [], 2)
        self.assertEqual(groups, [[], []])
        self.assertEqual(len(equipment_types), 2)
        self.assertEqual(resize_marker_groups(groups, equipment_types, 0), ([], []))
        self.assertEqual(resize_group_closed_modes([False, True], 0), [])

    def test_no_markers_are_a_valid_saved_state(self):
        data, incomplete = build_marker_data([[], []], ["标志碟", "标志桶"])
        self.assertEqual(data["groups"], [])
        self.assertEqual(incomplete, [])
        self.assertEqual(effective_marker_group_count([[], []]), 0)
        self.assertEqual(measurement_group_source_indices([[], []]), [])

    def test_partial_groups_still_report_skipped_planned_group(self):
        data, incomplete = build_marker_data([[], [(12, 34)]], ["标志碟", "标志桶"])
        self.assertEqual(incomplete, [1])
        self.assertEqual(data["groups"][0]["group_id"], 0)
        self.assertEqual(data["groups"][0]["equipment_type"], "标志桶")

    def test_zero_effective_groups_clear_previous_distance_settings(self):
        config = {
            "GROUP_CONFIGS": [{"group": 0, "is_closed": True}],
            "PAIR_CONFIGS": [{"group1": 0, "group2": 1, "type": "both"}],
        }
        sync_measurement_group_configs(
            config, count=0, source_indices=[], existing_source_indices=[0],
            closed_modes=[True],
        )
        self.assertEqual(config["GROUP_CONFIGS"], [])
        self.assertEqual(config["PAIR_CONFIGS"], [])

    def test_empty_saved_markers_auto_load_and_generate_base_pitch_outputs(self):
        with TemporaryDirectory() as temporary:
            store_root = Path(temporary) / "store"
            output = store_root / "manual" / "camera1"
            output.mkdir(parents=True)
            (output / "markers.json").write_text(
                json.dumps({"source": "markers", "groups": []}), encoding="utf-8"
            )
            camera = SimpleNamespace(
                name="camera1", config=SoccerPitchConfiguration(),
                pixel_to_pitch=lambda *_args: self.fail("empty markers were projected"),
            )
            config = (
                "unused.mp4", "camera1", "SoccerPitchConfiguration",
                [{"group": 0, "is_closed": True}],
                [{"group1": 0, "group2": 1, "type": "both"}],
                str(store_root), True,
            )
            with patch.object(measurement_module, "load_configure", return_value=config), \
                    patch.object(measurement_module, "Camera", return_value=camera), \
                    patch("builtins.input", side_effect=AssertionError("unexpected marker prompt")):
                measurement_module.main()

            coordinates = json.loads((output / "coordinates.json").read_text(encoding="utf-8"))
            report = (output / "measurement_report.txt").read_text(encoding="utf-8")
            self.assertEqual(coordinates["num_groups"], 0)
            self.assertEqual(coordinates["groups"], [])
            self.assertIn("本次无器材标注", report)
            self.assertTrue((output / "visualization.jpg").is_file())
            self.assertGreater((output / "visualization.jpg").stat().st_size, 0)

    def test_unreadable_marker_cache_exits_with_failure_status(self):
        with TemporaryDirectory() as temporary:
            store_root = Path(temporary) / "store"
            output = store_root / "manual" / "camera1"
            output.mkdir(parents=True)
            (output / "markers.json").write_text(
                json.dumps({"groups": [{"group_id": 0, "points": [
                    {"id": 0, "img_coord": None}
                ]}]}), encoding="utf-8"
            )
            config = (
                "unused.mp4", "camera1", "SoccerPitchConfiguration",
                [], [], str(store_root), True,
            )
            camera = SimpleNamespace(name="camera1", config=SoccerPitchConfiguration())
            with patch.object(measurement_module, "load_configure", return_value=config), \
                    patch.object(measurement_module, "Camera", return_value=camera), \
                    patch("builtins.input", return_value="y"):
                with self.assertRaises(SystemExit) as exit_status:
                    measurement_module.main()
            self.assertEqual(exit_status.exception.code, 1)
            self.assertFalse((output / "coordinates.json").exists())


if __name__ == "__main__":
    unittest.main()
