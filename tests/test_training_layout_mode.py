import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from annotation_client import AnnotationClient
import yaml
from marker_group_model import (
    load_group_closed_modes,
    resize_group_closed_modes,
    sync_measurement_group_configs,
)


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class TrainingLayoutModeTests(unittest.TestCase):
    def test_old_config_defaults_to_line(self):
        self.assertEqual(load_group_closed_modes({}, 2), [False, False])
        self.assertEqual(
            load_group_closed_modes(
                {"GROUP_CONFIGS": [{"group": 1, "is_closed": True}]}, 3
            ),
            [False, True, False],
        )

    def test_group_resize_keeps_existing_modes(self):
        self.assertEqual(resize_group_closed_modes([True, False], 3), [True, False, False])
        self.assertEqual(resize_group_closed_modes([True, False], 1), [True])

    def test_save_modes_after_skipping_an_empty_group(self):
        data = {
            "GROUP_CONFIGS": [
                {"group": 0, "is_closed": False, "note": "keep"},
                {"group": 1, "is_closed": True},
                {"group": 2, "is_closed": False},
            ],
            "PAIR_CONFIGS": [{"group1": 0, "group2": 2, "type": "nearest"}],
        }
        sync_measurement_group_configs(
            data, count=2, source_indices=[0, 2],
            existing_source_indices=[0, 1, 2],
            closed_modes=[True, False, True],
        )
        self.assertEqual(
            data["GROUP_CONFIGS"],
            [
                {"group": 0, "is_closed": True, "note": "keep"},
                {"group": 1, "is_closed": True},
            ],
        )
        self.assertEqual(
            data["PAIR_CONFIGS"],
            [{"group1": 0, "group2": 1, "type": "nearest"}],
        )
        sync_measurement_group_configs(
            data, count=2, source_indices=[0, 2],
            existing_source_indices=[0, 2],
            closed_modes=[True, False, True],
        )
        self.assertEqual(load_group_closed_modes(data, 2), [True, True])

    def test_switching_groups_shows_and_keeps_their_own_modes(self):
        client = SimpleNamespace(
            marker_groups=[[], []],
            marker_equipment_types=["标志碟", "标志桶"],
            marker_group_closed=[False, True],
            marker_group_count_var=FakeVar(),
            marker_equipment_type_var=FakeVar(),
            marker_layout_var=FakeVar(),
            current_group=FakeVar(1),
            _active_marker_type_index=0,
            _refresh_list=lambda: None,
            status=FakeVar(),
        )
        AnnotationClient._sync_marker_group_controls(client)
        self.assertEqual(client.marker_layout_var.get(), "line")
        client.current_group.set(2)
        AnnotationClient._sync_marker_group_controls(client)
        self.assertEqual(client.marker_layout_var.get(), "ring")
        client.marker_layout_var.set("line")
        AnnotationClient._remember_marker_layout_mode(client)
        self.assertEqual(client.marker_group_closed, [False, False])
        client.current_group.set(1)
        AnnotationClient._sync_marker_group_controls(client)
        self.assertEqual(client.marker_layout_var.get(), "line")

    def test_client_saves_current_modes_to_measurement_config(self):
        with TemporaryDirectory() as directory:
            client = SimpleNamespace(
                project_dir=Path(directory),
                video_path_var=FakeVar("training.mp4"),
                camera_name="camera1",
                selected_pitch_config=FakeVar("QingYuanConfiguration"),
                store_root="store",
                store_base_root="store",
                isolated_calibration_session=False,
                media_kind="video",
                source_path=None,
                media_identity=None,
                marker_groups=[[(1, 1)], [], [(2, 2)]],
                marker_group_closed=[False, False, True],
                _config_source_indices=[0, 1, 2],
                _load_config=lambda: {
                    "GROUP_CONFIGS": [
                        {"group": 0, "is_closed": False},
                        {"group": 1, "is_closed": False},
                        {"group": 2, "is_closed": False},
                    ],
                },
            )
            AnnotationClient._write_config_yaml(client)
            data = yaml.safe_load((Path(directory) / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(load_group_closed_modes(data, 2), [False, True])
            self.assertEqual(client._config_source_indices, [0, 2])

    def test_sync_rejects_missing_mode(self):
        with self.assertRaisesRegex(ValueError, "训练形式"):
            sync_measurement_group_configs(
                {}, count=2, source_indices=[0, 1], closed_modes=[False]
            )


if __name__ == "__main__":
    unittest.main()
