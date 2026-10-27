"""Pure data helpers for equipment groups in the annotation client."""

from __future__ import annotations

from typing import Any


DEFAULT_EQUIPMENT_TYPE = "未指定"


def load_group_closed_modes(data: dict[str, Any], count: int) -> list[bool]:
    """Read each group's line/ring setting; older configs default to line."""
    modes = [False] * count
    configs = data.get("GROUP_CONFIGS")
    if not isinstance(configs, list):
        return modes
    for item in configs:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("group"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            modes[index] = bool(item.get("is_closed", False))
    return modes


def resize_group_closed_modes(modes: list[bool], count: int) -> list[bool]:
    """Keep retained group settings and default new groups to line."""
    if not 0 <= count <= 99:
        raise ValueError("器材组数须为 0–99。")
    return [bool(value) for value in modes[:count]] + [False] * max(0, count - len(modes))


def normalize_equipment_type(value: Any) -> str:
    """Keep older marker files (without a type) readable."""
    return value.strip() if isinstance(value, str) and value.strip() else DEFAULT_EQUIPMENT_TYPE


def load_marker_groups(data: dict[str, Any]) -> tuple[list[list[tuple[int, int]]], list[str]]:
    """Read marker points and aligned equipment types from markers.json data."""
    groups: list[list[tuple[int, int]]] = []
    equipment_types: list[str] = []
    for group in sorted(data.get("groups", []), key=lambda item: int(item.get("group_id", 0))):
        groups.append([
            (int(point["img_coord"][0]), int(point["img_coord"][1]))
            for point in group.get("points", [])
        ])
        equipment_types.append(normalize_equipment_type(group.get("equipment_type")))
    return groups, equipment_types


def resize_marker_groups(
    groups: list[list[tuple[int, int]]], equipment_types: list[str], count: int
) -> tuple[list[list[tuple[int, int]]], list[str]]:
    """Resize planned groups while keeping each retained group's points and type."""
    if not 0 <= count <= 99:
        raise ValueError("器材组数须为 0–99。")
    resized_groups = [list(group) for group in groups[:count]]
    resized_types = [normalize_equipment_type(value) for value in equipment_types[:count]]
    while len(resized_groups) < count:
        resized_groups.append([])
    while len(resized_types) < count:
        resized_types.append(DEFAULT_EQUIPMENT_TYPE)
    return resized_groups, resized_types


def build_marker_data(
    groups: list[list[tuple[int, int]]], equipment_types: list[str]
) -> tuple[dict[str, Any], list[int]]:
    """Save annotated groups; no points means this training has no equipment."""
    saved_groups: list[dict[str, Any]] = []
    empty_group_numbers: list[int] = []
    for original_index, group in enumerate(groups):
        if not group:
            empty_group_numbers.append(original_index + 1)
            continue
        equipment_type = (
            equipment_types[original_index]
            if original_index < len(equipment_types) else DEFAULT_EQUIPMENT_TYPE
        )
        saved_groups.append({
            "group_id": len(saved_groups),
            "equipment_type": normalize_equipment_type(equipment_type),
            "points": [
                {"id": point_index, "img_coord": [int(x), int(y)]}
                for point_index, (x, y) in enumerate(group)
            ],
        })
    if not saved_groups:
        # Unannotated planning groups are optional and must not become
        # phantom measurement groups.
        empty_group_numbers = []
    return {"source": "markers", "type": "marker_annotation", "groups": saved_groups}, empty_group_numbers


def effective_marker_group_count(groups: list[list[tuple[int, int]]]) -> int:
    """Count only groups with markers; empty planning groups are optional."""
    return sum(bool(group) for group in groups)


def measurement_group_source_indices(groups: list[list[tuple[int, int]]]) -> list[int]:
    """Return original group indices in the same order as saved markers."""
    return [index for index, group in enumerate(groups) if group]


def sync_measurement_group_configs(
    data: dict[str, Any], count: int, source_indices: list[int] | None = None,
    existing_source_indices: list[int] | None = None,
    closed_modes: list[bool] | None = None,
) -> None:
    """Match config.yaml group indices to saved markers, preserving custom options."""
    if not 0 <= count <= 99:
        raise ValueError("有效器材组数须为 0–99。")
    if source_indices is None:
        source_indices = list(range(count))
    if len(source_indices) != count or len(set(source_indices)) != count:
        raise ValueError("器材组映射与有效组数不一致。")
    if closed_modes is not None and any(
        source_index < 0 or source_index >= len(closed_modes) for source_index in source_indices
    ):
        raise ValueError("器材组训练形式与有效组数不一致。")
    if count == 0:
        # Drop pairs from a previous training task as well as its group modes.
        data["GROUP_CONFIGS"] = []
        data["PAIR_CONFIGS"] = []
        return
    target_index = {original: current for current, original in enumerate(source_indices)}

    def old_source_index(config_index: int) -> int:
        if existing_source_indices is not None and 0 <= config_index < len(existing_source_indices):
            return existing_source_indices[config_index]
        return config_index

    original_configs = data.get("GROUP_CONFIGS")
    existing_groups: dict[int, dict[str, Any]] = {}
    if isinstance(original_configs, list):
        for item in original_configs:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("group"))
            except (TypeError, ValueError):
                continue
            source_index = old_source_index(index)
            if index >= 0 and source_index not in existing_groups:
                existing_groups[source_index] = dict(item)

    group_configs = []
    for source_index in source_indices:
        group_config = {
            **existing_groups.get(source_index, {"is_closed": False}),
            "group": target_index[source_index],
        }
        if closed_modes is not None:
            group_config["is_closed"] = bool(closed_modes[source_index])
        group_configs.append(group_config)
    data["GROUP_CONFIGS"] = group_configs

    original_pairs = data.get("PAIR_CONFIGS")
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    if isinstance(original_pairs, list):
        for item in original_pairs:
            if not isinstance(item, dict):
                continue
            try:
                first, second = int(item.get("group1")), int(item.get("group2"))
            except (TypeError, ValueError):
                continue
            first_source, second_source = old_source_index(first), old_source_index(second)
            if (first_source == second_source or first_source not in target_index
                    or second_source not in target_index):
                continue
            mapped_first, mapped_second = target_index[first_source], target_index[second_source]
            key = tuple(sorted((mapped_first, mapped_second)))
            if key not in seen:
                pairs.append({**item, "group1": mapped_first, "group2": mapped_second})
                seen.add(key)

    # Newly added groups get a default distance pair to the preceding group.
    # Existing pairs (and their custom type) remain untouched.
    for index, source_index in enumerate(source_indices[1:], start=1):
        preceding_source = source_indices[index - 1]
        if (source_index not in existing_groups or preceding_source not in existing_groups
                or not isinstance(original_pairs, list)):
            key = (index - 1, index)
            if key not in seen:
                pairs.append({"group1": index - 1, "group2": index, "type": "both"})
                seen.add(key)
    data["PAIR_CONFIGS"] = pairs
