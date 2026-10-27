"""The configuration picker is localized without changing persisted class names."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import soccer


SOURCE = Path(__file__).resolve().parents[1] / "annotation_client.py"


def _load_picker_code():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    labels = next(node.value for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "PITCH_CONFIG_LABELS"
                          for target in node.targets))
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "AnnotationClient")
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_get_pitch_config_names", "_on_pitch_config_selected"}]
    namespace = {"soccer": soccer, "List": list}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return ast.literal_eval(labels), namespace


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class PitchConfigLabelTests(unittest.TestCase):
    def test_picker_order_hides_legacy_venues_without_removing_their_configs(self):
        labels, methods = _load_picker_code()
        names = methods["_get_pitch_config_names"](None)
        self.assertEqual(names, [
            "GaoWuConfiguration", "HalfGaoWuConfiguration", "RightHalfGaoWuConfiguration",
            "EliteU12Configuration", "HalfEliteU12Configuration", "RightHalfEliteU12Configuration",
            "EliteU15Configuration", "HalfEliteU15Configuration", "RightHalfEliteU15Configuration",
            "PuGongYingU11Configuration", "HalfPuGongYingU11Configuration",
            "RightHalfPuGongYingU11Configuration",
            "SoccerPitchConfiguration", "HalfPitchConfiguration", "RightHalfPitchConfiguration",
        ])
        for legacy in (
            "QingYuanConfiguration", "HalfQingYuanConfiguration",
            "WuChuanConfiguration", "HalfWuChuanConfiguration",
            "WuChuanYiZhongConfiguration", "HalfWuChuanYiZhongConfiguration",
        ):
            self.assertNotIn(legacy, names)
            self.assertIn(legacy, labels)
            self.assertIsNotNone(getattr(soccer, legacy)())
        self.assertTrue(set(names).issubset(set(labels)))
        self.assertEqual(len(set(labels.values())), len(labels))
        self.assertTrue(all(any("\u4e00" <= character <= "\u9fff" for character in label)
                            for label in labels.values()))
        self.assertTrue(all(
            labels[name].endswith("右半场" if name.startswith("RightHalf") else "左半场")
            for name in names if "Half" in name
        ))
        self.assertTrue(all(labels[name].endswith("全场") for name in names if "Half" not in name))

    def test_user_selection_keeps_internal_class_name(self):
        labels, methods = _load_picker_code()
        label = labels["HalfEliteU15Configuration"]
        app = SimpleNamespace(
            selected_pitch_config_label=_Var(label),
            selected_pitch_config=_Var("QingYuanConfiguration"),
            pitch_config_label_to_name={display: name for name, display in labels.items()},
        )
        methods["_on_pitch_config_selected"](app)
        self.assertEqual(app.selected_pitch_config.get(), "HalfEliteU15Configuration")

    def test_right_half_selection_uses_new_class_name(self):
        labels, methods = _load_picker_code()
        label = labels["RightHalfEliteU15Configuration"]
        app = SimpleNamespace(
            selected_pitch_config_label=_Var(label),
            selected_pitch_config=_Var("HalfEliteU15Configuration"),
            pitch_config_label_to_name={display: name for name, display in labels.items()},
        )
        methods["_on_pitch_config_selected"](app)
        self.assertEqual(app.selected_pitch_config.get(), "RightHalfEliteU15Configuration")


if __name__ == "__main__":
    unittest.main()
