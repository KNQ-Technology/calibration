"""
标定标注客户端（分步向导版）

按照标定业务流程组织标注步骤：

    1. 项目设置      -> config.yaml
    2. 四条边线标定  -> store/{CAMERA_NAME}/field_boundary.json（默认）
       关闭四边标定时，插入必选的关键点、横向直线、纵向直线标注
    3/6. 训练区域    -> store/{CAMERA_NAME}/train_area.json
    4/7. 器材分组    -> store/manual/{CAMERA_NAME}/markers.json
    5/8. 保存与测量  -> 运行 measure_marker_position.py

生成的 JSON 与主流程 (camera.py / de_distortion.py / measure_marker_position.py)
完全兼容；启动时自动加载已有标注缓存，保存时直接写入规范路径。

启动时自动检测操作系统（Windows / macOS / Linux）并准备环境：
缺少必需依赖时自动 pip 安装；测量流程依赖缺失时给出提示但不阻塞标注。
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# ======================================================================
# 环境自检与准备（必须在导入第三方库之前执行）
# ======================================================================

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 控制台编码保护：遇到当前代码页无法表示的字符时用占位符替代而不是崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

_ENV_WARNINGS: List[str] = []

_REQUIRED_PACKAGES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("yaml", "PyYAML"),
]
# 「保存并运行测量」子流程额外需要的依赖：缺失只警告，不阻塞标注
_MEASURE_PACKAGES = [
    ("scipy", "scipy"),
    ("supervision", "supervision"),
]


def _fatal(message: str) -> None:
    """无法继续时报告错误并退出（优先弹窗，其次控制台）。"""
    print(f"[环境检查] {message}", file=sys.stderr)
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        _root = _tk.Tk()
        _root.withdraw()
        _mb.showerror("环境检查失败", message)
        _root.destroy()
    except Exception:
        pass
    sys.exit(1)


def _check_tkinter() -> None:
    if importlib.util.find_spec("tkinter") is not None:
        return
    if IS_MACOS:
        hint = "macOS: 请使用 python.org 安装包（自带 Tk），或执行 brew install python-tk"
    elif IS_LINUX:
        hint = "Linux: 请执行 sudo apt install python3-tk（或对应发行版的命令）"
    else:
        hint = "Windows: 请重新安装官方 Python 并勾选 tcl/tk 组件"
    print(f"[环境检查] 缺少 tkinter，无法启动图形界面。{hint}", file=sys.stderr)
    sys.exit(1)


def _prepare_environment() -> None:
    """检测系统并补齐缺失依赖。"""
    print(f"[环境检查] 系统: {platform.system()} {platform.release()} · "
          f"Python {platform.python_version()} ({sys.executable})")
    _check_tkinter()

    missing = [(m, p) for m, p in _REQUIRED_PACKAGES if importlib.util.find_spec(m) is None]
    if missing:
        pkgs = [p for _, p in missing]
        print(f"[环境检查] 缺少必需依赖: {', '.join(pkgs)}，正在自动安装…")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
        except (subprocess.CalledProcessError, OSError) as exc:
            _fatal(
                f"自动安装依赖失败: {', '.join(pkgs)}\n\n"
                f"请手动执行:\n{sys.executable} -m pip install {' '.join(pkgs)}\n\n{exc}"
            )
        importlib.invalidate_caches()
        still = [p for m, p in missing if importlib.util.find_spec(m) is None]
        if still:
            _fatal(f"依赖安装后仍不可用: {', '.join(still)}，请检查 pip 环境。")
        print("[环境检查] 依赖安装完成。")

    lacking = [p for m, p in _MEASURE_PACKAGES if importlib.util.find_spec(m) is None]
    if lacking:
        _ENV_WARNINGS.append(
            "缺少测量流程依赖: " + ", ".join(lacking)
            + "。标注功能可用；「保存并运行测量」前请安装: pip install " + " ".join(lacking)
        )
        for warning in _ENV_WARNINGS:
            print(f"[环境检查] ⚠ {warning}")


_prepare_environment()

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import yaml

from marker_group_model import (
    DEFAULT_EQUIPMENT_TYPE,
    build_marker_data,
    effective_marker_group_count,
    load_group_closed_modes,
    load_marker_groups,
    normalize_equipment_type,
    resize_group_closed_modes,
    resize_marker_groups,
    measurement_group_source_indices,
    sync_measurement_group_configs,
)
from field_boundary import (
    compute_boundary_homography,
    load_boundary,
    validate_boundary_data,
)
from field_reference import (
    orientation_hint,
    render_numbered_pitch,
    resolve_reference_configuration,
)
from calibration_quality import inspect_calibration_quality
import soccer


def _save_annotation_file(path: Path, data: Dict[str, Any],
                          derived_cache_paths: Tuple[Path, ...] = ()) -> bool:
    """Save an annotation and invalidate its derived caches only if it changed.

    Cache files are removed before the new annotation is written: if removal fails,
    the old annotation remains in place instead of silently using stale results.
    Returns whether the serialized annotation differs from the saved file.
    """
    content = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    try:
        if path.read_bytes() == content:
            return False
    except FileNotFoundError:
        pass

    for cache_path in derived_cache_paths:
        cache_path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def _validate_saved_keypoints(data: Dict[str, Any], pitch_config: Any
                              ) -> List[Tuple[int, Tuple[int, int]]]:
    """Reject pixel points whose saved pitch coordinates belong to another preset."""
    if not isinstance(data, dict):
        raise ValueError("关键点文件格式不正确")
    expected_type = pitch_config.__class__.__name__
    saved_type = data.get("config_type")
    if saved_type != expected_type:
        raise ValueError(f"保存的场地配置为 {saved_type or '未记录'}，当前为 {expected_type}")

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("关键点列表格式不正确")
    vertices = pitch_config.vertices
    validated = []
    for item in annotations:
        try:
            index = int(item["vertex_index"])
            if not 0 <= index < len(vertices):
                raise ValueError(f"顶点编号 {index} 不在当前场地配置中")
            saved_world = item["world_coord"]
            if len(saved_world) != 2:
                raise ValueError(f"顶点 {index} 的世界坐标格式不正确")
            expected_world = vertices[index]
            if any(not math.isfinite(float(saved_world[axis])) or
                   abs(float(saved_world[axis]) - float(expected_world[axis])) > 1.0
                   for axis in (0, 1)):
                raise ValueError(f"顶点 {index} 的世界坐标与当前场地配置不一致")
            image = item["img_coord"]
            if len(image) != 2:
                raise ValueError(f"顶点 {index} 的图像坐标格式不正确")
            validated.append((index, (int(image[0]), int(image[1]))))
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError("关键点文件包含无效的顶点或坐标") from exc
    return validated

# ======================================================================
# 平台相关 UI 常量与辅助
# ======================================================================

if IS_WINDOWS:
    UI_FONT, MONO_FONT = "Microsoft YaHei UI", "Consolas"
elif IS_MACOS:
    UI_FONT, MONO_FONT = "PingFang SC", "Menlo"
else:
    UI_FONT, MONO_FONT = "Noto Sans CJK SC", "DejaVu Sans Mono"


def open_path_in_file_manager(path: str) -> None:
    """跨平台打开资源管理器 / 访达 / 文件管理器。"""
    if IS_WINDOWS:
        os.startfile(path)  # noqa: S606 - 本地工具
    elif IS_MACOS:
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def wheel_steps(event) -> int:
    """跨平台滚轮步数：Windows/Linux 的 delta 为 ±120 的倍数，macOS 为小整数。"""
    delta = getattr(event, "delta", 0)
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        return int(delta / 120)
    return 1 if delta > 0 else -1


Point = Tuple[int, int]
BOUNDARY_KEYS = ("top", "right", "bottom", "left")
# 只控制向导的展示顺序；JSON 字段及场地坐标映射仍按 BOUNDARY_KEYS 保存。
BOUNDARY_UI_ORDER = ("left", "top", "right", "bottom")
BOUNDARY_LABELS = {
    "top": "上边线",
    "right": "右端线",
    "bottom": "下边线",
    "left": "左端线",
}
BOUNDARY_COLORS = {
    "top": "#22c55e", "right": "#38bdf8",
    "bottom": "#f59e0b", "left": "#f472b6",
}

# 仅用于界面显示；config.yaml 和标定数据仍使用原有配置类名。
PITCH_CONFIG_LABELS = {
    "GaoWuConfiguration": "高屋小学 · 全场",
    "HalfGaoWuConfiguration": "高屋小学 · 左半场",
    "RightHalfGaoWuConfiguration": "高屋小学 · 右半场",
    "EliteU12Configuration": "精英队 U12 · 全场",
    "HalfEliteU12Configuration": "精英队 U12 · 左半场",
    "RightHalfEliteU12Configuration": "精英队 U12 · 右半场",
    "EliteU15Configuration": "精英队 U15 · 全场",
    "HalfEliteU15Configuration": "精英队 U15 · 左半场",
    "RightHalfEliteU15Configuration": "精英队 U15 · 右半场",
    "PuGongYingU11Configuration": "蒲公英 U11 · 全场",
    "HalfPuGongYingU11Configuration": "蒲公英 U11 · 左半场",
    "RightHalfPuGongYingU11Configuration": "蒲公英 U11 · 右半场",
    "QingYuanConfiguration": "清源小学 · 全场",
    "HalfQingYuanConfiguration": "清源小学 · 左半场",
    "WuChuanConfiguration": "吴川 · 全场",
    "HalfWuChuanConfiguration": "吴川 · 左半场",
    "WuChuanYiZhongConfiguration": "吴川一中 · 全场",
    "HalfWuChuanYiZhongConfiguration": "吴川一中 · 左半场",
    "SoccerPitchConfiguration": "标准球场 · 全场",
    "HalfPitchConfiguration": "标准球场 · 左半场",
    "RightHalfPitchConfiguration": "标准球场 · 右半场",
}


@dataclass
class KeypointAnnotation:
    vertex_index: int
    img_coord: Point


@dataclass
class StepDef:
    key: str
    title: str
    instructions: str
    optional: bool = False


STEP_DEFS: List[StepDef] = [
    StepDef(
        "setup",
        "项目设置",
        "确认视频文件、相机名称和场地配置。\n\n"
        "「应用并保存配置」会更新根目录 config.yaml，\n"
        "后续所有标注文件都会保存到\n"
        "store/{相机名}/ 与 store/manual/{相机名}/ 下。\n\n"
        "器材组数、每组器材类型在“器材分组”步骤设置。",
    ),
    StepDef(
        "field_boundary",
        "四条边线标定",
        "默认启用四边标定：按球场示意图依次选择左、上、右、下边线，"
        "每条在画面中点击两个位置。\n\n"
        "标完后请预览球场线投影并核对方向。"
        "关闭四边标定后，改用必选的关键点、横向直线和纵向直线标注。",
    ),
    StepDef(
        "keypoints",
        "关键点标注",
        "旧模式必选：在画面中标注球场关键点，用于计算单应矩阵。\n\n"
        "1. 在下拉框选择顶点编号（可点「查看场地图编号」对照）\n"
        "2. 左键在画面中点击该关键点的位置\n"
        "3. 标注后自动跳到下一个未标注的顶点\n\n"
        "至少需要 4 个点，点越多、分布越广，标定越准。\n"
        "重复标注同一编号会覆盖旧点。",
    ),
    StepDef(
        "manual_h",
        "横向直线标注",
        "旧模式必选：标注画面中真实世界为『横向直线』的线\n"
        "（如上下边线、禁区横线），用于畸变标定。\n\n"
        "每组对应一条直线：沿线均匀点 5 个以上点，\n"
        "然后「新建分组」标注下一条。\n\n"
        "建议至少标注 2 条分布在画面不同位置的横线。",
    ),
    StepDef(
        "manual_v",
        "纵向直线标注",
        "旧模式必选：标注画面中真实世界为『纵向直线』的线\n"
        "（如底线、中线、禁区纵线），用于畸变标定。\n\n"
        "每组对应一条直线：沿线均匀点 5 个以上点，\n"
        "然后「新建分组」标注下一条。\n\n"
        "建议至少标注 2 条分布在画面不同位置的纵线。",
    ),
    StepDef(
        "train_area",
        "训练区域",
        "定义球场上的训练区域（可选步骤）。\n\n"
        "步骤 A（可选）：左键在画面中点自定义点 V1、V2…\n"
        "（用于关键点之外的位置，如中场某处）\n\n"
        "步骤 B：在下方输入构成区域的点序列，\n"
        "可混用球场关键点标签和 V 点，如：1, 14, V1, V2\n"
        "点「生成区域」转换为球场坐标（四边模式用四边映射，旧模式用关键点）。\n\n"
        "区域至少 3 个点。跳过此步则由主流程交互标定。",
        optional=True,
    ),
    StepDef(
        "markers",
        "器材分组",
        "器材标注为选填。没有器材可直接进入“保存与测量”。\n\n"
        "需要测量器材距离时，设置组数、类型和训练形式，"
        "切换组后在画面中标点；未标点的空组会被忽略。\n\n"
        "线型只测量相邻点；环型还测量末点到首点。\n"
        "组间距离仍由 config.yaml 中的 PAIR_CONFIGS 决定。",
        optional=True,
    ),
    StepDef(
        "finish",
        "保存与测量",
        "检查各步骤完成情况，一键保存全部标注，\n"
        "并运行测量流程生成报告与可视化。\n\n"
        "注意：畸变参数由主流程自动计算；\n"
        "关键点、横纵线或边线内容变化并保存时会自动清理相关旧缓存。\n"
        "升级程序而标注未变化时，可手动清理派生缓存后重算。",
    ),
]

STEP_KEYS = [s.key for s in STEP_DEFS]
GROUP_STEPS = {"manual_h", "manual_v", "markers"}
LEGACY_STEP_KEYS = {"keypoints", "manual_h", "manual_v"}


class AnnotationClient(tk.Tk):
    """按标定业务流程组织的分步标注客户端。"""

    POINT_COLORS = [
        "#22c55e",
        "#ef4444",
        "#3b82f6",
        "#eab308",
        "#a855f7",
        "#06b6d4",
        "#f97316",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title(f"标定标注客户端 - 分步向导 · {platform.system()} / Python {platform.python_version()}")
        self.geometry("1420x860")
        self.minsize(1080, 680)

        self.project_dir = Path(__file__).resolve().parent
        self.config_data = self._load_config()
        self.pitch_config_names = self._get_pitch_config_names()
        initial_config = self.config_data.get("config") or self.pitch_config_names[0]
        self.selected_pitch_config = tk.StringVar(value=initial_config)
        self.pitch_config_label_to_name = {
            PITCH_CONFIG_LABELS.get(name, name): name for name in self.pitch_config_names
        }
        self.selected_pitch_config_label = tk.StringVar(
            value=PITCH_CONFIG_LABELS.get(initial_config, initial_config))
        self.pitch_config = self._load_pitch_config(initial_config)
        self.camera_name = self.config_data.get("CAMERA_NAME") or "camera1"

        self.status = tk.StringVar(value="就绪")
        self._keypoint_geometry_warning: Optional[str] = None
        self.frame_index = tk.IntVar(value=0)
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.source_path: Optional[Path] = None
        self.media_kind = self.config_data.get("annotation_media_kind", "video")
        self.media_identity = self.config_data.get("annotation_media_identity")
        self._measurement_running = False
        self._measurement_worker: Optional[threading.Thread] = None
        self._measurement_process: Optional[subprocess.Popen] = None
        self._measurement_cancel: Optional[threading.Event] = None
        self._measurement_result: Optional[Tuple[int, str, str]] = None
        self._measurement_run_id = 0
        self._measurement_restart_pending = False
        self._measurement_log_window = None
        self._measurement_log_text = None
        self.frame_bgr = self._load_default_frame()
        self.image_h, self.image_w = self.frame_bgr.shape[:2]

        # ---- 标注数据 ----
        self.keypoints: List[KeypointAnnotation] = []
        self.keypoint_source_name = ""
        # Only saved marker points establish an existing group.  Older configs
        # often contain a default group 0 even when no equipment was marked.
        self.marker_groups: List[List[Point]] = []
        self.marker_equipment_types: List[str] = []
        self.marker_group_closed: List[bool] = []
        self._config_source_indices: List[int] = []
        self.manual_h_groups: List[List[Point]] = [[]]
        self.manual_v_groups: List[List[Point]] = [[]]
        self.boundary_edges: Dict[str, List[Point]] = {key: [] for key in BOUNDARY_KEYS}
        self.boundary_edge_var = tk.StringVar(value=BOUNDARY_UI_ORDER[0])
        self._boundary_last_advanced_from: Optional[str] = None
        self.boundary_enabled_var = tk.BooleanVar(value=True)
        self.boundary_status_var = tk.StringVar(value="四条边线各标记两个点后可保存")
        self.boundary_preview_lines = False
        self.boundary_source_name = ""
        self.train_area_points: List[Point] = []       # 视频画面自定义点 V1..Vn
        self.train_area_region: List[List[float]] = []  # 球场坐标区域点 (cm)
        self.train_area_tokens = tk.StringVar(value="")
        self._saved_preview_paths: Dict[str, Path] = {}

        self.current_step_index = 0
        self.current_group = tk.IntVar(value=0)
        self.marker_group_count_var = tk.StringVar(value=str(len(self.marker_groups)))
        self.marker_equipment_type_var = tk.StringVar(value=DEFAULT_EQUIPMENT_TYPE)
        self.marker_layout_var = tk.StringVar(value="line")
        self._active_marker_type_index = -1
        self.selected_vertex = tk.IntVar(value=0)

        # 项目设置变量
        default_video = self._default_video_path()
        self.video_path_var = tk.StringVar(value=str(default_video) if default_video else "")
        self.camera_name_var = tk.StringVar(value=self.camera_name)
        self.store_root = self.config_data.get("store_dir") or "store"
        self.store_base_root = self.config_data.get("store_base_dir") or self.store_root
        self.isolated_calibration_session = bool(
            self.config_data.get("isolated_calibration_session", False))
        self.store_root_var = tk.StringVar(value=self.store_root)

        self._load_existing_annotations()
        self._build_ui()
        self._sync_media_controls()
        self.marker_equipment_type_var.trace_add("write", self._remember_marker_equipment_type)
        self._refresh_vertex_options()
        self._select_step(self._first_incomplete_step())
        self._fit_image()
        self._redraw()
        self._refresh_step_list()
        if _ENV_WARNINGS:
            self.status.set("⚠ " + _ENV_WARNINGS[0])
            messagebox.showwarning("环境提示", "\n\n".join(_ENV_WARNINGS))

    # ------------------------------------------------------------------
    # 配置与路径
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.project_dir / "config.yaml"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _get_pitch_config_names(self) -> List[str]:
        # Hide retired choices from the picker only. Their classes and saved
        # configuration names remain loadable for existing projects.
        hidden_names = {
            "QingYuanConfiguration", "HalfQingYuanConfiguration",
            "WuChuanConfiguration", "HalfWuChuanConfiguration",
            "WuChuanYiZhongConfiguration", "HalfWuChuanYiZhongConfiguration",
        }
        names = []
        for name in dir(soccer):
            config_cls = getattr(soccer, name)
            if (name.endswith("Configuration") and isinstance(config_cls, type)
                    and name not in hidden_names):
                names.append(name)
        preferred = [
            "GaoWuConfiguration",
            "HalfGaoWuConfiguration",
            "RightHalfGaoWuConfiguration",
            "EliteU12Configuration",
            "HalfEliteU12Configuration",
            "RightHalfEliteU12Configuration",
            "EliteU15Configuration",
            "HalfEliteU15Configuration",
            "RightHalfEliteU15Configuration",
            "PuGongYingU11Configuration",
            "HalfPuGongYingU11Configuration",
            "RightHalfPuGongYingU11Configuration",
            "SoccerPitchConfiguration",
            "HalfPitchConfiguration",
            "RightHalfPitchConfiguration",
        ]
        ordered = [name for name in preferred if name in names]
        ordered.extend(sorted(name for name in names if name not in ordered))
        return ordered

    def _on_pitch_config_selected(self, _event=None) -> None:
        label = self.selected_pitch_config_label.get()
        self.selected_pitch_config.set(self.pitch_config_label_to_name[label])

    def _load_pitch_config(self, class_name: str) -> Any:
        config_cls = getattr(soccer, class_name, soccer.QingYuanConfiguration)
        return config_cls()

    def _default_video_path(self) -> Optional[Path]:
        dir_path = self.config_data.get("dir_path") or "."
        video_name = self.config_data.get("video_name") or ""
        if not video_name:
            return None
        path = Path(dir_path) / video_name
        if not path.is_absolute():
            path = self.project_dir / path
        return path

    def _resolve_store_root(self) -> Path:
        raw = (self.store_root or "store").strip()
        path = Path(raw)
        if not path.is_absolute():
            path = self.project_dir / path
        return path

    def _store_dir(self) -> Path:
        return self._resolve_store_root() / self.camera_name

    def _manual_dir(self) -> Path:
        return self._resolve_store_root() / "manual" / self.camera_name

    def _step_save_path(self, key: str) -> Optional[Path]:
        return {
            "keypoints": self._store_dir() / "keypoints.json",
            "manual_h": self._store_dir() / "manual_horizontal.json",
            "manual_v": self._store_dir() / "manual_vertical.json",
            "field_boundary": self._store_dir() / "field_boundary.json",
            "train_area": self._store_dir() / "train_area.json",
            "markers": self._manual_dir() / "markers.json",
        }.get(key)

    # ------------------------------------------------------------------
    # 帧加载
    # ------------------------------------------------------------------

    def _load_default_frame(self):
        saved_source = self.config_data.get("annotation_media_path")
        source = Path(saved_source) if saved_source else self._default_video_path()
        self.source_path = source
        if source and self.media_kind == "image":
            frame = self._read_image(source)
        else:
            frame = self._read_video_frame(source, self.frame_index.get())
        if frame is not None:
            return frame
        blank = np.full((720, 1280, 3), (32, 24, 16), dtype=np.uint8)
        return blank

    def _read_video_frame(self, path: Path, frame_index: int):
        if not path or not Path(path).exists():
            return None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def _read_image(self, path: Path):
        if not path.exists():
            return None
        try:
            return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):
            return None

    @staticmethod
    def _media_file_identity(path: Path) -> Dict[str, Any]:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}

    def _sync_media_controls(self) -> None:
        state = ["disabled"] if self.media_kind == "image" else ["!disabled"]
        for widget in (getattr(self, "frame_index_entry", None),
                       getattr(self, "load_frame_button", None)):
            if widget is not None:
                widget.state(state)

    def _reset_annotation_state(self) -> None:
        """Clear all in-memory annotations before displaying a different source."""
        # Variable traces can refresh the current step while the reset runs.
        self.current_step_index = 0
        self.boundary_enabled_var.set(True)
        self.keypoints = []
        self.keypoint_source_name = ""
        self._keypoint_geometry_warning = None
        self.manual_h_groups = [[]]
        self.manual_v_groups = [[]]
        self.boundary_edges = {key: [] for key in BOUNDARY_KEYS}
        self._boundary_last_advanced_from = None
        self.boundary_source_name = ""
        self.boundary_preview_lines = False
        self.train_area_points = []
        self.train_area_region = []
        self.train_area_tokens.set("")
        self._saved_preview_paths.clear()
        self.marker_groups = []
        self.marker_equipment_types = []
        self.marker_group_closed = []
        self._config_source_indices = []
        self.marker_group_count_var.set("0")
        self.current_group.set(0)
        self._active_marker_type_index = -1
        self.marker_equipment_type_var.set(DEFAULT_EQUIPMENT_TYPE)
        self.marker_layout_var.set("line")
        if hasattr(self, "_sync_marker_group_controls"):
            self._sync_marker_group_controls()
        self.boundary_edge_var.set(BOUNDARY_UI_ORDER[0])
        self.boundary_status_var.set("四条边线各标记两个点后可保存")
        self.frame_index.set(0)
        self.selected_vertex.set(0)
        if hasattr(self, "vertex_combo") and self.vertex_combo["values"]:
            self.vertex_combo.current(0)

    def _activate_new_media(self, path: Path, frame, kind: str) -> bool:
        """Give a new image/video a blank, private output root; never delete old work."""
        if self._measurement_running:
            messagebox.showwarning("导入文件", "测量正在运行，请完成后再导入新文件。")
            return False
        path = Path(path).resolve()
        try:
            identity = self._media_file_identity(path)
        except OSError as exc:
            messagebox.showerror("导入文件", f"无法读取文件：\n{path}\n{exc}")
            return False
        if (identity == self.media_identity and kind == self.media_kind
                and self.source_path and self.source_path.resolve() == path):
            self.frame_index.set(0)
            self.set_frame(frame)
            self.status.set(f"已重新打开当前文件：{path.name}，标注保持不变")
            return True
        if self.source_path and not messagebox.askokcancel(
            "开始新标定任务",
            f"将导入新{('视频' if kind == 'video' else '图片')}：{path.name}\n\n"
            "当前画布和完成状态会清空；已保存的旧标定及报告会保留。"
            "尚未保存的修改无法恢复。\n\n确定继续吗？",
        ):
            return False
        configured_base = Path(self.store_base_root or "store")
        base = configured_base if configured_base.is_absolute() else self.project_dir / configured_base
        label = re.sub(r"[^\w-]+", "-", path.stem, flags=re.UNICODE).strip("-")[:32] or "media"
        session_name = f"{datetime.now():%Y%m%d-%H%M%S}-{label}-{uuid4().hex[:8]}"
        session_root = base / "_media_sessions" / session_name
        configured_session = configured_base / "_media_sessions" / session_name
        config = self._load_config()
        config["store_base_dir"] = str(configured_base)
        config["store_dir"] = str(configured_session)
        config["isolated_calibration_session"] = True
        config["annotation_media_path"] = str(path)
        config["annotation_media_kind"] = kind
        config["annotation_media_identity"] = identity
        config["dir_path"] = str(path.parent) if kind == "video" else ""
        config["video_name"] = path.name if kind == "video" else ""
        config["CAMERA_NAME"] = self.camera_name
        config["config"] = self.selected_pitch_config.get()
        config["GROUP_CONFIGS"] = []
        config["PAIR_CONFIGS"] = []
        modes = config.get("four_boundary_enabled_by_camera")
        modes = dict(modes) if isinstance(modes, dict) else {}
        modes[self.camera_name] = True
        config["four_boundary_enabled_by_camera"] = modes
        config_path = self.project_dir / "config.yaml"
        temp_path = config_path.with_name(f".{config_path.name}.{uuid4().hex}.tmp")
        try:
            session_root.mkdir(parents=True, exist_ok=False)
            if config_path.exists():
                (session_root / "previous_config.yaml").write_bytes(config_path.read_bytes())
            with temp_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
            os.replace(temp_path, config_path)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            messagebox.showerror("导入文件", f"无法创建新标定任务：\n{exc}")
            return False
        self.config_data = config
        self.store_root = str(configured_session)
        self.store_root_var.set(self.store_root)
        self.isolated_calibration_session = True
        self.media_kind = kind
        self.media_identity = identity
        self.source_path = path
        self.video_path_var.set(str(path) if kind == "video" else "")
        self._reset_annotation_state()
        self.set_frame(frame)
        if hasattr(self, "train_area_status"):
            self._refresh_train_area_status()
        self._sync_media_controls()
        self._refresh_store_dir_hint()
        self._select_step(0)
        self.status.set(f"新标定任务已建立：{path.name}；旧文件已保留")
        return True

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(self.project_dir),
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        frame = self._read_video_frame(Path(path), 0)
        if frame is None:
            messagebox.showerror("打开视频", "视频打开失败或无法读取首帧；当前标定未改变。")
            return
        self._activate_new_media(Path(path), frame, "video")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(self.project_dir),
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        frame = self._read_image(Path(path))
        if frame is None:
            messagebox.showerror("打开图片", "图片打开失败。")
            return
        self._activate_new_media(Path(path), frame, "image")

    def load_frame_at_index(self) -> None:
        if self.media_kind == "image":
            messagebox.showinfo("加载帧", "当前来源是图片，没有视频帧号。")
            return
        if not self.source_path:
            self.source_path = self._default_video_path()
        frame = self._read_video_frame(self.source_path, self.frame_index.get())
        if frame is None:
            messagebox.showerror("加载帧", "无法从视频加载该帧。")
            return
        self.set_frame(frame)
        self.status.set(f"已加载帧 {self.frame_index.get()}，来源：{self.source_path.name}")

    def set_frame(self, frame) -> None:
        old_size = (self.image_w, self.image_h)
        self.frame_bgr = frame
        self.image_h, self.image_w = frame.shape[:2]
        boundary_reset = old_size != (self.image_w, self.image_h) and any(
            self.boundary_edges.values())
        if boundary_reset:
            self.boundary_edges = {key: [] for key in BOUNDARY_KEYS}
            self.boundary_preview_lines = False
            self.boundary_status_var.set("画面尺寸变化，已清除内存中的边线点；原标定文件未改动")
        self._fit_image()
        self._redraw()
        if boundary_reset:
            self._refresh_step_list()
        if self.current_step.key == "field_boundary":
            self._refresh_boundary_status()

    # ------------------------------------------------------------------
    # 已有标注自动加载
    # ------------------------------------------------------------------

    def _load_existing_annotations(self) -> None:
        loaded = []
        self.keypoints = []
        self.keypoint_source_name = ""
        self.boundary_edges = {key: [] for key in BOUNDARY_KEYS}
        self.boundary_preview_lines = False
        self.boundary_source_name = ""
        saved_modes = self.config_data.get("four_boundary_enabled_by_camera", {})
        saved_mode = saved_modes.get(self.camera_name) if isinstance(saved_modes, dict) else None
        self.boundary_enabled_var.set(saved_mode if isinstance(saved_mode, bool) else True)
        self._keypoint_geometry_warning = None
        path = self._step_save_path("keypoints")
        if path and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.keypoints = [
                    KeypointAnnotation(index, image)
                    for index, image in _validate_saved_keypoints(data, self.pitch_config)
                ]
                self.keypoint_source_name = str(data.get("source") or "")
                loaded.append(f"关键点 {len(self.keypoints)} 个")
            except (OSError, ValueError, TypeError) as exc:
                self._keypoint_geometry_warning = f"已有关键点未加载：{exc}"
                messagebox.showwarning(
                    "场地配置与关键点不匹配",
                    f"{self._keypoint_geometry_warning}\n\n原文件保持不变：\n{path}\n\n"
                    "请核对场地配置后重新标注关键点；若要保留旧标注，请使用新的相机名称。",
                )

        for key, attr in (("manual_h", "manual_h_groups"),
                          ("manual_v", "manual_v_groups"),
                          ("markers", "marker_groups")):
            path = self._step_save_path(key)
            if path and path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if key == "markers":
                        groups, equipment_types = load_marker_groups(data)
                    else:
                        groups = []
                        for group in sorted(data.get("groups", []), key=lambda g: g.get("group_id", 0)):
                            pts = [(int(p["img_coord"][0]), int(p["img_coord"][1]))
                                   for p in group.get("points", [])]
                            groups.append(pts)
                    if groups:
                        setattr(self, attr, groups)
                        if key == "markers":
                            self.marker_equipment_types = equipment_types
                            self.marker_group_closed = load_group_closed_modes(
                                self.config_data, len(groups))
                            self._config_source_indices = list(range(len(groups)))
                            self.marker_group_count_var.set(str(len(groups)))
                            self.current_group.set(1)
                        loaded.append(f"{key} {len(groups)} 组")
                except Exception:
                    pass

        path = self._step_save_path("field_boundary")
        if path and path.exists():
            try:
                data = load_boundary(
                    path, self.camera_name, self.pitch_config,
                    (self.image_w, self.image_h),
                )
                for key in BOUNDARY_KEYS:
                    self.boundary_edges[key] = [
                        (int(x), int(y)) for x, y in data["edges"][key]["points"]
                    ]
                self.boundary_enabled_var.set(bool(data["enabled"]))
                self.boundary_source_name = str(data.get("source") or "")
                loaded.append("四条边线" if data["enabled"] else "四条边线已停用")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                messagebox.showwarning(
                    "边线标定未加载",
                    f"{exc}\n\n文件未改动：\n{path}\n"
                    "请核对当前相机、场地配置和画面尺寸后重新标定。",
                )

        path = self._step_save_path("train_area")
        if path and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.train_area_region = data.get("region_points", []) or []
                self.train_area_points = [
                    (int(x), int(y)) for x, y in data.get("image_points", []) or []
                ]
                self.train_area_tokens.set(data.get("region_tokens", ""))
                if self.train_area_region:
                    loaded.append(f"训练区域 {len(self.train_area_region)} 点")
            except Exception:
                pass

        if self._keypoint_geometry_warning:
            self.status.set("⚠ " + self._keypoint_geometry_warning)
        elif loaded:
            self.status.set("已自动加载缓存：" + "、".join(loaded))

    # ------------------------------------------------------------------
    # 步骤状态
    # ------------------------------------------------------------------

    def _step_complete(self, key: str) -> bool:
        if key == "setup":
            return bool(self.source_path and Path(self.source_path).exists())
        if key == "keypoints":
            return len(self.keypoints) >= 4
        if key == "manual_h":
            return any(len(g) >= 2 for g in self.manual_h_groups)
        if key == "manual_v":
            return any(len(g) >= 2 for g in self.manual_v_groups)
        if key == "field_boundary":
            return not self.boundary_enabled_var.get() or all(
                len(self.boundary_edges[name]) == 2 for name in BOUNDARY_KEYS)
        if key == "train_area":
            return len(self.train_area_region) >= 3
        if key == "markers":
            return any(bool(group) for group in self.marker_groups)
        if key == "finish":
            return (self._manual_dir() / "measurement_report.txt").exists()
        return False

    def _first_incomplete_step(self) -> int:
        steps = self._active_steps()
        for i, step in enumerate(steps):
            if step.key == "finish":
                return i
            if step.optional:
                continue
            if not self._step_complete(step.key):
                return i
        return len(steps) - 1

    def _active_steps(self) -> List[StepDef]:
        if self.boundary_enabled_var.get():
            return [step for step in STEP_DEFS if step.key not in LEGACY_STEP_KEYS]
        return STEP_DEFS

    def _select_step_by_key(self, key: str) -> None:
        for index, step in enumerate(self._active_steps()):
            if step.key == key:
                self._select_step(index)
                return
        raise ValueError(f"当前流程没有步骤：{key}")

    def _refresh_step_list(self) -> None:
        if not hasattr(self, "step_listbox"):
            return
        steps = self._active_steps()
        selection = min(self.current_step_index, len(steps) - 1)
        self.step_listbox.delete(0, tk.END)
        self.step_listbox.configure(height=len(steps))
        for i, step in enumerate(steps):
            done = self._step_complete(step.key)
            icon = "✔" if done else ("△" if step.optional else "○")
            self.step_listbox.insert(tk.END, f" {icon}  {i + 1}. {step.title}")
            color = "#16a34a" if done else ("#94a3b8" if step.optional else "#334155")
            self.step_listbox.itemconfigure(i, foreground=color)
        self.step_listbox.selection_clear(0, tk.END)
        self.step_listbox.selection_set(selection)
        self._refresh_cache_panel()

    def _refresh_cache_panel(self) -> None:
        if not hasattr(self, "cache_text"):
            return
        lines = []
        entries = [
            ("keypoints.json", self._step_save_path("keypoints")),
            ("manual_horizontal.json", self._step_save_path("manual_h")),
            ("manual_vertical.json", self._step_save_path("manual_v")),
            ("field_boundary.json", self._step_save_path("field_boundary")),
            ("distortion_params.json", self._store_dir() / "distortion_params.json"),
            ("train_area.json", self._step_save_path("train_area")),
            ("markers.json", self._step_save_path("markers")),
            ("camera_info.json", self._store_dir() / "camera_info.json"),
            ("measurement_report.txt", self._manual_dir() / "measurement_report.txt"),
        ]
        for name, path in entries:
            mark = "✔" if path and path.exists() else "—"
            lines.append(f"{mark} {name}")
        self.cache_text.configure(state="normal")
        self.cache_text.delete("1.0", tk.END)
        self.cache_text.insert(tk.END, "\n".join(lines))
        self.cache_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_toolbar()

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        # ---- 左栏：步骤列表 + 缓存状态 ----
        left = ttk.Frame(body, padding=10)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        body.add(left, weight=0)

        ttk.Label(left, text="标定流程", font=(UI_FONT, 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        self.step_listbox = tk.Listbox(
            left, height=len(self._active_steps()), exportselection=False,
            font=(UI_FONT, 10), activestyle="none",
        )
        self.step_listbox.grid(row=1, column=0, sticky="new")
        self.step_listbox.bind("<<ListboxSelect>>", self._on_step_listbox)

        nav = ttk.Frame(left)
        nav.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)
        ttk.Button(nav, text="◀ 上一步", command=self.prev_step).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(nav, text="下一步 ▶", command=self.next_step).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        cache_box = ttk.LabelFrame(left, text="缓存文件状态", padding=8)
        cache_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        cache_box.columnconfigure(0, weight=1)
        self.cache_text = tk.Text(
            cache_box, height=9, width=28, relief="flat",
            font=(MONO_FONT, 9), state="disabled", background=self.cget("background"),
        )
        self.cache_text.grid(row=0, column=0, sticky="ew")
        ttk.Button(cache_box, text="刷新状态", command=self._refresh_step_list).grid(
            row=1, column=0, sticky="ew", pady=(6, 0))

        # ---- 中栏：画布 ----
        canvas_frame = ttk.Frame(body)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        body.add(canvas_frame, weight=4)

        self.canvas = tk.Canvas(canvas_frame, background="#0f172a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        if IS_LINUX:
            self.canvas.bind("<Button-4>", lambda _e: self.zoom(1.1))
            self.canvas.bind("<Button-5>", lambda _e: self.zoom(1 / 1.1))
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self._pan_start: Optional[Tuple[int, int]] = None

        # ---- 右栏：当前步骤面板 ----
        right_container = ttk.Frame(body)
        right_container.rowconfigure(0, weight=1)
        right_container.columnconfigure(0, weight=1)
        body.add(right_container, weight=0)

        right_canvas = tk.Canvas(right_container, highlightthickness=0, width=330)
        right_scroll = ttk.Scrollbar(right_container, orient=tk.VERTICAL, command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scroll.grid(row=0, column=1, sticky="ns")

        self.side = ttk.Frame(right_canvas, padding=12)
        self.side.columnconfigure(0, weight=1)
        side_window = right_canvas.create_window((0, 0), window=self.side, anchor="nw")

        def update_scrollregion(_event=None) -> None:
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))

        def resize_content(event) -> None:
            right_canvas.itemconfigure(side_window, width=event.width)
            update_scrollregion()

        def enable_wheel(_event=None) -> None:
            right_canvas.bind_all("<MouseWheel>",
                                  lambda e: right_canvas.yview_scroll(-wheel_steps(e), "units"))
            if IS_LINUX:
                right_canvas.bind_all("<Button-4>", lambda _e: right_canvas.yview_scroll(-1, "units"))
                right_canvas.bind_all("<Button-5>", lambda _e: right_canvas.yview_scroll(1, "units"))

        def disable_wheel(_event=None) -> None:
            right_canvas.unbind_all("<MouseWheel>")
            if IS_LINUX:
                right_canvas.unbind_all("<Button-4>")
                right_canvas.unbind_all("<Button-5>")

        self.side.bind("<Configure>", update_scrollregion)
        right_canvas.bind("<Configure>", resize_content)
        right_canvas.bind("<Enter>", enable_wheel)
        right_canvas.bind("<Leave>", disable_wheel)
        self.side.bind("<Enter>", enable_wheel)
        self.side.bind("<Leave>", disable_wheel)

        self._build_side_panels()

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        ttk.Button(toolbar, text="打开视频", command=self.open_video).grid(row=0, column=0, padx=4)
        ttk.Button(toolbar, text="打开图片", command=self.open_image).grid(row=0, column=1, padx=4)
        ttk.Label(toolbar, text="帧号").grid(row=0, column=2, padx=(14, 4))
        self.frame_index_entry = ttk.Entry(toolbar, textvariable=self.frame_index, width=8)
        self.frame_index_entry.grid(row=0, column=3, padx=4)
        self.load_frame_button = ttk.Button(toolbar, text="加载帧", command=self.load_frame_at_index)
        self.load_frame_button.grid(row=0, column=4, padx=4)
        ttk.Button(toolbar, text="适应窗口", command=self.fit_and_redraw).grid(row=0, column=5, padx=(14, 4))
        ttk.Button(toolbar, text="放大", command=lambda: self.zoom(1.2)).grid(row=0, column=6, padx=4)
        ttk.Button(toolbar, text="缩小", command=lambda: self.zoom(1 / 1.2)).grid(row=0, column=7, padx=4)
        ttk.Label(toolbar, textvariable=self.status).grid(row=0, column=8, sticky="e", padx=10)

    def _build_side_panels(self) -> None:
        # 步骤标题与说明
        self.step_title_label = ttk.Label(self.side, text="", font=(UI_FONT, 12, "bold"))
        self.step_title_label.grid(row=0, column=0, sticky="w")
        self.step_help_label = ttk.Label(self.side, text="", justify="left",
                                         foreground="#475569", wraplength=300)
        self.step_help_label.grid(row=1, column=0, sticky="ew", pady=(4, 10))

        # 每个步骤的专属面板（同一格子，按需显示）
        self.step_panels: Dict[str, ttk.Frame] = {}
        for key in STEP_KEYS:
            panel = ttk.Frame(self.side)
            panel.columnconfigure(0, weight=1)
            panel.grid(row=2, column=0, sticky="ew")
            panel.grid_remove()
            self.step_panels[key] = panel

        self._build_setup_panel(self.step_panels["setup"])
        self._build_keypoints_panel(self.step_panels["keypoints"])
        self._build_groups_panel(self.step_panels["manual_h"], "横线")
        self._build_groups_panel(self.step_panels["manual_v"], "纵线")
        self._build_boundary_panel(self.step_panels["field_boundary"])
        self._build_train_area_panel(self.step_panels["train_area"])
        self._build_groups_panel(self.step_panels["markers"], "标志物")
        self._build_finish_panel(self.step_panels["finish"])

        # 通用操作
        self.common_actions = ttk.LabelFrame(self.side, text="操作", padding=8)
        self.common_actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for i in range(2):
            self.common_actions.columnconfigure(i, weight=1)
        self.undo_button = ttk.Button(self.common_actions, text="撤销", command=self.undo)
        self.undo_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self.clear_step_button = ttk.Button(self.common_actions, text="清空本步", command=self.clear_current)
        self.clear_step_button.grid(
            row=0, column=1, sticky="ew", padx=(4, 0))
        self.save_step_button = ttk.Button(self.common_actions, text="保存本步",
                                           command=self.save_current_step)
        self.save_step_button.grid(row=1, column=0, sticky="ew", pady=(8, 0), padx=(0, 4))
        self.preview_step_button = ttk.Button(self.common_actions, text="生成预览图", command=self.save_preview)
        self.preview_step_button.grid(
            row=1, column=1, sticky="ew", pady=(8, 0), padx=(4, 0))
        self.open_step_result_button = ttk.Button(
            self.common_actions, text="打开结果目录",
            command=lambda: self.open_step_result_dir(self.current_step.key))
        self.open_step_result_button.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.open_step_result_button.grid_remove()

        # 标注列表
        list_box = ttk.LabelFrame(self.side, text="标注列表", padding=8)
        list_box.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        self.annotation_list = tk.Listbox(list_box, height=6)
        self.annotation_list.pack(fill="both", expand=True)

        hint = ttk.Label(
            self.side,
            text="左键：添加点\n右键/中键拖动：平移\n滚轮：缩放画面",
            justify="left",
            foreground="#94a3b8",
        )
        hint.grid(row=5, column=0, sticky="ew", pady=(8, 0))

    def _build_setup_panel(self, panel: ttk.Frame) -> None:
        box = ttk.LabelFrame(panel, text="项目配置", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)

        ttk.Label(box, text="视频文件（输入）").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.video_path_var).grid(row=1, column=0, sticky="ew")
        ttk.Button(box, text="浏览视频…", command=self.browse_setup_video).grid(row=2, column=0, sticky="ew", pady=(4, 8))

        ttk.Label(box, text="输出目录（标定与测量结果根目录）").grid(row=3, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.store_root_var).grid(row=4, column=0, sticky="ew")
        ttk.Button(box, text="浏览输出目录…", command=self.browse_store_root).grid(
            row=5, column=0, sticky="ew", pady=(4, 0))
        self.store_dir_hint = ttk.Label(box, text="", foreground="#475569",
                                        wraplength=280, justify="left")
        self.store_dir_hint.grid(row=6, column=0, sticky="w", pady=(2, 8))

        ttk.Label(box, text="相机名称（决定存储子目录）").grid(row=7, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.camera_name_var).grid(row=8, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(box, text="场地配置").grid(row=9, column=0, sticky="w")
        self.pitch_config_combo = ttk.Combobox(
            box,
            textvariable=self.selected_pitch_config_label,
            values=[PITCH_CONFIG_LABELS.get(name, name) for name in self.pitch_config_names],
            state="readonly",
        )
        self.pitch_config_combo.grid(row=10, column=0, sticky="ew", pady=(0, 8))
        self.pitch_config_combo.bind("<<ComboboxSelected>>", self._on_pitch_config_selected)

        ttk.Button(box, text="应用并保存配置 (config.yaml)",
                   command=self.apply_setup).grid(row=11, column=0, sticky="ew")
        ttk.Label(box, text="器材组数、类型和训练形式请在“器材分组”步骤设置。",
                  foreground="#475569", wraplength=280, justify="left").grid(
                      row=12, column=0, sticky="w", pady=(12, 4))
        ttk.Button(box, text="前往器材分组",
                   command=lambda: self._select_step_by_key("markers")).grid(
                       row=13, column=0, sticky="ew")
        self._refresh_store_dir_hint()

    def browse_store_root(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self._resolve_store_root().parent
                                                      if self._resolve_store_root().exists()
                                                      else self.project_dir))
        if path:
            self.store_root_var.set(path)

    def browse_setup_video(self) -> None:
        """In image mode, choose a measurement video without replacing its frame."""
        if self.media_kind != "image":
            self.open_video()
            return
        path = filedialog.askopenfilename(
            initialdir=str(self.project_dir),
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        frame = self._read_video_frame(Path(path), 0)
        if frame is None:
            messagebox.showerror("选择视频", "视频打开失败或无法读取首帧。")
            return
        if frame.shape[:2] != (self.image_h, self.image_w):
            messagebox.showwarning(
                "选择视频", "视频与当前图片的分辨率不同，标注坐标不能直接使用。"
                "请导入同分辨率的视频，或改用“打开视频”开始新任务。")
            return
        self.video_path_var.set(str(Path(path).resolve()))
        self.status.set(f"已关联测量视频：{Path(path).name}；请应用并保存配置")

    def _refresh_store_dir_hint(self) -> None:
        if hasattr(self, "store_dir_hint"):
            self.store_dir_hint.configure(
                text=f"标定参数 → {self._store_dir()}\n测量结果 → {self._manual_dir()}")

    def _build_keypoints_panel(self, panel: ttk.Frame) -> None:
        box = ttk.LabelFrame(panel, text="当前顶点", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.vertex_combo = ttk.Combobox(box, state="readonly", width=28)
        self.vertex_combo.grid(row=0, column=0, sticky="ew")
        self.vertex_combo.bind("<<ComboboxSelected>>", self.on_vertex_selected)
        self.keypoint_progress = ttk.Label(box, text="")
        self.keypoint_progress.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(box, text="下一个未标注顶点", command=self.next_unannotated_vertex).grid(
            row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(box, text="查看场地图编号", command=self.open_field_reference).grid(
            row=3, column=0, sticky="ew", pady=(6, 0))

        map_box = ttk.LabelFrame(panel, text="场地示意图（红圈=当前顶点，可点击选择）", padding=8)
        map_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.minimap_canvas = tk.Canvas(map_box, width=300, height=230,
                                        background="#14532d", highlightthickness=0)
        self.minimap_canvas.grid(row=0, column=0)
        self.minimap_canvas.bind("<Button-1>", self._on_minimap_click)
        minimap_legend = ttk.Label(
            map_box,
            text="绿点=已标注   灰点=未标注",
            foreground="#475569",
        )
        minimap_legend.grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _build_groups_panel(self, panel: ttk.Frame, noun: str) -> None:
        if noun == "标志物":
            self._build_marker_groups_panel(panel)
            return
        box = ttk.LabelFrame(panel, text=f"{noun}分组", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        row = ttk.Frame(box)
        row.grid(row=0, column=0, sticky="ew")
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="当前组").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(row, from_=1, to=99, textvariable=self.current_group,
                    width=6, command=self.ensure_group).grid(row=0, column=1, sticky="e")
        ttk.Button(box, text=f"新建分组（下一条{noun}）", command=self.new_group).grid(
            row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_boundary_panel(self, panel: ttk.Frame) -> None:
        box = ttk.LabelFrame(panel, text="四条场地边线", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            box, text="启用四边标定",
            variable=self.boundary_enabled_var,
            command=self._on_boundary_mode_change,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.boundary_edit_widgets: List[ttk.Widget] = []
        self.boundary_count_labels: Dict[str, ttk.Label] = {}
        for row_index, key in enumerate(BOUNDARY_UI_ORDER):
            row = ttk.Frame(box)
            row.grid(row=row_index + 1, column=0, sticky="ew", pady=(0, 6))
            row.columnconfigure(0, weight=1)
            edge_radio = ttk.Radiobutton(
                row, text=BOUNDARY_LABELS[key], value=key,
                variable=self.boundary_edge_var,
                command=self._on_boundary_edge_selected,
            )
            edge_radio.grid(row=0, column=0, sticky="w")
            self.boundary_edit_widgets.append(edge_radio)
            count_label = ttk.Label(row, text="0/2")
            count_label.grid(row=0, column=1, padx=(4, 0))
            self.boundary_count_labels[key] = count_label

        for row, label, action in (
            (5, "清除选中边线", self.clear_selected_boundary_edge),
            (6, "检查并预览四边范围", self.preview_boundary),
            (7, "查看场地图编号与方向", self.open_field_reference),
        ):
            button = ttk.Button(box, text=label, command=action)
            button.grid(row=row, column=0, sticky="ew", pady=(6, 0))
            self.boundary_edit_widgets.append(button)
        ttk.Label(
            box, textvariable=self.boundary_status_var, wraplength=285,
            justify="left", foreground="#475569",
        ).grid(row=8, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            panel,
            text="边线名称按场地坐标命名，不按画面上下左右命名。"
                 "黄色虚线是按四边投影的球场线，"
                 "请对照可见中线、禁区线核对方向与位置。",
            wraplength=300, justify="left", foreground="#475569",
        ).grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self._sync_boundary_controls()
        self._refresh_boundary_status()

    def _sync_boundary_controls(self) -> None:
        enabled = self.boundary_enabled_var.get()
        for widget in getattr(self, "boundary_edit_widgets", []):
            widget.state(["!disabled"] if enabled else ["disabled"])
        if self.current_step.key == "field_boundary":
            for widget_name in ("undo_button", "clear_step_button", "preview_step_button"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.state(["!disabled"] if enabled else ["disabled"])
            if hasattr(self, "save_step_button"):
                self.save_step_button.configure(
                    text="保存本步" if enabled else "保存旧模式")

    def _on_boundary_mode_change(self) -> None:
        self._boundary_last_advanced_from = None
        self.boundary_preview_lines = False
        self._sync_boundary_controls()
        self._refresh_boundary_status()
        self._refresh_step_list()
        self._refresh_finish_summary()
        self._redraw()

    def _on_boundary_edge_selected(self) -> None:
        """A manual selection overrides the last automatic advance."""
        self._boundary_last_advanced_from = None
        self._refresh_boundary_status()

    def _build_marker_groups_panel(self, panel: ttk.Frame) -> None:
        box = ttk.LabelFrame(panel, text="器材分组设置", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)

        ttk.Label(box, text="器材选填；无器材可直接保存并测量。",
                  foreground="#475569", wraplength=280, justify="left").grid(
                      row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(box, text="器材组数（0–99 组）").grid(
            row=1, column=0, sticky="w")
        count_row = ttk.Frame(box)
        count_row.grid(row=2, column=0, sticky="ew", pady=(3, 10))
        count_row.columnconfigure(0, weight=1)
        count_spin = ttk.Spinbox(count_row, from_=0, to=99,
                                 textvariable=self.marker_group_count_var, width=8)
        count_spin.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        count_spin.bind("<Return>", lambda _event: self.apply_marker_group_count())
        ttk.Button(count_row, text="应用组数", command=self.apply_marker_group_count).grid(
            row=0, column=1, sticky="e")

        ttk.Label(box, text="当前标注组").grid(row=3, column=0, sticky="w")
        self.marker_current_group_spin = ttk.Spinbox(
            box, from_=1, to=99, textvariable=self.current_group, width=8,
            command=self.select_marker_group,
        )
        self.marker_current_group_spin.grid(row=4, column=0, sticky="ew", pady=(3, 10))
        self.marker_current_group_spin.bind("<Return>", lambda _event: self.select_marker_group())
        self.marker_current_group_spin.bind("<FocusOut>", lambda _event: self.select_marker_group())

        ttk.Label(box, text="当前组器材类型（可选择或自行输入）").grid(
            row=5, column=0, sticky="w")
        self.marker_equipment_type_combo = ttk.Combobox(
            box,
            textvariable=self.marker_equipment_type_var,
            values=("未指定", "标志桶", "标志碟", "标志杆", "跨栏", "小球门", "敏捷梯", "其他"),
            state="normal",
        )
        self.marker_equipment_type_combo.grid(row=6, column=0, sticky="ew", pady=(3, 8))

        ttk.Label(box, text="当前组训练形式").grid(row=7, column=0, sticky="w")
        mode_row = ttk.Frame(box)
        mode_row.grid(row=8, column=0, sticky="ew", pady=(3, 4))
        mode_row.columnconfigure(0, weight=1)
        mode_row.columnconfigure(1, weight=1)
        line_button = ttk.Radiobutton(
            mode_row, text="线型训练", variable=self.marker_layout_var,
            value="line", command=self._remember_marker_layout_mode,
        )
        line_button.grid(row=0, column=0, sticky="w")
        ring_button = ttk.Radiobutton(
            mode_row, text="环型训练", variable=self.marker_layout_var,
            value="ring", command=self._remember_marker_layout_mode,
        )
        ring_button.grid(row=0, column=1, sticky="w")
        self.marker_layout_buttons = (line_button, ring_button)
        ttk.Label(
            box, text="环型会额外测量最后一点到第一点，建议至少标注 3 个点。",
            foreground="#475569", wraplength=280, justify="left",
        ).grid(row=9, column=0, sticky="w", pady=(0, 8))

        ttk.Button(box, text="新增一组器材", command=self.new_group).grid(
            row=10, column=0, sticky="ew")
        ttk.Label(box, text="只保存已标点的组；空组不参与测量。",
                  foreground="#475569", wraplength=280, justify="left").grid(
                      row=11, column=0, sticky="w", pady=(8, 0))
        self._sync_marker_group_controls()

    def _build_train_area_panel(self, panel: ttk.Frame) -> None:
        box = ttk.LabelFrame(panel, text="区域构成", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        ttk.Label(box, text="点序列（关键点标签 / V点，逗号分隔）",
                  wraplength=280, justify="left").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.train_area_tokens).grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(box, text="生成区域（转换为球场坐标）",
                   command=self.generate_train_area_region).grid(row=2, column=0, sticky="ew")
        self.train_area_status = ttk.Label(box, text="区域：未生成", foreground="#475569",
                                           wraplength=280, justify="left")
        self.train_area_status.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self._refresh_train_area_status()

    def _build_finish_panel(self, panel: ttk.Frame) -> None:
        box = ttk.LabelFrame(panel, text="完成情况", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.finish_summary = ttk.Label(box, text="", justify="left", font=(UI_FONT, 9))
        self.finish_summary.grid(row=0, column=0, sticky="w")

        actions = ttk.LabelFrame(panel, text="执行", padding=8)
        actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="保存全部标注", command=self.save_all_default).grid(
            row=0, column=0, sticky="ew")
        ttk.Button(actions, text="清理派生缓存（畸变参数等）",
                   command=self.clear_derived_cache).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(actions, text="保存并运行测量", command=self.save_and_run_measurement).grid(
            row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(actions, text="打开结果目录", command=self.open_output_dir).grid(
            row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(actions, text="打开器材测量目录", command=self.open_measurement_dir).grid(
            row=4, column=0, sticky="ew", pady=(6, 0))

    # ------------------------------------------------------------------
    # 步骤切换
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> StepDef:
        return self._active_steps()[self.current_step_index]

    def _on_step_listbox(self, _event=None) -> None:
        selection = self.step_listbox.curselection()
        if selection:
            self._select_step(selection[0])

    def prev_step(self) -> None:
        self._select_step(max(0, self.current_step_index - 1))

    def next_step(self) -> None:
        self._select_step(min(len(self._active_steps()) - 1, self.current_step_index + 1))

    def _select_step(self, index: int) -> None:
        self.current_step_index = max(0, min(index, len(self._active_steps()) - 1))
        step = self.current_step
        for key, panel in self.step_panels.items():
            if key == step.key:
                panel.grid()
            else:
                panel.grid_remove()
        self.step_title_label.configure(
            text=f"第 {self.current_step_index + 1} 步 · {step.title}" + ("（可选）" if step.optional else ""))
        self.step_help_label.configure(text=step.instructions)
        if hasattr(self, "save_step_button"):
            savable = step.key not in ("setup", "finish")
            self.save_step_button.state(["!disabled"] if savable else ["disabled"])
            self.save_step_button.configure(text="保存本步")
        for widget_name in ("undo_button", "clear_step_button", "preview_step_button"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.state(["!disabled"])
        self._sync_step_result_button()
        if step.key == "field_boundary":
            self._sync_boundary_controls()
        self.current_group.set(1)
        if step.key == "markers":
            self._sync_marker_group_controls()
        if step.key == "finish":
            self._refresh_finish_summary()
        if step.key == "keypoints":
            self._refresh_keypoint_progress()
        if step.key == "field_boundary":
            self._refresh_boundary_status()
        self._refresh_step_list()
        self._redraw()

    def _sync_step_result_button(self) -> None:
        if self.current_step.key in ("field_boundary", "train_area", "markers"):
            self.open_step_result_button.grid()
        else:
            self.open_step_result_button.grid_remove()

    def _refresh_finish_summary(self) -> None:
        if not hasattr(self, "finish_summary"):
            return
        lines = []
        for i, step in enumerate(self._active_steps()[:-1]):
            done = self._step_complete(step.key)
            icon = "✔" if done else ("△ 可选" if step.optional else "✘")
            lines.append(f"{icon}  {i + 1}. {step.title}")
        distortion = self._store_dir() / "distortion_params.json"
        lines.append("")
        if self.boundary_enabled_var.get():
            lines.append("畸变参数：四边模式不使用旧畸变参数")
        else:
            lines.append("畸变参数：" + ("已缓存（标注变化后自动清理）" if distortion.exists() else "将由主流程自动计算"))
        lines.append("场地边界：" + ("四边标定" if self.boundary_enabled_var.get() else "旧模式"))
        self.finish_summary.configure(text="\n".join(lines))

    # ------------------------------------------------------------------
    # 项目设置
    # ------------------------------------------------------------------

    def apply_setup(self) -> None:
        if self._measurement_running:
            messagebox.showwarning("项目设置", "测量正在运行，请完成后再修改项目配置。")
            return
        selected_step_key = self.current_step.key
        new_name = self.camera_name_var.get().strip()
        if not new_name:
            messagebox.showwarning("项目设置", "相机名称不能为空。")
            return
        video_text = self.video_path_var.get().strip()
        video = Path(video_text).resolve() if video_text else None
        if self.media_kind == "image":
            if not self.source_path or not self.source_path.exists():
                messagebox.showwarning("项目设置", "当前图片不存在，请重新导入。")
                return
            if video:
                frame = self._read_video_frame(video, 0)
                if frame is None or frame.shape[:2] != (self.image_h, self.image_w):
                    messagebox.showwarning(
                        "项目设置", "关联的视频无法读取，或分辨率与当前图片不一致。")
                    return
        else:
            if not video:
                messagebox.showwarning("项目设置", "请先导入视频。")
                return
            try:
                identity = self._media_file_identity(video)
            except OSError:
                messagebox.showwarning("项目设置", f"视频文件不存在：\n{video}")
                return
            if (self.source_path != video or
                    (self.media_identity is not None and identity != self.media_identity)):
                frame = self._read_video_frame(video, 0)
                if frame is None:
                    messagebox.showwarning("项目设置", "新视频无法读取首帧。")
                    return
                if not self._activate_new_media(video, frame, "video"):
                    return
                selected_step_key = "setup"
        new_root = self.store_root_var.get().strip() or "store"
        root_path = Path(new_root)
        if not root_path.is_absolute():
            root_path = self.project_dir / root_path
        try:
            root_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showwarning("项目设置", f"输出目录无法创建：\n{root_path}\n{exc}")
            return

        name_changed = new_name != self.camera_name
        root_changed = new_root != self.store_root
        config_name = self.selected_pitch_config.get()
        config_changed = config_name != self.pitch_config.__class__.__name__
        self.camera_name = new_name
        self.store_root = new_root
        if self.media_kind == "video":
            self.source_path = video

        self.pitch_config = self._load_pitch_config(config_name)
        self._refresh_vertex_options()

        self._write_config_yaml()
        self._refresh_store_dir_hint()
        if name_changed or root_changed or config_changed:
            self._reset_annotation_state()
        frame = (self._read_image(self.source_path) if self.media_kind == "image"
                 else self._read_video_frame(video, self.frame_index.get()))
        if frame is not None:
            self.set_frame(frame)
        if name_changed or root_changed or config_changed:
            self._load_existing_annotations()
        visible_keys = {step.key for step in self._active_steps()}
        self._select_step_by_key(
            selected_step_key if selected_step_key in visible_keys else "field_boundary")
        if self._keypoint_geometry_warning:
            self.status.set("⚠ " + self._keypoint_geometry_warning)
        else:
            self.status.set(f"配置已保存：相机 {self.camera_name}，场地 {config_name}")

    def _write_config_yaml(self, *, save_boundary_mode: bool = False) -> None:
        video_text = self.video_path_var.get().strip()
        video = Path(video_text).resolve() if video_text else None
        data = self._load_config()
        data["dir_path"] = str(video.parent) if video else ""
        data["video_name"] = video.name if video else ""
        data["CAMERA_NAME"] = self.camera_name
        data["config"] = self.selected_pitch_config.get()
        data["store_dir"] = self.store_root
        data["store_base_dir"] = self.store_base_root
        data["isolated_calibration_session"] = self.isolated_calibration_session
        data["annotation_media_kind"] = self.media_kind
        data["annotation_media_path"] = str(self.source_path) if self.source_path else ""
        data["annotation_media_identity"] = self.media_identity
        if save_boundary_mode:
            saved_modes = data.get("four_boundary_enabled_by_camera")
            saved_modes = dict(saved_modes) if isinstance(saved_modes, dict) else {}
            saved_modes[self.camera_name] = bool(self.boundary_enabled_var.get())
            data["four_boundary_enabled_by_camera"] = saved_modes
        data.setdefault("MCS_NAME", None)
        source_indices = measurement_group_source_indices(self.marker_groups)
        sync_measurement_group_configs(
            data, effective_marker_group_count(self.marker_groups), source_indices,
            self._config_source_indices, self.marker_group_closed,
        )
        data.setdefault("jac", "3-point")
        data.setdefault("reprojection_eps", 5.0)
        data.setdefault("enable_orientation", True)
        with (self.project_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        self.config_data = data
        self._config_source_indices = source_indices

    # ------------------------------------------------------------------
    # 关键点
    # ------------------------------------------------------------------

    def _refresh_vertex_options(self) -> None:
        labels = list(self.pitch_config.labels)
        values = [f"{i}: {labels[i]}  {tuple(map(int, self.pitch_config.vertices[i]))}"
                  for i in range(len(labels))]
        if hasattr(self, "vertex_combo"):
            self.vertex_combo["values"] = values
            if values:
                self.vertex_combo.current(0)
                self.selected_vertex.set(0)
            self._draw_pitch_minimap()
        self._vertex_values = values

    def on_vertex_selected(self, _event=None) -> None:
        self.selected_vertex.set(self.vertex_combo.current())
        self._draw_pitch_minimap()

    def next_unannotated_vertex(self) -> None:
        count = len(self.vertex_combo["values"])
        if count == 0:
            return
        annotated = {a.vertex_index for a in self.keypoints}
        start = self.selected_vertex.get()
        idx = (start + 1) % count
        for offset in range(1, count + 1):
            candidate = (start + offset) % count
            if candidate not in annotated:
                idx = candidate
                break
        self.vertex_combo.current(idx)
        self.selected_vertex.set(idx)
        self._draw_pitch_minimap()

    def _refresh_keypoint_progress(self) -> None:
        if hasattr(self, "keypoint_progress"):
            total = len(self.pitch_config.labels)
            done = len({a.vertex_index for a in self.keypoints})
            need = "（已满足最少 4 个）" if done >= 4 else f"（至少还需 {4 - done} 个）"
            self.keypoint_progress.configure(text=f"已标注 {done}/{total} 个顶点 {need}")
        self._draw_pitch_minimap()

    def _refresh_boundary_status(self) -> None:
        if hasattr(self, "boundary_count_labels"):
            for key in BOUNDARY_UI_ORDER:
                self.boundary_count_labels[key].configure(
                    text=f"{len(self.boundary_edges[key])}/2")
        if not self.boundary_enabled_var.get():
            self.boundary_status_var.set(
                "已切换旧模式；四边标注操作不可用。请完成关键点、横向和纵向直线标注。"
                "如已有四边文件，点“保存旧模式”停用该文件。")
            return
        selected = self.boundary_edge_var.get()
        count = len(self.boundary_edges.get(selected, []))
        if all(len(self.boundary_edges[key]) == 2 for key in BOUNDARY_KEYS):
            self.boundary_status_var.set(
                "四条边线已标齐。请检查轮廓，再点“保存本步”。")
        else:
            self.boundary_status_var.set(
                f"当前：{BOUNDARY_LABELS.get(selected, selected)}，已标 {count}/2 点。"
                "每条线选两个相距较远的位置。")
        if (self.boundary_source_name and self.source_path and
                Path(self.boundary_source_name).name != self.source_path.name):
            self.boundary_status_var.set(
                self.boundary_status_var.get() + "\n注意：边线取自另一视频/图片；"
                "请确认是同一固定机位且构图未变。")

    def _next_unfinished_boundary_edge(self, current: str) -> Optional[str]:
        """Advance in the displayed order, wrapping past completed edges."""
        if current not in BOUNDARY_UI_ORDER:
            return None
        start = BOUNDARY_UI_ORDER.index(current)
        for offset in range(1, len(BOUNDARY_UI_ORDER)):
            candidate = BOUNDARY_UI_ORDER[(start + offset) % len(BOUNDARY_UI_ORDER)]
            if len(self.boundary_edges[candidate]) < 2:
                return candidate
        return None

    def _build_boundary_data(self) -> Dict[str, Any]:
        saved_path = self._step_save_path("field_boundary")
        if not self.boundary_enabled_var.get() and saved_path.exists():
            # 关闭四边模式只停用已保存的几何，不写入尚未保存的画布修改。
            previous = load_boundary(
                saved_path, self.camera_name, self.pitch_config,
                (self.image_w, self.image_h),
            )
            return {
                **previous, "enabled": False,
                "edges": {
                    key: {field: value for field, value in previous["edges"][key].items()
                          if field != "estimated"}
                    for key in BOUNDARY_KEYS
                },
            }
        if not all(len(self.boundary_edges[key]) == 2 for key in BOUNDARY_KEYS):
            remaining = [BOUNDARY_LABELS[key] for key in BOUNDARY_UI_ORDER
                         if len(self.boundary_edges[key]) != 2]
            raise ValueError("请先给四条边线各标记两个点；未完成：" + "、".join(remaining))
        data = {
            "schema_version": 1,
            "enabled": bool(self.boundary_enabled_var.get()),
            "camera_id": self.camera_name,
            "config_type": self.pitch_config.__class__.__name__,
            "pitch_size_cm": [float(self.pitch_config.length), float(self.pitch_config.width)],
            "image_size": [int(self.image_w), int(self.image_h)],
            "source": self.source_path.name if self.source_path else "frame",
            "frame_index": int(self.frame_index.get()),
            "edges": {
                key: {
                    "points": [[int(x), int(y)] for x, y in self.boundary_edges[key]],
                }
                for key in BOUNDARY_KEYS
            },
        }
        # 保存前就检查相邻边线交点和四边形，避免生成主流程无法加载的标定。
        validate_boundary_data(
            data, camera_id=self.camera_name, config=self.pitch_config,
            image_size=(self.image_w, self.image_h),
        )
        compute_boundary_homography({**data, "enabled": True})
        return data

    def preview_boundary(self) -> None:
        if not self.boundary_enabled_var.get():
            return
        if not all(len(self.boundary_edges[key]) == 2 for key in BOUNDARY_KEYS):
            messagebox.showwarning("四条边线标定", "请先给四条边线各标记两个点，再预览。")
            return
        try:
            matrix, _quad = compute_boundary_homography(
                {**self._build_boundary_data(), "enabled": True})
        except ValueError as exc:
            self.boundary_status_var.set(f"无法预览：{exc}")
            messagebox.showwarning("四条边线标定", str(exc))
            return
        suffix = ""
        if self.keypoints:
            image_points = np.asarray(
                [item.img_coord for item in self.keypoints], dtype=np.float32,
            ).reshape(-1, 1, 2)
            expected = np.asarray(
                [self.pitch_config.vertices[item.vertex_index] for item in self.keypoints],
                dtype=np.float32,
            )
            projected = cv2.perspectiveTransform(image_points, matrix).reshape(-1, 2)
            errors = np.linalg.norm(projected - expected, axis=1)
            if np.isfinite(errors).all():
                suffix += f"；与已标关键点的中位差 {np.median(errors) / 100:.1f} 米（仅供参考）"
        self.boundary_status_var.set(
            f"已计算四角和球场线投影{suffix}。"
            "几何通过不代表真实边线正确，请核对可见的中线、禁区线与方向后保存。")
        self.boundary_preview_lines = True
        self.status.set("边线轮廓已预览；运行测量前请保存本步")
        self._redraw()

    def clear_selected_boundary_edge(self) -> None:
        if not self.boundary_enabled_var.get():
            return
        key = self.boundary_edge_var.get()
        if key not in self.boundary_edges:
            return
        self.boundary_edges[key].clear()
        self._boundary_last_advanced_from = None
        self.boundary_preview_lines = False
        self._refresh_boundary_status()
        self._refresh_step_list()
        self._redraw()

    # ------------------------------------------------------------------
    # 场地示意图（小地图）
    # ------------------------------------------------------------------

    def _minimap_transform(self) -> Tuple[float, float, float]:
        """返回 (scale, offset_x, offset_y)：球场坐标(cm) -> 小地图画布像素。"""
        length = float(self.pitch_config.length)
        width = float(self.pitch_config.width)
        cw = int(self.minimap_canvas["width"])
        ch = int(self.minimap_canvas["height"])
        pad = 18
        s = min((cw - 2 * pad) / length, (ch - 2 * pad) / width)
        ox = (cw - length * s) / 2
        oy = (ch - width * s) / 2
        return s, ox, oy

    def _draw_pitch_minimap(self) -> None:
        if not hasattr(self, "minimap_canvas"):
            return
        c = self.minimap_canvas
        c.delete("all")
        vertices = list(self.pitch_config.vertices)
        labels = list(self.pitch_config.labels)
        s, ox, oy = self._minimap_transform()

        def pt(v) -> Tuple[float, float]:
            return ox + float(v[0]) * s, oy + float(v[1]) * s

        line_color = "#86efac"
        # 场地外框
        x0, y0 = pt((0, 0))
        x1, y1 = pt((self.pitch_config.length, self.pitch_config.width))
        c.create_rectangle(x0, y0, x1, y1, outline=line_color)
        # 中圈
        if hasattr(self.pitch_config, "centre_circle_radius"):
            r = float(self.pitch_config.centre_circle_radius) * s
            is_half = isinstance(self.pitch_config, (
                soccer.HalfPitchConfiguration,
                soccer.HalfQingYuanConfiguration,
                soccer.HalfWuChuanConfiguration,
                soccer.HalfWuChuanYiZhongConfiguration,
            ))
            if is_half:
                is_right = getattr(self.pitch_config, "is_right_half", False)
                centre_x = 0 if is_right else self.pitch_config.length
                cx, cy = pt((centre_x, self.pitch_config.width / 2))
                c.create_arc(
                    cx - r, cy - r, cx + r, cy + r,
                    start=270 if is_right else 90, extent=180,
                    style=tk.ARC, outline=line_color,
                )
            else:
                cx, cy = pt((self.pitch_config.length / 2, self.pitch_config.width / 2))
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=line_color)
        # 场地线（edges 为 1-based 顶点编号）
        for edge in getattr(self.pitch_config, "edges", []):
            a, b = edge[0] - 1, edge[1] - 1
            if 0 <= a < len(vertices) and 0 <= b < len(vertices):
                c.create_line(*pt(vertices[a]), *pt(vertices[b]), fill=line_color)

        annotated = {a.vertex_index for a in self.keypoints}
        current = self.selected_vertex.get()
        for i, v in enumerate(vertices):
            if i == current:
                continue
            x, y = pt(v)
            fill = "#22c55e" if i in annotated else "#d1d5db"
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=fill, outline="")

        # 当前顶点高亮（最后画，保证在最上层）
        if 0 <= current < len(vertices):
            x, y = pt(vertices[current])
            c.create_oval(x - 9, y - 9, x + 9, y + 9, outline="#ef4444", width=2)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ef4444", outline="white")
            label = labels[current] if current < len(labels) else str(current)
            ty = y - 14 if y > oy + 20 else y + 14
            c.create_text(x, ty, text=label, fill="#fbbf24", font=(UI_FONT, 10, "bold"))

    def _on_minimap_click(self, event) -> None:
        vertices = list(self.pitch_config.vertices)
        s, ox, oy = self._minimap_transform()
        best = None
        best_d = 12 ** 2
        for i, v in enumerate(vertices):
            x = ox + float(v[0]) * s
            y = oy + float(v[1]) * s
            d = (x - event.x) ** 2 + (y - event.y) ** 2
            if d < best_d:
                best, best_d = i, d
        if best is None:
            return
        self.vertex_combo.current(best)
        self.selected_vertex.set(best)
        self._draw_pitch_minimap()
        labels = list(self.pitch_config.labels)
        label = labels[best] if best < len(labels) else str(best)
        self.status.set(f"已选择顶点 {best}: {label}，请在画面中点击其位置")

    # ------------------------------------------------------------------
    # 画布交互
    # ------------------------------------------------------------------

    def _fit_image(self) -> None:
        self.update_idletasks()
        cw = max(1, self.canvas.winfo_width() or 900) if hasattr(self, "canvas") else 900
        ch = max(1, self.canvas.winfo_height() or 650) if hasattr(self, "canvas") else 650
        self.scale = min(cw / self.image_w, ch / self.image_h) * 0.96
        self.offset_x = int((cw - self.image_w * self.scale) / 2)
        self.offset_y = int((ch - self.image_h * self.scale) / 2)

    def fit_and_redraw(self) -> None:
        self._fit_image()
        self._redraw()

    def zoom(self, factor: float) -> None:
        self.scale = max(0.05, min(8.0, self.scale * factor))
        self._redraw()

    def on_mousewheel(self, event) -> None:
        steps = wheel_steps(event)
        if steps == 0:
            return
        self.zoom(1.1 if steps > 0 else 1 / 1.1)

    def start_pan(self, event) -> None:
        self._pan_start = (event.x, event.y)

    def do_pan(self, event) -> None:
        if self._pan_start is None:
            return
        px, py = self._pan_start
        self.offset_x += event.x - px
        self.offset_y += event.y - py
        self._pan_start = (event.x, event.y)
        self._redraw()

    def image_to_canvas(self, x: int, y: int) -> Tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def canvas_to_image(self, x: int, y: int) -> Optional[Point]:
        ix = int(round((x - self.offset_x) / self.scale))
        iy = int(round((y - self.offset_y) / self.scale))
        if 0 <= ix < self.image_w and 0 <= iy < self.image_h:
            return ix, iy
        return None

    def _active_group_store(self) -> Optional[List[List[Point]]]:
        key = self.current_step.key
        if key == "markers":
            return self.marker_groups
        if key == "manual_h":
            return self.manual_h_groups
        if key == "manual_v":
            return self.manual_v_groups
        return None

    def on_canvas_click(self, event) -> None:
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        key = self.current_step.key
        if key in ("setup", "finish"):
            self.status.set("当前步骤无需在画面上标注。")
            return
        if key == "keypoints":
            vertex_index = self.selected_vertex.get()
            self.keypoints = [a for a in self.keypoints if a.vertex_index != vertex_index]
            self.keypoints.append(KeypointAnnotation(vertex_index=vertex_index, img_coord=point))
            self.keypoint_source_name = ""
            self.status.set(f"已标注关键点 {vertex_index}: {point}")
            self.next_unannotated_vertex()
            self._refresh_keypoint_progress()
        elif key == "field_boundary":
            if not self.boundary_enabled_var.get():
                self.status.set("四边标定已关闭；请在旧模式步骤标注。")
                return
            self._boundary_last_advanced_from = None
            edge_key = self.boundary_edge_var.get()
            edge_points = self.boundary_edges[edge_key]
            if len(edge_points) >= 2:
                self.status.set("该边线已有两个点；请先点“清除选中边线”再重画")
                return
            edge_points.append(point)
            self.boundary_preview_lines = False
            if len(edge_points) == 2:
                next_key = self._next_unfinished_boundary_edge(edge_key)
                if next_key is None:
                    self.status.set("四条边线均已标满两个点，请检查并预览四边范围。")
                else:
                    self.boundary_edge_var.set(next_key)
                    self._boundary_last_advanced_from = edge_key
                    self.status.set(
                        f"{BOUNDARY_LABELS[edge_key]}已完成，"
                        f"已自动切换到{BOUNDARY_LABELS[next_key]}。")
            else:
                self.status.set(
                    f"{BOUNDARY_LABELS[edge_key]}：已标 {len(edge_points)}/2 点 {point}")
            self._refresh_boundary_status()
        elif key == "train_area":
            self.train_area_points.append(point)
            self.status.set(f"已添加自定义点 V{len(self.train_area_points)}: {point}")
        else:
            if key == "markers" and not self.marker_groups:
                self.status.set("当前为 0 组器材；如需标注，请先点“新增一组器材”。")
                return
            group_idx = self.ensure_group() - 1
            groups = self._active_group_store()
            if groups is None:
                return
            groups[group_idx].append(point)
            self.status.set(f"已添加点：第 {group_idx + 1} 组 {point}")
        self._refresh_step_list()
        self._redraw()

    # ------------------------------------------------------------------
    # 分组 / 撤销 / 清空
    # ------------------------------------------------------------------

    def _remember_marker_equipment_type(self, *_args) -> None:
        index = self._active_marker_type_index
        if 0 <= index < len(self.marker_equipment_types):
            self.marker_equipment_types[index] = normalize_equipment_type(
                self.marker_equipment_type_var.get())
            self._refresh_list()

    def _remember_marker_layout_mode(self) -> None:
        index = self._active_marker_type_index
        if 0 <= index < len(self.marker_group_closed):
            self.marker_group_closed[index] = self.marker_layout_var.get() == "ring"
            self._refresh_list()
            mode = "环型训练" if self.marker_group_closed[index] else "线型训练"
            self.status.set(f"第 {index + 1} 组已设为{mode}；保存本步后生效。")

    def _sync_marker_group_controls(self) -> None:
        count = len(self.marker_groups)
        self.marker_group_count_var.set(str(count))
        for widget in (
            getattr(self, "marker_current_group_spin", None),
            getattr(self, "marker_equipment_type_combo", None),
            *getattr(self, "marker_layout_buttons", ()),
        ):
            if widget is not None:
                widget.state(["!disabled"] if count else ["disabled"])
        if count == 0:
            self.current_group.set(0)
            self._active_marker_type_index = -1
            self.marker_equipment_type_var.set(DEFAULT_EQUIPMENT_TYPE)
            self.marker_layout_var.set("line")
            return
        try:
            group_number = int(self.current_group.get())
        except (tk.TclError, ValueError):
            group_number = 1
        group_number = max(1, min(group_number, count))
        self.current_group.set(group_number)
        self._active_marker_type_index = group_number - 1
        self.marker_equipment_type_var.set(self.marker_equipment_types[self._active_marker_type_index])
        self.marker_layout_var.set(
            "ring" if self.marker_group_closed[self._active_marker_type_index] else "line")

    def apply_marker_group_count(self) -> None:
        try:
            target = int(self.marker_group_count_var.get().strip())
            if not 0 <= target <= 99:
                raise ValueError
        except ValueError:
            messagebox.showwarning("器材组数", "请输入 0–99 之间的整数。")
            self.marker_group_count_var.set(str(len(self.marker_groups)))
            return

        old_count = len(self.marker_groups)
        if target < old_count:
            removed_groups = [
                f"第 {index + 1} 组（{len(group)} 个点，{self.marker_equipment_types[index]}，"
                f"{'环型' if self.marker_group_closed[index] else '线型'}）"
                for index, group in enumerate(self.marker_groups[target:], start=target)
                if (group or self.marker_equipment_types[index] != DEFAULT_EQUIPMENT_TYPE
                    or self.marker_group_closed[index])
            ]
            if removed_groups and not messagebox.askyesno(
                "减少器材组数",
                "减少组数会删除以下组的标注点、器材类型或训练形式：\n"
                + "\n".join(removed_groups) + "\n\n确定继续吗？",
            ):
                self.marker_group_count_var.set(str(old_count))
                return

        self.marker_groups, self.marker_equipment_types = resize_marker_groups(
            self.marker_groups, self.marker_equipment_types, target)
        self.marker_group_closed = resize_group_closed_modes(
            self.marker_group_closed, target)
        try:
            selected_group = int(self.current_group.get())
        except (tk.TclError, ValueError):
            selected_group = 1
        self.current_group.set(min(selected_group, target) if target else 0)
        self._sync_marker_group_controls()
        self._refresh_step_list()
        self._redraw()
        self.status.set(
            "已设置 0 组器材；可直接跳过此步骤。" if target == 0
            else f"已设置 {target} 组器材；请为每组选择类型并在画面中标点。"
        )

    def select_marker_group(self) -> None:
        if not self.marker_groups:
            self._sync_marker_group_controls()
            return
        try:
            group_number = int(self.current_group.get())
            if not 1 <= group_number <= 99:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showwarning("当前标注组", "请输入 1–99 之间的组号。")
            self.current_group.set(self._active_marker_type_index + 1)
            return
        if group_number > len(self.marker_groups):
            self.marker_groups, self.marker_equipment_types = resize_marker_groups(
                self.marker_groups, self.marker_equipment_types, group_number)
            self.marker_group_closed = resize_group_closed_modes(
                self.marker_group_closed, group_number)
        self._sync_marker_group_controls()
        self._refresh_step_list()
        self._redraw()

    def ensure_group(self) -> int:
        if self.current_step.key == "markers" and not self.marker_groups:
            # Selecting a step or clicking the canvas must not create an
            # equipment group behind an explicit 0-group setting.
            self.current_group.set(0)
            return 0
        try:
            group_number = max(1, min(99, int(self.current_group.get())))
        except (tk.TclError, ValueError):
            group_number = 1
        groups = self._active_group_store()
        if groups is not None:
            while len(groups) < group_number:
                groups.append([])
                if self.current_step.key == "markers":
                    self.marker_equipment_types.append(DEFAULT_EQUIPMENT_TYPE)
                    self.marker_group_closed.append(False)
        self.current_group.set(group_number)
        if self.current_step.key == "markers":
            self._sync_marker_group_controls()
        return group_number

    def new_group(self) -> None:
        groups = self._active_group_store()
        if groups is not None:
            if self.current_step.key == "markers" and len(groups) >= 99:
                messagebox.showwarning("新增器材组", "最多可设置 99 组器材。")
                return
            groups.append([])
            if self.current_step.key == "markers":
                self.marker_equipment_types.append(DEFAULT_EQUIPMENT_TYPE)
                self.marker_group_closed.append(False)
            self.current_group.set(len(groups))
            if self.current_step.key == "markers":
                self._sync_marker_group_controls()
                self.status.set(f"已新增第 {len(groups)} 组器材，请在画面中标点。")
        self._refresh_step_list()
        self._redraw()

    def undo(self) -> None:
        key = self.current_step.key
        if key == "keypoints":
            if self.keypoints:
                removed = self.keypoints.pop()
                self.keypoint_source_name = ""
                self.status.set(f"已撤销关键点 {removed.vertex_index}")
                self._refresh_keypoint_progress()
        elif key == "field_boundary":
            if not self.boundary_enabled_var.get():
                return
            edge_key = self.boundary_edge_var.get()
            edge_points = self.boundary_edges[edge_key]
            if not edge_points and self._boundary_last_advanced_from:
                edge_key = self._boundary_last_advanced_from
                edge_points = self.boundary_edges[edge_key]
                self.boundary_edge_var.set(edge_key)
            if edge_points:
                edge_points.pop()
                self.boundary_preview_lines = False
                self.status.set(f"已撤销{BOUNDARY_LABELS[edge_key]}的最后一个点。")
            self._boundary_last_advanced_from = None
            self._refresh_boundary_status()
        elif key == "train_area":
            if self.train_area_points:
                self.train_area_points.pop()
        elif key in GROUP_STEPS:
            groups = self._active_group_store()
            if groups:
                group_idx = self.ensure_group() - 1
                if 0 <= group_idx < len(groups) and groups[group_idx]:
                    groups[group_idx].pop()
        self._refresh_step_list()
        self._redraw()

    def clear_current(self) -> None:
        key = self.current_step.key
        if key == "keypoints":
            if self.keypoints and not messagebox.askyesno("清空", "确定清空全部关键点标注？"):
                return
            self.keypoints.clear()
            self.keypoint_source_name = ""
            self._refresh_keypoint_progress()
        elif key == "field_boundary":
            if not self.boundary_enabled_var.get():
                return
            if any(self.boundary_edges.values()) and not messagebox.askyesno(
                "清空", "确定清空内存中的四条边线？已保存文件不会删除。"
            ):
                return
            self.boundary_edges = {name: [] for name in BOUNDARY_KEYS}
            self._boundary_last_advanced_from = None
            self.boundary_preview_lines = False
            self._refresh_boundary_status()
        elif key == "train_area":
            self.train_area_points.clear()
            self.train_area_region = []
            self._refresh_train_area_status()
        elif key in GROUP_STEPS:
            groups = self._active_group_store()
            if groups:
                group_idx = self.ensure_group() - 1
                if 0 <= group_idx < len(groups):
                    groups[group_idx].clear()
        self._refresh_step_list()
        self._redraw()

    # ------------------------------------------------------------------
    # 训练区域
    # ------------------------------------------------------------------

    def _video_points_to_pitch(self, points: List[Point]) -> np.ndarray:
        if self.boundary_enabled_var.get():
            boundary = self._build_boundary_data()
            H, _quad = compute_boundary_homography(boundary)
            return cv2.perspectiveTransform(
                np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), H,
            ).reshape(-1, 2)
        if len(self.keypoints) < 4:
            raise ValueError("需要先完成至少 4 个球场关键点标注")
        from tools.transform import compute_transform_matrix, video_pixel_to_pitch

        indices = [a.vertex_index for a in self.keypoints]
        image_points = [a.img_coord for a in self.keypoints]
        distortion_params = None
        distortion_info = None
        distortion_path = self._store_dir() / "distortion_params.json"
        if distortion_path.exists():
            from de_distortion import load_distortion_params
            distortion_params, distortion_info = load_distortion_params(str(distortion_path))
        center = distortion_info.get("center") if distortion_info else None
        H = compute_transform_matrix(indices, image_points, self.pitch_config,
                                     distortion_params=distortion_params, image_center=center)
        return video_pixel_to_pitch(np.array(points, dtype=np.float32), H,
                                    distortion_params=distortion_params,
                                    distortion_info=distortion_info)

    def generate_train_area_region(self) -> None:
        raw = self.train_area_tokens.get().strip()
        if not raw:
            messagebox.showwarning("训练区域", "请先输入点序列，如：1, 14, V1, V2")
            return
        tokens = raw.replace("，", ",").replace(",", " ").split()
        labels = list(self.pitch_config.labels)
        vertices = list(self.pitch_config.vertices)

        v_pitch: Optional[np.ndarray] = None
        region: List[List[float]] = []
        problems: List[str] = []
        for token in tokens:
            t = token.upper()
            if t.startswith("V") and t[1:].isdigit():
                idx = int(t[1:]) - 1
                if not (0 <= idx < len(self.train_area_points)):
                    problems.append(f"视频点 {t} 不存在")
                    continue
                if v_pitch is None:
                    try:
                        v_pitch = self._video_points_to_pitch(self.train_area_points)
                    except Exception as exc:
                        messagebox.showwarning("训练区域", f"V 点转换球场坐标失败：\n{exc}")
                        return
                region.append([float(v_pitch[idx][0]), float(v_pitch[idx][1])])
            else:
                idx = None
                if t in labels:
                    idx = labels.index(t)
                elif t.isdigit() and f"{int(t):02d}" in labels:
                    idx = labels.index(f"{int(t):02d}")
                if idx is None or idx >= len(vertices):
                    problems.append(f"未找到关键点 {token}")
                    continue
                x, y = vertices[idx]
                region.append([float(x), float(y)])

        if problems:
            messagebox.showwarning("训练区域", "以下点已忽略：\n" + "\n".join(problems))
        if len(region) < 3:
            messagebox.showwarning("训练区域", f"有效点仅 {len(region)} 个，至少需要 3 个才能构成区域。")
            return
        self.train_area_region = region
        self._refresh_train_area_status()
        self._refresh_step_list()
        self.status.set(f"训练区域已生成：{len(region)} 个点，点「保存本步」写入 train_area.json")

    def _refresh_train_area_status(self) -> None:
        if not hasattr(self, "train_area_status"):
            return
        if self.train_area_region:
            preview = "，".join(f"({p[0] / 100:.1f}m, {p[1] / 100:.1f}m)" for p in self.train_area_region[:4])
            more = "…" if len(self.train_area_region) > 4 else ""
            self.train_area_status.configure(
                text=f"区域：{len(self.train_area_region)} 个点\n{preview}{more}")
        else:
            self.train_area_status.configure(text="区域：未生成")

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    def _require_applied_pitch_config(self) -> bool:
        if self.selected_pitch_config.get() == self.pitch_config.__class__.__name__:
            return True
        messagebox.showwarning("场地配置未应用", "请先点击「应用并保存配置」，再保存标注或运行测量。")
        return False

    def _confirm_replacing_mismatched_keypoints(self, path: Path) -> bool:
        if not self._keypoint_geometry_warning or not path.exists():
            return True
        return messagebox.askyesno(
            "确认覆盖旧关键点",
            f"{self._keypoint_geometry_warning}\n\n旧文件目前仍保留：\n{path}\n\n"
            "仅在您已按当前场地配置重新标注关键点、并确认不再需要旧关键点时，"
            "才选择“是”覆盖。否则请选择“否”，或更换相机名称保存新标注。",
        )

    def _archive_invalid_boundary_for_legacy(self, path: Path, error: ValueError) -> bool:
        """Keep an unusable boundary file recoverable before switching to legacy mode."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.stem}.disabled-backup-{stamp}{path.suffix}")
        suffix = 1
        while backup.exists():
            backup = path.with_name(
                f"{path.stem}.disabled-backup-{stamp}-{suffix}{path.suffix}")
            suffix += 1
        if not messagebox.askyesno(
            "备份无效的四边标定",
            f"已有四边标定文件与当前相机、场地或画面不匹配：\n{error}\n\n"
            f"旧模式无法读取该文件。是否将它备份到：\n{backup}\n\n"
            "原内容会保留在备份中，活动路径移空后才能使用旧模式。",
        ):
            return False
        try:
            path.rename(backup)
        except OSError as exc:
            messagebox.showwarning("备份失败", f"无法备份 {path}：{exc}")
            return False
        self.status.set(f"已备份无效四边标定：{backup}")
        return True

    def save_current_step(self) -> None:
        if not self._require_applied_pitch_config():
            return
        key = self.current_step.key
        path = self._step_save_path(key)
        if path is None:
            return
        if key == "field_boundary" and not self.boundary_enabled_var.get() and not path.exists():
            self._write_config_yaml(save_boundary_mode=True)
            self.status.set("已选择旧模式；没有四边标定文件需要停用。")
            self._refresh_step_list()
            return
        try:
            data = self._build_step_data(key)
        except ValueError as exc:
            if key == "field_boundary" and not self.boundary_enabled_var.get() and path.exists():
                if self._archive_invalid_boundary_for_legacy(path, exc):
                    self._write_config_yaml(save_boundary_mode=True)
                    self._refresh_step_list()
                return
            messagebox.showwarning("保存", str(exc))
            return
        if key == "keypoints" and not self._confirm_replacing_mismatched_keypoints(path):
            return
        try:
            _save_annotation_file(
                path, data,
                self._derived_cache_paths() if key in ("keypoints", "manual_h", "manual_v")
                else (self._store_dir() / "camera_info.json",) if key == "field_boundary"
                else self._marker_result_paths() if key == "markers"
                else (),
            )
        except OSError as exc:
            messagebox.showwarning("保存", f"保存 {path.name} 失败：{exc}")
            return
        if key == "markers":
            self._write_config_yaml()
        if key == "keypoints":
            self._keypoint_geometry_warning = None
            self.keypoint_source_name = str(data.get("source") or "")
        if key == "field_boundary":
            self._write_config_yaml(save_boundary_mode=True)
            self.boundary_source_name = str(data.get("source") or "")
            self._refresh_boundary_status()
        self.status.set(f"已保存 {path}")
        self._refresh_step_list()
        if key == "markers":
            self._warn_unsaved_marker_groups()
            self._refresh_finish_summary()

    def _warn_unsaved_marker_groups(self) -> None:
        empty_numbers = [str(index + 1) for index, group in enumerate(self.marker_groups) if not group]
        if not empty_numbers:
            return
        if not any(self.marker_groups):
            self.status.set(self.status.get() + "；未标注器材，已按无器材训练保存")
        else:
            self.status.set(self.status.get() + "；空白器材组 "
                            + "、".join(empty_numbers) + " 已忽略")

    def _build_step_data(self, key: str) -> Dict[str, Any]:
        if key == "keypoints":
            if len(self.keypoints) < 4:
                raise ValueError(f"关键点至少需要 4 个，当前只有 {len(self.keypoints)} 个。")
            return self._build_keypoints_data()
        if key == "manual_h":
            return self._build_groups_data(self.manual_h_groups, "manual_horizontal", min_points=2)
        if key == "manual_v":
            return self._build_groups_data(self.manual_v_groups, "manual_vertical", min_points=2)
        if key == "field_boundary":
            return self._build_boundary_data()
        if key == "markers":
            data, _empty_group_numbers = build_marker_data(
                self.marker_groups, self.marker_equipment_types)
            return data
        if key == "train_area":
            if len(self.train_area_region) < 3:
                raise ValueError("请先在「区域构成」中生成至少 3 个点的区域，再保存。")
            return {
                "camera_id": self.camera_name,
                "image_points": [[int(x), int(y)] for x, y in self.train_area_points],
                "region_tokens": self.train_area_tokens.get().strip(),
                "region_points": self.train_area_region,
            }
        raise ValueError(f"步骤 {key} 无可保存内容。")

    def _build_keypoints_data(self) -> Dict[str, Any]:
        source_name = self.source_path.name if self.source_path else "frame"
        labels = list(self.pitch_config.labels)
        vertices = list(self.pitch_config.vertices)
        annotations = []
        for i, ann in enumerate(sorted(self.keypoints, key=lambda item: item.vertex_index)):
            wx, wy = vertices[ann.vertex_index]
            annotations.append({
                "id": i,
                "label": labels[ann.vertex_index],
                "img_coord": [int(ann.img_coord[0]), int(ann.img_coord[1])],
                "world_coord": [int(wx), int(wy)],
                "vertex_index": int(ann.vertex_index),
            })
        return {
            "source": source_name,
            "config_type": self.pitch_config.__class__.__name__,
            "annotations": annotations,
        }

    def _build_groups_data(self, groups_source: List[List[Point]], source: str,
                           min_points: int = 1) -> Dict[str, Any]:
        valid = [g for g in groups_source if len(g) >= min_points]
        if not valid:
            raise ValueError(f"没有满足条件的分组（每组至少 {min_points} 个点）。")
        groups = []
        for group_idx, group in enumerate(valid):
            groups.append({
                "group_id": group_idx,
                "points": [
                    {"id": point_idx, "img_coord": [int(x), int(y)]}
                    for point_idx, (x, y) in enumerate(group)
                ],
            })
        return {"source": source, "type": "marker_annotation", "groups": groups}

    def save_all_default(self) -> bool:
        if not self._require_applied_pitch_config():
            return False
        boundary_path = self._step_save_path("field_boundary")
        boundary_enabled = self.boundary_enabled_var.get()
        if boundary_enabled or boundary_path.exists():
            try:
                self._build_boundary_data()
            except ValueError as exc:
                if not boundary_enabled:
                    if not self._archive_invalid_boundary_for_legacy(boundary_path, exc):
                        return False
                else:
                    messagebox.showwarning(
                        "四条边线未保存",
                        f"{exc}\n\n请完成四边标定，或关闭四边标定后再保存。",
                    )
                    self._select_step_by_key("field_boundary")
                    return False
        self._write_config_yaml()
        saved, skipped = [], []
        steps_to_save = (
            (("field_boundary",) if boundary_enabled else ("keypoints", "manual_h", "manual_v"))
            + (("field_boundary",) if not boundary_enabled and boundary_path.exists() else ())
            + ("train_area", "markers")
        )
        for key in steps_to_save:
            path = self._step_save_path(key)
            try:
                data = self._build_step_data(key)
            except ValueError as exc:
                if key == "field_boundary" and (any(self.boundary_edges.values()) or path.exists()):
                    messagebox.showwarning(
                        "四条边线未保存",
                        f"{exc}\n\n已有边线文件未被更改；请完成并保存，或关闭四边标定后再运行。",
                    )
                    self._select_step_by_key("field_boundary")
                    return False
                skipped.append(key)
                continue
            if key == "keypoints" and not self._confirm_replacing_mismatched_keypoints(path):
                return False
            try:
                _save_annotation_file(
                    path, data,
                    self._derived_cache_paths() if key in ("keypoints", "manual_h", "manual_v")
                    else (self._store_dir() / "camera_info.json",) if key == "field_boundary"
                    else self._marker_result_paths() if key == "markers"
                    else (),
                )
            except OSError as exc:
                messagebox.showwarning("保存", f"保存 {path.name} 失败：{exc}")
                return False
            if key == "keypoints":
                self._keypoint_geometry_warning = None
                self.keypoint_source_name = str(data.get("source") or "")
            if key == "field_boundary":
                self.boundary_source_name = str(data.get("source") or "")
            saved.append(path.name)
        self._refresh_step_list()
        if saved:
            self.status.set("已保存：" + "、".join(saved))
        if skipped:
            names = {"keypoints": "关键点", "manual_h": "横线", "manual_v": "纵线",
                     "field_boundary": "四条边线",
                     "train_area": "训练区域", "markers": "标志物"}
            self.status.set(self.status.get() + "；跳过：" + "、".join(names[k] for k in skipped))
        if "markers.json" in saved:
            self._warn_unsaved_marker_groups()
            self._refresh_finish_summary()
        if saved:
            self._write_config_yaml(save_boundary_mode=True)
        return bool(saved)

    def _derived_cache_paths(self) -> Tuple[Path, ...]:
        return (
            self._store_dir() / "distortion_params.json",
            self._store_dir() / "camera_info.json",
        )

    def _marker_result_paths(self) -> Tuple[Path, ...]:
        output_dir = self._manual_dir()
        return (
            output_dir / "coordinates.json",
            output_dir / "measurement_report.txt",
            output_dir / "visualization.jpg",
        )

    def clear_derived_cache(self) -> None:
        derived = self._derived_cache_paths()
        existing = [p for p in derived if p.exists()]
        if not existing:
            messagebox.showinfo("清理派生缓存", "没有需要清理的派生缓存。")
            return
        names = "\n".join(f"  - {p}" for p in existing)
        if not messagebox.askyesno("清理派生缓存", f"将删除以下文件：\n{names}\n\n确认？"):
            return
        for p in existing:
            try:
                p.unlink()
            except OSError as exc:
                messagebox.showwarning("清理缓存", f"删除 {p.name} 失败：{exc}")
        self._refresh_step_list()
        self._refresh_finish_summary()
        self.status.set("派生缓存已清理。")

    # ------------------------------------------------------------------
    # 预览图
    # ------------------------------------------------------------------

    def _default_preview_path(self) -> Path:
        key = self.current_step.key
        json_path = self._step_save_path(key) or (self._manual_dir() / "preview.json")
        return json_path.with_name(f"{json_path.stem}_preview.jpg")

    def save_preview(self) -> None:
        if self.current_step.key in ("setup", "finish"):
            messagebox.showinfo("生成预览图", "请先切换到具体的标注步骤。")
            return
        if self.current_step.key == "field_boundary" and not self.boundary_enabled_var.get():
            return
        default_path = self._default_preview_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            defaultextension=".jpg",
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png"), ("所有文件", "*.*")],
        )
        if not path:
            return
        preview = self._render_preview_image()
        self._write_image(Path(path), preview)
        self._saved_preview_paths[self.current_step.key] = Path(path)
        self.status.set(f"预览图已保存 {path}")

    def _render_preview_image(self):
        preview = self.frame_bgr.copy()
        key = self.current_step.key
        if key == "keypoints":
            labels = list(self.pitch_config.labels)
            for i, ann in enumerate(self.keypoints):
                color = self._hex_to_bgr(self.POINT_COLORS[i % len(self.POINT_COLORS)])
                label = labels[ann.vertex_index] if ann.vertex_index < len(labels) else str(ann.vertex_index)
                self._draw_cv_point(preview, ann.img_coord, label, color)
        elif key == "field_boundary":
            for name in BOUNDARY_KEYS:
                points = self.boundary_edges[name]
                color = self._hex_to_bgr(BOUNDARY_COLORS[name])
                self._draw_cv_group(preview, points, name[:1].upper(), color)
            try:
                matrix, quad = compute_boundary_homography(
                    {**self._build_boundary_data(), "enabled": True})
                outline = np.round(np.asarray(quad)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview, [outline], True, (255, 255, 255), 2)
                inverse = np.linalg.inv(matrix)
                vertices = np.asarray(self.pitch_config.vertices, dtype=np.float32)
                projected = cv2.perspectiveTransform(
                    vertices.reshape(-1, 1, 2), inverse,
                ).reshape(-1, 2)
                if np.isfinite(projected).all():
                    for start, end in getattr(self.pitch_config, "edges", []):
                        if 1 <= start <= len(projected) and 1 <= end <= len(projected):
                            p0 = tuple(map(int, np.round(projected[start - 1])))
                            p1 = tuple(map(int, np.round(projected[end - 1])))
                            cv2.line(preview, p0, p1, (0, 215, 255), 2)
            except ValueError:
                pass
        elif key == "train_area":
            self._draw_cv_group(preview, self.train_area_points, "V", self._hex_to_bgr(self.POINT_COLORS[4]))
        else:
            groups = self._active_group_store() or []
            prefix = {"markers": "G", "manual_h": "H", "manual_v": "V"}.get(key, "P")
            for group_idx, group in enumerate(groups):
                color = self._hex_to_bgr(self.POINT_COLORS[group_idx % len(self.POINT_COLORS)])
                self._draw_cv_group(preview, group, f"{prefix}{group_idx + 1}-", color)
        return preview

    def _draw_cv_group(self, image, group: List[Point], prefix: str,
                       color: Tuple[int, int, int], close: bool = False) -> None:
        for idx, point in enumerate(group):
            if idx > 0:
                cv2.line(image, group[idx - 1], point, color, 2)
            self._draw_cv_point(image, point, f"{prefix}{idx + 1}", color)
        if close and len(group) >= 3:
            cv2.line(image, group[-1], group[0], color, 2)

    def _draw_cv_point(self, image, point: Point, label: str, color: Tuple[int, int, int]) -> None:
        cv2.circle(image, point, 7, color, -1)
        cv2.circle(image, point, 9, (255, 255, 255), 2)
        cv2.putText(image, label, (point[0] + 10, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _hex_to_bgr(self, color: str) -> Tuple[int, int, int]:
        color = color.lstrip("#")
        return int(color[4:6], 16), int(color[2:4], 16), int(color[0:2], 16)

    def _write_image(self, path: Path, image) -> None:
        ext = path.suffix or ".jpg"
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            raise RuntimeError("预览图编码失败")
        encoded.tofile(str(path))

    # ------------------------------------------------------------------
    # 画布绘制
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        new_size = (max(1, int(self.image_w * self.scale)), max(1, int(self.image_h * self.scale)))
        self.display_image = pil.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)
        self._draw_annotations()
        self._refresh_list()

    def _draw_annotations(self) -> None:
        key = self.current_step.key
        if key in ("setup", "finish"):
            # 概览：淡化显示关键点，帮助确认画面正确
            for ann in self.keypoints:
                x, y = self.image_to_canvas(*ann.img_coord)
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#64748b", outline="")
            return
        if key == "keypoints":
            labels = list(self.pitch_config.labels)
            for i, ann in enumerate(self.keypoints):
                x, y = self.image_to_canvas(*ann.img_coord)
                label = labels[ann.vertex_index] if ann.vertex_index < len(labels) else str(ann.vertex_index)
                self._draw_point(x, y, label, self.POINT_COLORS[i % len(self.POINT_COLORS)])
            return
        if key == "field_boundary":
            if not self.boundary_enabled_var.get():
                return
            for name in BOUNDARY_KEYS:
                points = self.boundary_edges[name]
                color = BOUNDARY_COLORS[name]
                if len(points) == 2:
                    p0 = self.image_to_canvas(*points[0])
                    p1 = self.image_to_canvas(*points[1])
                    self.canvas.create_line(*p0, *p1, fill=color, width=3)
                for index, point in enumerate(points):
                    x, y = self.image_to_canvas(*point)
                    self._draw_point(x, y, f"{name[:1].upper()}{index + 1}", color)
            try:
                matrix, quad = compute_boundary_homography(
                    {**self._build_boundary_data(), "enabled": True})
                canvas_points = [self.image_to_canvas(*point) for point in quad]
                self.canvas.create_polygon(
                    *[coordinate for point in canvas_points for coordinate in point],
                    fill="", outline="white", width=2, dash=(6, 4),
                )
                if self.boundary_preview_lines:
                    inverse = np.linalg.inv(matrix)
                    vertices = np.asarray(self.pitch_config.vertices, dtype=np.float32)
                    projected = cv2.perspectiveTransform(
                        vertices.reshape(-1, 1, 2), inverse,
                    ).reshape(-1, 2)
                    if np.isfinite(projected).all():
                        for start, end in getattr(self.pitch_config, "edges", []):
                            if 1 <= start <= len(projected) and 1 <= end <= len(projected):
                                p0 = self.image_to_canvas(*projected[start - 1])
                                p1 = self.image_to_canvas(*projected[end - 1])
                                self.canvas.create_line(
                                    *p0, *p1, fill="#facc15", width=2, dash=(5, 5),
                                )
            except ValueError:
                pass
            return
        if key == "train_area":
            self._draw_canvas_group(self.train_area_points, "V", self.POINT_COLORS[4])
            return
        groups = self._active_group_store() or []
        prefix = {"markers": "G", "manual_h": "H", "manual_v": "V"}.get(key, "P")
        for group_idx, group in enumerate(groups):
            self._draw_canvas_group(group, f"{prefix}{group_idx + 1}-",
                                    self.POINT_COLORS[group_idx % len(self.POINT_COLORS)])

    def _draw_canvas_group(self, group: List[Point], prefix: str, color: str,
                           close: bool = False) -> None:
        prev = None
        first = None
        for idx, point in enumerate(group):
            x, y = self.image_to_canvas(*point)
            if prev is not None:
                self.canvas.create_line(prev[0], prev[1], x, y, fill=color, width=2)
            if first is None:
                first = (x, y)
            self._draw_point(x, y, f"{prefix}{idx + 1}", color)
            prev = (x, y)
        if close and len(group) >= 3 and first and prev:
            self.canvas.create_line(prev[0], prev[1], first[0], first[1],
                                    fill=color, width=2, dash=(6, 4))

    def _draw_point(self, x: float, y: float, label: str, color: str) -> None:
        r = 6
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=2)
        self.canvas.create_text(x + 10, y - 10, text=label, fill=color, anchor="sw",
                                font=(UI_FONT, 10, "bold"))

    def _refresh_list(self) -> None:
        if not hasattr(self, "annotation_list"):
            return
        self.annotation_list.delete(0, tk.END)
        key = self.current_step.key
        if key == "keypoints":
            labels = list(self.pitch_config.labels)
            for ann in sorted(self.keypoints, key=lambda item: item.vertex_index):
                label = labels[ann.vertex_index] if ann.vertex_index < len(labels) else str(ann.vertex_index)
                self.annotation_list.insert(tk.END, f"{ann.vertex_index}: {label} -> {ann.img_coord}")
            return
        if key == "field_boundary":
            for name in BOUNDARY_UI_ORDER:
                points = self.boundary_edges[name]
                self.annotation_list.insert(
                    tk.END, f"{BOUNDARY_LABELS[name]}：{len(points)}/2")
                for index, point in enumerate(points):
                    self.annotation_list.insert(tk.END, f"  点 {index + 1}: {point}")
            return
        if key == "train_area":
            self.annotation_list.insert(tk.END, f"视频自定义点：{len(self.train_area_points)} 个")
            for point_idx, point in enumerate(self.train_area_points):
                self.annotation_list.insert(tk.END, f"  V{point_idx + 1}: {point}")
            self.annotation_list.insert(tk.END, f"区域点（球场坐标）：{len(self.train_area_region)} 个")
            return
        if key in GROUP_STEPS:
            groups = self._active_group_store() or []
            prefix = {"markers": "G", "manual_h": "H", "manual_v": "V"}.get(key, "G")
            for group_idx, group in enumerate(groups):
                type_label = (f" · {self.marker_equipment_types[group_idx]}"
                              if key == "markers" and group_idx < len(self.marker_equipment_types)
                              else "")
                layout_label = (
                    " · 环型" if self.marker_group_closed[group_idx] else " · 线型"
                ) if key == "markers" and group_idx < len(self.marker_group_closed) else ""
                self.annotation_list.insert(
                    tk.END, f"第 {group_idx + 1} 组{type_label}{layout_label}: {len(group)} 个点")
                for point_idx, point in enumerate(group):
                    self.annotation_list.insert(tk.END, f"  {prefix}{group_idx + 1}-{point_idx + 1}: {point}")

    # ------------------------------------------------------------------
    # 场地参考图
    # ------------------------------------------------------------------

    def open_field_reference(self) -> None:
        selected_name = self.selected_pitch_config.get()
        try:
            config, not_applied = resolve_reference_configuration(
                selected_name, self.pitch_config)
            reference_image = render_numbered_pitch(config)
        except (ValueError, AttributeError, TypeError) as exc:
            messagebox.showerror("场地图编号", f"无法生成当前场地图：{exc}")
            return

        display_name = PITCH_CONFIG_LABELS.get(selected_name, selected_name)
        direction = orientation_hint(config)
        if not_applied:
            direction += " 当前仅预览所选配置；请先“应用并保存配置”再继续标注。"

        window = tk.Toplevel(self)
        window.title(f"场地图编号与方向 · {display_name}")
        window.geometry("1100x820")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)

        zoom_value = tk.DoubleVar(value=1.0)
        photo_ref: Dict[str, Optional[ImageTk.PhotoImage]] = {"photo": None}

        toolbar = ttk.Frame(window, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(toolbar, text=display_name, font=(UI_FONT, 13, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Button(toolbar, text="放大", command=lambda: set_zoom(zoom_value.get() * 1.25)).grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text="缩小", command=lambda: set_zoom(zoom_value.get() / 1.25)).grid(row=0, column=2, padx=3)
        ttk.Button(toolbar, text="适应窗口", command=lambda: fit_to_window()).grid(row=0, column=3, padx=3)
        ttk.Label(toolbar, text=direction, wraplength=1000, justify="left").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        frame = ttk.Frame(window)
        frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        canvas = tk.Canvas(frame, background="#e5e7eb", highlightthickness=0)
        hbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        vbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        def draw_image() -> None:
            image = reference_image
            scale = max(0.2, min(zoom_value.get(), 6.0))
            width = max(1, int(image.width * scale))
            height = max(1, int(image.height * scale))
            resized = image.resize((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            photo_ref["photo"] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.configure(scrollregion=(0, 0, width, height))

        def set_zoom(value: float) -> None:
            zoom_value.set(max(0.2, min(value, 6.0)))
            draw_image()

        def fit_to_window() -> None:
            image = reference_image
            canvas.update_idletasks()
            available_w = max(100, canvas.winfo_width() - 20)
            available_h = max(100, canvas.winfo_height() - 20)
            zoom_value.set(min(available_w / image.width, available_h / image.height, 1.0))
            draw_image()

        def on_wheel(event) -> str:
            if event.delta > 0:
                set_zoom(zoom_value.get() * 1.15)
            else:
                set_zoom(zoom_value.get() / 1.15)
            return "break"

        canvas.bind("<MouseWheel>", on_wheel)
        canvas.bind("<ButtonPress-1>", lambda e: canvas.scan_mark(e.x, e.y))
        canvas.bind("<B1-Motion>", lambda e: canvas.scan_dragto(e.x, e.y, gain=1))
        window.bind("<Control-plus>", lambda _e: set_zoom(zoom_value.get() * 1.25))
        window.bind("<Control-minus>", lambda _e: set_zoom(zoom_value.get() / 1.25))
        window.bind("<Control-0>", lambda _e: fit_to_window())
        window.after(100, fit_to_window)

    # ------------------------------------------------------------------
    # 测量流程
    # ------------------------------------------------------------------

    def open_output_dir(self) -> None:
        target = self._store_dir()
        if not target.exists():
            messagebox.showinfo("打开结果目录", "相机标定结果目录尚不存在，请先保存标注或运行测量。")
            return
        open_path_in_file_manager(str(target))

    def open_measurement_dir(self) -> None:
        target = self._manual_dir()
        if not target.exists():
            messagebox.showinfo("打开器材测量目录", "器材测量目录尚不存在，请先保存器材标注或运行测量。")
            return
        open_path_in_file_manager(str(target))

    def open_step_result_dir(self, key: str) -> None:
        if key not in ("field_boundary", "train_area", "markers"):
            raise ValueError(f"没有对应的结果目录：{key}")
        saved_path = self._saved_preview_paths.get(key)
        json_path = self._step_save_path(key)
        target = saved_path.parent if saved_path else json_path.parent
        if not target.exists():
            messagebox.showinfo("打开结果目录", "结果目录尚不存在，请先生成并保存本步预览图。")
            return
        open_path_in_file_manager(str(target))

    def save_and_run_measurement(self) -> None:
        if not self._confirm_measurement_retry():
            return
        video_text = self.video_path_var.get().strip()
        if not video_text:
            messagebox.showwarning(
                "运行测量", "当前任务尚未关联视频。请在“项目设置”中选择"
                "与标注图片分辨率一致的测量视频，并应用配置。")
            self._select_step_by_key("setup")
            return
        video_frame = self._read_video_frame(Path(video_text), 0)
        if video_frame is None:
            messagebox.showwarning("运行测量", "测量视频无法读取，请重新选择视频。")
            self._select_step_by_key("setup")
            return
        if self.media_kind == "image" and video_frame.shape[:2] != (self.image_h, self.image_w):
            messagebox.showwarning("运行测量", "测量视频与标注图片的分辨率不同，不能复用标注坐标。")
            self._select_step_by_key("setup")
            return
        boundary_mode = bool(self.boundary_enabled_var.get())
        boundary_data = None
        if boundary_mode:
            try:
                boundary_data = self._build_boundary_data()
            except ValueError as exc:
                messagebox.showwarning("运行测量", f"四条边线标定尚未完成：{exc}")
                self._select_step_by_key("field_boundary")
                return
        if not boundary_mode and not self._step_complete("keypoints"):
            messagebox.showwarning("运行测量", "关键点未完成（至少 4 个），无法计算单应矩阵。")
            self._select_step_by_key("keypoints")
            return
        for key, title in (("manual_h", "横向直线标注"), ("manual_v", "纵向直线标注")):
            if not boundary_mode and not self._step_complete(key):
                messagebox.showwarning("运行测量", f"旧模式必选步骤“{title}”尚未完成，请先标注并保存。")
                self._select_step_by_key(key)
                return
        quality_messages = []
        try:
            # Use the annotations currently visible in the editor, not old JSON
            # files. A cancelled warning must not save or start the measurement.
            quality_boundary = boundary_data if boundary_mode else None
            if quality_boundary and self.boundary_source_name:
                quality_boundary = {**quality_boundary, "source": self.boundary_source_name}
            quality_issues = inspect_calibration_quality(
                pitch_config=self.pitch_config,
                image_size=(self.image_w, self.image_h),
                keypoints=[(item.vertex_index, item.img_coord) for item in self.keypoints],
                field_boundary=quality_boundary,
                source_name=self.source_path.name if self.source_path else None,
                keypoint_source_name=self.keypoint_source_name or None,
            )
            quality_messages.extend(issue.message for issue in quality_issues)
        except Exception as exc:
            # A preflight failure is also visible to the operator, but does
            # not silently turn a quality warning into a hard stop.
            quality_messages.append(f"标定质量检查未完成：{exc}")
        if quality_messages:
            details = "\n".join(f"{index}. {message}" for index, message in enumerate(quality_messages, 1))
            if not messagebox.askokcancel(
                "标定质量提醒",
                "检测到以下可能影响效果图的异常：\n\n"
                f"{details}\n\n"
                "点击“确定”继续生成效果图；点击“取消”返回检查标注。",
            ):
                self.status.set("已取消生成，请检查标定提示后再试。")
                return
        if not self.save_all_default():
            messagebox.showwarning("运行测量", "没有可保存的标注，已取消。")
            return

        log_window, log_text = self._open_measurement_log_window()
        log_text.insert(tk.END, f"相机：{self.camera_name}\n")
        log_text.insert(tk.END, f"标志物文件：{self._step_save_path('markers')}\n")
        if quality_messages:
            log_text.insert(tk.END, "标定质量提醒（已确认继续）：\n")
            for index, message in enumerate(quality_messages, 1):
                log_text.insert(tk.END, f"  {index}. {message}\n")
        log_text.insert(tk.END, "开始运行测量流程...\n\n")
        log_text.configure(state="disabled")
        self.status.set("测量运行中…")
        self._measurement_run_id += 1
        run_id = self._measurement_run_id
        cancel_event = threading.Event()
        self._measurement_running = True
        self._measurement_process = None
        self._measurement_result = None
        self._measurement_cancel = cancel_event
        self._measurement_log_window = log_window
        self._measurement_log_text = log_text
        worker = threading.Thread(target=self._run_measurement_worker,
                                  args=(run_id, cancel_event), daemon=True)
        self._measurement_worker = worker
        try:
            worker.start()
        except Exception as exc:
            self._measurement_result = (run_id, f"测量启动失败：{exc}\n", "测量启动失败")
            self._finish_measurement_if_done(run_id)
            return
        self.after(200, lambda: self._poll_measurement(run_id))

    def _open_measurement_log_window(self):
        window = tk.Toplevel(self)
        window.title("测量结果")
        window.geometry("900x620")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        log_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=(MONO_FONT, 10))
        log_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        return window, log_text

    def _append_measurement_log(self, log_text, message: str) -> None:
        log_text.configure(state="normal")
        log_text.insert(tk.END, message)
        log_text.see(tk.END)
        log_text.configure(state="disabled")

    def _measurement_task_active(self) -> bool:
        """Check the worker and child process, not a possibly stale UI flag."""
        worker = self._measurement_worker
        process = self._measurement_process
        return bool((worker is not None and worker.is_alive())
                    or (process is not None and process.poll() is None))

    def _finish_measurement_if_done(self, run_id: int) -> bool:
        """Reconcile finished work on the Tk thread, including abnormal exits."""
        if run_id != self._measurement_run_id or self._measurement_task_active():
            return False
        result = self._measurement_result
        if result is None or result[0] != run_id:
            output, status = "测量任务已结束，但未返回结果；请重新测量。\n", "测量异常结束"
        else:
            _, output, status = result
        self._measurement_running = False
        self._measurement_worker = None
        self._measurement_process = None
        self._measurement_cancel = None
        self._measurement_result = None
        log_window = self._measurement_log_window
        log_text = self._measurement_log_text
        self._measurement_log_window = None
        self._measurement_log_text = None
        if log_window is not None and log_window.winfo_exists():
            self._append_measurement_log(log_text, output)
        self.status.set(status)
        self._refresh_step_list()
        self._refresh_finish_summary()
        if self._measurement_restart_pending:
            self._measurement_restart_pending = False
            self.after_idle(self.save_and_run_measurement)
        return True

    def _poll_measurement(self, run_id: int) -> None:
        if run_id != self._measurement_run_id:
            return
        if not self._finish_measurement_if_done(run_id):
            self.after(200, lambda: self._poll_measurement(run_id))

    def _force_stop_measurement(self, run_id: int) -> None:
        if run_id != self._measurement_run_id or not self._measurement_restart_pending:
            return
        process = self._measurement_process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def _confirm_measurement_retry(self) -> bool:
        if self._measurement_restart_pending:
            messagebox.showinfo("重新测量", "正在结束上一次测量，结束后会自动重新测量。")
            return False
        active = self._measurement_task_active()
        if not active and not self._measurement_running:
            return True
        if not active:
            self._finish_measurement_if_done(self._measurement_run_id)
            return messagebox.askyesno(
                "重新测量", "上一次测量已经结束，但界面状态未及时更新。是否重新测量？")
        if not messagebox.askyesno(
            "重新测量", "当前测量仍在运行。是否结束当前任务并重新测量？\n"
            "当前任务尚未完成的结果可能无法保留。"):
            return False
        # The child may finish while the confirmation dialog is open.
        # In that case there is nothing to cancel, so start the new run now.
        if not self._measurement_task_active():
            if self._measurement_running:
                self._finish_measurement_if_done(self._measurement_run_id)
            return True
        self._measurement_restart_pending = True
        cancel_event = self._measurement_cancel
        if cancel_event is not None:
            cancel_event.set()
        process = self._measurement_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        self.status.set("正在结束当前测量，随后重新测量…")
        run_id = self._measurement_run_id
        self.after(5000, lambda: self._force_stop_measurement(run_id))
        return False

    def _run_measurement_worker(self, run_id: int,
                                cancel_event: threading.Event) -> None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            if cancel_event.is_set():
                output, status = "测量已取消，准备重新测量。\n", "测量已取消"
            else:
                process = subprocess.Popen(
                    [sys.executable, "measure_marker_position.py"],
                    cwd=str(self.project_dir),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                self._measurement_process = process
                if cancel_event.is_set() and process.poll() is None:
                    process.terminate()
                stdout, stderr = process.communicate(input="y\n")
                output = stdout or ""
                if stderr:
                    output += "\n[错误输出]\n" + stderr
                if cancel_event.is_set():
                    output += "\n当前测量已停止，准备重新测量。\n"
                    status = "测量已取消"
                elif process.returncode == 0:
                    output += f"\n测量完成。结果目录：{self._manual_dir()}\n"
                    status = "测量完成"
                else:
                    output += f"\n测量失败，退出码：{process.returncode}\n"
                    status = "测量失败"
        except Exception as exc:
            output = f"测量启动失败：{exc}\n"
            status = "测量启动失败"
        self._measurement_result = (run_id, output, status)


def main() -> None:
    print("[环境检查] 环境就绪，启动标注客户端…")
    app = AnnotationClient()
    app.mainloop()


if __name__ == "__main__":
    main()
