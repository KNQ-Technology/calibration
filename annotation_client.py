"""
标定标注客户端（分步向导版）

按照标定业务流程组织标注步骤：

    1. 项目设置      -> config.yaml
    2. 关键点标注    -> store/{CAMERA_NAME}/keypoints.json
    3. 横向直线标注  -> store/{CAMERA_NAME}/manual_horizontal.json
    4. 纵向直线标注  -> store/{CAMERA_NAME}/manual_vertical.json
    5. 训练区域      -> store/{CAMERA_NAME}/train_area.json
    6. 标志物分组    -> store/manual/{CAMERA_NAME}/markers.json
    7. 保存与测量    -> 运行 measure_marker_position.py

生成的 JSON 与主流程 (camera.py / de_distortion.py / measure_marker_position.py)
完全兼容；启动时自动加载已有标注缓存，保存时直接写入规范路径。

启动时自动检测操作系统（Windows / macOS / Linux）并准备环境：
缺少必需依赖时自动 pip 安装；测量流程依赖缺失时给出提示但不阻塞标注。
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

import soccer

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
        "store/{相机名}/ 与 store/manual/{相机名}/ 下。",
    ),
    StepDef(
        "keypoints",
        "关键点标注",
        "在画面中标注球场关键点（用于计算单应矩阵）。\n\n"
        "1. 在下拉框选择顶点编号（可点「查看场地图编号」对照）\n"
        "2. 左键在画面中点击该关键点的位置\n"
        "3. 标注后自动跳到下一个未标注的顶点\n\n"
        "至少需要 4 个点，点越多、分布越广，标定越准。\n"
        "重复标注同一编号会覆盖旧点。",
    ),
    StepDef(
        "manual_h",
        "横向直线标注",
        "标注画面中真实世界为『横向直线』的线\n"
        "（如上下边线、禁区横线），用于畸变标定。\n\n"
        "每组对应一条直线：沿线均匀点 5 个以上点，\n"
        "然后「新建分组」标注下一条。\n\n"
        "建议至少标注 2 条分布在画面不同位置的横线。",
    ),
    StepDef(
        "manual_v",
        "纵向直线标注",
        "标注画面中真实世界为『纵向直线』的线\n"
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
        "点「生成区域」转换为球场坐标（需先完成关键点标注）。\n\n"
        "区域至少 3 个点。跳过此步则由主流程交互标定。",
        optional=True,
    ),
    StepDef(
        "markers",
        "标志物分组",
        "标注要测量的标志物（如障碍桶）。\n\n"
        "每组代表一类标志物（如一排桶）：\n"
        "左键添加点，「新建分组」开始下一组。\n\n"
        "组内/组间距离的计算方式由 config.yaml 中的\n"
        "GROUP_CONFIGS / PAIR_CONFIGS 决定。",
    ),
    StepDef(
        "finish",
        "保存与测量",
        "检查各步骤完成情况，一键保存全部标注，\n"
        "并运行测量流程生成报告与可视化。\n\n"
        "注意：畸变参数由主流程自动计算；\n"
        "若重新标注了关键点或横纵线，请先清理派生缓存，\n"
        "否则主流程会直接使用旧的畸变参数。",
    ),
]

STEP_KEYS = [s.key for s in STEP_DEFS]
GROUP_STEPS = {"manual_h", "manual_v", "markers"}


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
        initial_config = self.config_data.get("config", "QingYuanConfiguration")
        self.selected_pitch_config = tk.StringVar(value=initial_config)
        self.pitch_config = self._load_pitch_config(initial_config)
        self.camera_name = self.config_data.get("CAMERA_NAME") or "camera1"

        self.status = tk.StringVar(value="就绪")
        self.frame_index = tk.IntVar(value=0)
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.source_path: Optional[Path] = None
        self.frame_bgr = self._load_default_frame()
        self.image_h, self.image_w = self.frame_bgr.shape[:2]

        # ---- 标注数据 ----
        self.keypoints: List[KeypointAnnotation] = []
        self.marker_groups: List[List[Point]] = [[]]
        self.manual_h_groups: List[List[Point]] = [[]]
        self.manual_v_groups: List[List[Point]] = [[]]
        self.train_area_points: List[Point] = []       # 视频画面自定义点 V1..Vn
        self.train_area_region: List[List[float]] = []  # 球场坐标区域点 (cm)
        self.train_area_tokens = tk.StringVar(value="")

        self.current_step_index = 0
        self.current_group = tk.IntVar(value=1)
        self.selected_vertex = tk.IntVar(value=0)

        # 项目设置变量
        self.video_path_var = tk.StringVar(value=str(self._default_video_path()))
        self.camera_name_var = tk.StringVar(value=self.camera_name)
        self.store_root = self.config_data.get("store_dir") or "store"
        self.store_root_var = tk.StringVar(value=self.store_root)

        # 测量配置变量
        self.num_marker_groups = tk.IntVar(value=2)
        self.group_config_widgets: List[Dict[str, Any]] = []
        self.pair_enabled_vars: Dict[Tuple[int, int], tk.BooleanVar] = {}
        self.pair_type_vars: Dict[Tuple[int, int], tk.StringVar] = {}
        self._group_closed_vars: Dict[int, tk.BooleanVar] = {}
        self._build_measurement_config_defaults()
        self.num_marker_groups.trace_add("write", lambda *_: self._on_num_groups_changed())

        self._load_existing_annotations()
        self._build_ui()
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
        names = []
        for name in dir(soccer):
            config_cls = getattr(soccer, name)
            if name.endswith("Configuration") and isinstance(config_cls, type):
                names.append(name)
        preferred = [
            "QingYuanConfiguration",
            "HalfQingYuanConfiguration",
            "WuChuanConfiguration",
            "HalfWuChuanConfiguration",
            "SoccerPitchConfiguration",
            "HalfPitchConfiguration",
        ]
        ordered = [name for name in preferred if name in names]
        ordered.extend(sorted(name for name in names if name not in ordered))
        return ordered

    def _load_pitch_config(self, class_name: str) -> Any:
        config_cls = getattr(soccer, class_name, soccer.QingYuanConfiguration)
        return config_cls()

    def _default_video_path(self) -> Path:
        dir_path = self.config_data.get("dir_path") or "."
        video_name = self.config_data.get("video_name") or ""
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
            "train_area": self._store_dir() / "train_area.json",
            "markers": self._manual_dir() / "markers.json",
        }.get(key)

    # ------------------------------------------------------------------
    # 帧加载
    # ------------------------------------------------------------------

    def _load_default_frame(self):
        video_path = self._default_video_path()
        self.source_path = video_path
        frame = self._read_video_frame(video_path, self.frame_index.get())
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

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(self.project_dir),
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self.source_path = Path(path)
        self.video_path_var.set(path)
        self.load_frame_at_index()

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(self.project_dir),
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror("打开图片", "图片打开失败。")
            return
        self.source_path = Path(path)
        self.set_frame(frame)

    def load_frame_at_index(self) -> None:
        if not self.source_path:
            self.source_path = self._default_video_path()
        frame = self._read_video_frame(self.source_path, self.frame_index.get())
        if frame is None:
            messagebox.showerror("加载帧", "无法从视频加载该帧。")
            return
        self.set_frame(frame)
        self.status.set(f"已加载帧 {self.frame_index.get()}，来源：{self.source_path.name}")

    def set_frame(self, frame) -> None:
        self.frame_bgr = frame
        self.image_h, self.image_w = frame.shape[:2]
        self._fit_image()
        self._redraw()

    # ------------------------------------------------------------------
    # 已有标注自动加载
    # ------------------------------------------------------------------

    def _load_existing_annotations(self) -> None:
        loaded = []
        path = self._step_save_path("keypoints")
        if path and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.keypoints = [
                    KeypointAnnotation(int(item["vertex_index"]),
                                       (int(item["img_coord"][0]), int(item["img_coord"][1])))
                    for item in data.get("annotations", [])
                ]
                loaded.append(f"关键点 {len(self.keypoints)} 个")
            except Exception:
                pass

        for key, attr in (("manual_h", "manual_h_groups"),
                          ("manual_v", "manual_v_groups"),
                          ("markers", "marker_groups")):
            path = self._step_save_path(key)
            if path and path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    groups = []
                    for group in sorted(data.get("groups", []), key=lambda g: g.get("group_id", 0)):
                        pts = [(int(p["img_coord"][0]), int(p["img_coord"][1]))
                               for p in group.get("points", [])]
                        groups.append(pts)
                    if groups:
                        setattr(self, attr, groups)
                        loaded.append(f"{key} {len(groups)} 组")
                except Exception:
                    pass

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

        if loaded:
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
        if key == "train_area":
            return len(self.train_area_region) >= 3
        if key == "markers":
            return any(g for g in self.marker_groups)
        if key == "finish":
            return (self._manual_dir() / "measurement_report.txt").exists()
        return False

    def _first_incomplete_step(self) -> int:
        for i, step in enumerate(STEP_DEFS):
            if step.key == "finish":
                return i
            if step.optional:
                continue
            if not self._step_complete(step.key):
                return i
        return len(STEP_DEFS) - 1

    def _refresh_step_list(self) -> None:
        if not hasattr(self, "step_listbox"):
            return
        selection = self.current_step_index
        self.step_listbox.delete(0, tk.END)
        for i, step in enumerate(STEP_DEFS):
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
            left, height=len(STEP_DEFS), exportselection=False,
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
            cache_box, height=8, width=28, relief="flat",
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
        ttk.Entry(toolbar, textvariable=self.frame_index, width=8).grid(row=0, column=3, padx=4)
        ttk.Button(toolbar, text="加载帧", command=self.load_frame_at_index).grid(row=0, column=4, padx=4)
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
        self._build_train_area_panel(self.step_panels["train_area"])
        self._build_groups_panel(self.step_panels["markers"], "标志物")
        self._build_finish_panel(self.step_panels["finish"])

        # 通用操作
        self.common_actions = ttk.LabelFrame(self.side, text="操作", padding=8)
        self.common_actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for i in range(2):
            self.common_actions.columnconfigure(i, weight=1)
        ttk.Button(self.common_actions, text="撤销", command=self.undo).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(self.common_actions, text="清空本步", command=self.clear_current).grid(
            row=0, column=1, sticky="ew", padx=(4, 0))
        self.save_step_button = ttk.Button(self.common_actions, text="保存本步",
                                           command=self.save_current_step)
        self.save_step_button.grid(row=1, column=0, sticky="ew", pady=(8, 0), padx=(0, 4))
        ttk.Button(self.common_actions, text="生成预览图", command=self.save_preview).grid(
            row=1, column=1, sticky="ew", pady=(8, 0), padx=(4, 0))

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
        ttk.Button(box, text="浏览视频…", command=self.open_video).grid(row=2, column=0, sticky="ew", pady=(4, 8))

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
            textvariable=self.selected_pitch_config,
            values=self.pitch_config_names,
            state="readonly",
        )
        self.pitch_config_combo.grid(row=10, column=0, sticky="ew", pady=(0, 8))

        # ---- 标志物测量配置 ----
        measure_box = ttk.LabelFrame(panel, text="标志物测量配置", padding=8)
        measure_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        measure_box.columnconfigure(0, weight=1)

        # 组数选择
        groups_header = ttk.Frame(measure_box)
        groups_header.grid(row=0, column=0, sticky="ew")
        groups_header.columnconfigure(2, weight=1)
        ttk.Label(groups_header, text="器材组数").grid(row=0, column=0, sticky="w")
        self.num_groups_spin = ttk.Spinbox(
            groups_header, from_=1, to=9, width=5,
            textvariable=self.num_marker_groups,
            command=self._on_num_groups_changed,
        )
        self.num_groups_spin.grid(row=0, column=1, padx=(8, 0), sticky="w")

        # 每组配置区域（动态生成）
        self.group_configs_frame = ttk.Frame(measure_box)
        self.group_configs_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.group_configs_frame.columnconfigure(0, weight=1)

        # 组间距离配置区域
        pair_header = ttk.Frame(measure_box)
        pair_header.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(pair_header, text="组间距离测量", font=(UI_FONT, 10, "bold")).grid(
            row=0, column=0, sticky="w")
        self.pair_configs_frame = ttk.Frame(measure_box)
        self.pair_configs_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.pair_configs_frame.columnconfigure(0, weight=1)

        ttk.Button(box, text="应用并保存配置 (config.yaml)",
                   command=self.apply_setup).grid(row=11, column=0, sticky="ew")
        self._refresh_store_dir_hint()
        self._rebuild_group_configs()
        self._rebuild_pair_configs()

    # ------------------------------------------------------------------
    # 测量配置 UI 辅助方法
    # ------------------------------------------------------------------

    def _build_measurement_config_defaults(self) -> None:
        """从 config.yaml 加载 GROUP_CONFIGS / PAIR_CONFIGS 初始化变量"""
        group_cfgs = self.config_data.get("GROUP_CONFIGS") or [
            {"group": 0, "is_closed": False}, {"group": 1, "is_closed": False}
        ]
        pair_cfgs = self.config_data.get("PAIR_CONFIGS") or []

        max_g = 0
        self._group_closed_vars: Dict[int, tk.BooleanVar] = {}
        for gc in group_cfgs:
            g = gc.get("group", 0)
            max_g = max(max_g, g)
            self._group_closed_vars[g] = tk.BooleanVar(value=gc.get("is_closed", False))
        self.num_marker_groups.set(max(max_g + 1, 2))

        for pc in pair_cfgs:
            g1 = pc.get("group1", 0)
            g2 = pc.get("group2", 1)
            key = (g1, g2)
            self.pair_enabled_vars[key] = tk.BooleanVar(value=True)
            self.pair_type_vars[key] = tk.StringVar(value=pc.get("type", "both"))

    def _on_num_groups_changed(self) -> None:
        """组数改变时，动态重建每组配置和组间配置"""
        self._rebuild_group_configs()
        self._rebuild_pair_configs()

    def _rebuild_group_configs(self) -> None:
        """重建每组的配置行（环形/线形复选框）"""
        if not hasattr(self, "group_configs_frame"):
            return
        for w in self.group_configs_frame.winfo_children():
            w.destroy()
        self.group_config_widgets.clear()

        n = max(1, int(self.num_marker_groups.get()))
        for i in range(n):
            if i not in self._group_closed_vars:
                self._group_closed_vars[i] = tk.BooleanVar(value=False)
            row = ttk.Frame(self.group_configs_frame)
            row.grid(row=i, column=0, sticky="ew", pady=1)
            ttk.Label(row, text=f"第 {i+1} 组").grid(row=0, column=0, sticky="w")
            closed_var = self._group_closed_vars[i]
            cb = ttk.Checkbutton(row, text="环形（闭合）", variable=closed_var)
            cb.grid(row=0, column=1, padx=(12, 0), sticky="w")
            self.group_config_widgets.append({"frame": row, "closed_var": closed_var})

    def _rebuild_pair_configs(self) -> None:
        """重建组间距离配置行"""
        if not hasattr(self, "pair_configs_frame"):
            return
        for w in self.pair_configs_frame.winfo_children():
            w.destroy()

        n = max(1, int(self.num_marker_groups.get()))
        row_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                if key not in self.pair_enabled_vars:
                    self.pair_enabled_vars[key] = tk.BooleanVar(value=False)
                if key not in self.pair_type_vars:
                    self.pair_type_vars[key] = tk.StringVar(value="center")
                enabled_var = self.pair_enabled_vars[key]
                type_var = self.pair_type_vars[key]

                row = ttk.Frame(self.pair_configs_frame)
                row.grid(row=row_idx, column=0, sticky="ew", pady=1)
                ttk.Checkbutton(row, text=f"组{i+1} ↔ 组{j+1}",
                                variable=enabled_var).grid(row=0, column=0, sticky="w")
                ttk.Label(row, text="  方式:").grid(row=0, column=1, sticky="w")
                cb = ttk.Combobox(row, textvariable=type_var, width=10, state="readonly",
                                  values=["center", "nearest", "both"])
                cb.grid(row=0, column=2, padx=(4, 0), sticky="w")
                row_idx += 1

    def _collect_group_configs(self) -> List[Dict[str, Any]]:
        """从 GUI 收集 GROUP_CONFIGS"""
        configs = []
        n = max(1, int(self.num_marker_groups.get()))
        for i in range(n):
            is_closed = False
            if i in self._group_closed_vars:
                is_closed = bool(self._group_closed_vars[i].get())
            configs.append({"group": i, "is_closed": is_closed})
        return configs

    def _collect_pair_configs(self) -> List[Dict[str, Any]]:
        """从 GUI 收集 PAIR_CONFIGS"""
        configs = []
        n = max(1, int(self.num_marker_groups.get()))
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                enabled = self.pair_enabled_vars.get(key)
                if enabled and enabled.get():
                    ptype = self.pair_type_vars.get(key)
                    ptype_val = ptype.get() if ptype else "center"
                    configs.append({"group1": i, "group2": j, "type": ptype_val})
        return configs

    def browse_store_root(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self._resolve_store_root().parent
                                                      if self._resolve_store_root().exists()
                                                      else self.project_dir))
        if path:
            self.store_root_var.set(path)

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

    # ------------------------------------------------------------------
    # 步骤切换
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> StepDef:
        return STEP_DEFS[self.current_step_index]

    def _on_step_listbox(self, _event=None) -> None:
        selection = self.step_listbox.curselection()
        if selection:
            self._select_step(selection[0])

    def prev_step(self) -> None:
        self._select_step(max(0, self.current_step_index - 1))

    def next_step(self) -> None:
        step = self.current_step
        # 离开可标注步骤时提示未保存的必要条件
        if step.key == "keypoints" and 0 < len(self.keypoints) < 4:
            messagebox.showwarning("关键点", "关键点少于 4 个，单应矩阵无法计算，请继续标注。")
            return
        self._select_step(min(len(STEP_DEFS) - 1, self.current_step_index + 1))

    def _select_step(self, index: int) -> None:
        self.current_step_index = max(0, min(index, len(STEP_DEFS) - 1))
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
        self.current_group.set(1)
        if step.key == "finish":
            self._refresh_finish_summary()
        if step.key == "keypoints":
            self._refresh_keypoint_progress()
        self._refresh_step_list()
        self._redraw()

    def _refresh_finish_summary(self) -> None:
        if not hasattr(self, "finish_summary"):
            return
        lines = []
        for i, step in enumerate(STEP_DEFS[:-1]):
            done = self._step_complete(step.key)
            icon = "✔" if done else ("△ 可选" if step.optional else "✘")
            lines.append(f"{icon}  {i + 1}. {step.title}")
        distortion = self._store_dir() / "distortion_params.json"
        lines.append("")
        lines.append("畸变参数：" + ("已缓存（如重标过直线请清理）" if distortion.exists() else "将由主流程自动计算"))
        self.finish_summary.configure(text="\n".join(lines))

    # ------------------------------------------------------------------
    # 项目设置
    # ------------------------------------------------------------------

    def apply_setup(self) -> None:
        new_name = self.camera_name_var.get().strip()
        if not new_name:
            messagebox.showwarning("项目设置", "相机名称不能为空。")
            return
        video = Path(self.video_path_var.get().strip())
        if not video.exists():
            messagebox.showwarning("项目设置", f"视频文件不存在：\n{video}")
            return
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
        self.camera_name = new_name
        self.store_root = new_root
        self.source_path = video

        config_name = self.selected_pitch_config.get()
        self.pitch_config = self._load_pitch_config(config_name)
        self._refresh_vertex_options()

        self._write_config_yaml()
        self._refresh_store_dir_hint()
        if name_changed or root_changed:
            self._load_existing_annotations()
        frame = self._read_video_frame(video, self.frame_index.get())
        if frame is not None:
            self.set_frame(frame)
        self._refresh_step_list()
        self._redraw()
        self.status.set(f"配置已保存：相机 {self.camera_name}，场地 {config_name}")

    def _write_config_yaml(self) -> None:
        video = Path(self.video_path_var.get().strip() or self._default_video_path())
        data = self._load_config()
        data["dir_path"] = str(video.parent)
        data["video_name"] = video.name
        data["CAMERA_NAME"] = self.camera_name
        data["config"] = self.selected_pitch_config.get()
        data["store_dir"] = self.store_root
        data.setdefault("MCS_NAME", None)
        data["GROUP_CONFIGS"] = self._collect_group_configs()
        data["PAIR_CONFIGS"] = self._collect_pair_configs()
        data.setdefault("jac", "3-point")
        data.setdefault("reprojection_eps", 5.0)
        data.setdefault("enable_orientation", True)
        with (self.project_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

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
            self.status.set(f"已标注关键点 {vertex_index}: {point}")
            self.next_unannotated_vertex()
            self._refresh_keypoint_progress()
        elif key == "train_area":
            self.train_area_points.append(point)
            self.status.set(f"已添加自定义点 V{len(self.train_area_points)}: {point}")
        else:
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

    def ensure_group(self) -> int:
        group_number = max(1, int(self.current_group.get()))
        groups = self._active_group_store()
        if groups is not None:
            while len(groups) < group_number:
                groups.append([])
        self.current_group.set(group_number)
        return group_number

    def new_group(self) -> None:
        groups = self._active_group_store()
        if groups is not None:
            groups.append([])
            self.current_group.set(len(groups))
        self._redraw()

    def undo(self) -> None:
        key = self.current_step.key
        if key == "keypoints":
            if self.keypoints:
                removed = self.keypoints.pop()
                self.status.set(f"已撤销关键点 {removed.vertex_index}")
                self._refresh_keypoint_progress()
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
            self._refresh_keypoint_progress()
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

    def save_current_step(self) -> None:
        key = self.current_step.key
        path = self._step_save_path(key)
        if path is None:
            return
        try:
            data = self._build_step_data(key)
        except ValueError as exc:
            messagebox.showwarning("保存", str(exc))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status.set(f"已保存 {path}")
        self._refresh_step_list()
        if key in ("keypoints", "manual_h", "manual_v"):
            self._maybe_invalidate_derived_cache()

    def _build_step_data(self, key: str) -> Dict[str, Any]:
        if key == "keypoints":
            if len(self.keypoints) < 4:
                raise ValueError(f"关键点至少需要 4 个，当前只有 {len(self.keypoints)} 个。")
            return self._build_keypoints_data()
        if key == "manual_h":
            return self._build_groups_data(self.manual_h_groups, "manual_horizontal", min_points=2)
        if key == "manual_v":
            return self._build_groups_data(self.manual_v_groups, "manual_vertical", min_points=2)
        if key == "markers":
            return self._build_groups_data(self.marker_groups, "markers", min_points=1)
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
        self._write_config_yaml()
        saved, skipped = [], []
        for key in ("keypoints", "manual_h", "manual_v", "train_area", "markers"):
            path = self._step_save_path(key)
            try:
                data = self._build_step_data(key)
            except ValueError:
                skipped.append(key)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            saved.append(path.name)
        self._refresh_step_list()
        if saved:
            self.status.set("已保存：" + "、".join(saved))
        if skipped:
            names = {"keypoints": "关键点", "manual_h": "横线", "manual_v": "纵线",
                     "train_area": "训练区域", "markers": "标志物"}
            self.status.set(self.status.get() + "；跳过：" + "、".join(names[k] for k in skipped))
        return bool(saved)

    def _maybe_invalidate_derived_cache(self) -> None:
        derived = [
            self._store_dir() / "distortion_params.json",
            self._store_dir() / "camera_info.json",
        ]
        existing = [p for p in derived if p.exists()]
        if not existing:
            return
        names = "\n".join(f"  - {p.name}" for p in existing)
        if messagebox.askyesno(
            "派生缓存已过期",
            "关键点/直线标注已更新，以下缓存基于旧标注计算：\n"
            f"{names}\n\n是否删除，让主流程重新计算？（推荐）",
        ):
            for p in existing:
                try:
                    p.unlink()
                except OSError as exc:
                    messagebox.showwarning("清理缓存", f"删除 {p.name} 失败：{exc}")
            self._refresh_step_list()
            self.status.set("已清理派生缓存，主流程将重新计算畸变参数。")

    def clear_derived_cache(self) -> None:
        derived = [
            self._store_dir() / "distortion_params.json",
            self._store_dir() / "camera_info.json",
        ]
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
                self.annotation_list.insert(tk.END, f"第 {group_idx + 1} 组: {len(group)} 个点")
                for point_idx, point in enumerate(group):
                    self.annotation_list.insert(tk.END, f"  {prefix}{group_idx + 1}-{point_idx + 1}: {point}")

    # ------------------------------------------------------------------
    # 场地参考图
    # ------------------------------------------------------------------

    def _reference_images(self) -> List[Tuple[str, Path]]:
        ref_dir = self.project_dir / "images" / "calibration_reference"
        return [
            ("吴川半场", ref_dir / "01_吴川半场.png"),
            ("清源半场", ref_dir / "02_清源半场.png"),
            ("清源全场", ref_dir / "03_清源全场.png"),
            ("完整说明", ref_dir / "04_完整说明.png"),
        ]

    def open_field_reference(self) -> None:
        references = [(name, path) for name, path in self._reference_images() if path.exists()]
        if not references:
            messagebox.showwarning("场地图编号", "没有找到场地标定参考图。")
            return

        window = tk.Toplevel(self)
        window.title("场地标定图编号参考")
        window.geometry("1100x820")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)

        selected = tk.StringVar(value=references[0][0])
        zoom_value = tk.DoubleVar(value=1.0)
        original_image: Dict[str, Optional[Image.Image]] = {"image": None}
        photo_ref: Dict[str, Optional[ImageTk.PhotoImage]] = {"photo": None}

        toolbar = ttk.Frame(window, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)
        combo = ttk.Combobox(toolbar, textvariable=selected,
                             values=[name for name, _ in references], state="readonly")
        combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(toolbar, text="放大", command=lambda: set_zoom(zoom_value.get() * 1.25)).grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text="缩小", command=lambda: set_zoom(zoom_value.get() / 1.25)).grid(row=0, column=2, padx=3)
        ttk.Button(toolbar, text="适应窗口", command=lambda: fit_to_window()).grid(row=0, column=3, padx=3)

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
            image = original_image["image"]
            if image is None:
                return
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
            image = original_image["image"]
            if image is None:
                return
            canvas.update_idletasks()
            available_w = max(100, canvas.winfo_width() - 20)
            available_h = max(100, canvas.winfo_height() - 20)
            zoom_value.set(min(available_w / image.width, available_h / image.height, 1.0))
            draw_image()

        def show_selected(_event=None) -> None:
            ref_path = dict(references)[selected.get()]
            original_image["image"] = Image.open(ref_path).convert("RGB")
            zoom_value.set(1.0)
            fit_to_window()

        def on_wheel(event) -> str:
            if event.delta > 0:
                set_zoom(zoom_value.get() * 1.15)
            else:
                set_zoom(zoom_value.get() / 1.15)
            return "break"

        combo.bind("<<ComboboxSelected>>", show_selected)
        canvas.bind("<MouseWheel>", on_wheel)
        canvas.bind("<ButtonPress-1>", lambda e: canvas.scan_mark(e.x, e.y))
        canvas.bind("<B1-Motion>", lambda e: canvas.scan_dragto(e.x, e.y, gain=1))
        window.bind("<Control-plus>", lambda _e: set_zoom(zoom_value.get() * 1.25))
        window.bind("<Control-minus>", lambda _e: set_zoom(zoom_value.get() / 1.25))
        window.bind("<Control-0>", lambda _e: fit_to_window())
        window.after(100, show_selected)

    # ------------------------------------------------------------------
    # 测量流程
    # ------------------------------------------------------------------

    def open_output_dir(self) -> None:
        target = self._manual_dir()
        if not target.exists():
            target = self._store_dir()
        if not target.exists():
            messagebox.showinfo("打开结果目录", "结果目录尚不存在，请先保存标注或运行测量。")
            return
        open_path_in_file_manager(str(target))

    def save_and_run_measurement(self) -> None:
        if not self._step_complete("keypoints"):
            messagebox.showwarning("运行测量", "关键点未完成（至少 4 个），无法计算单应矩阵。")
            self._select_step(STEP_KEYS.index("keypoints"))
            return
        if not self._step_complete("markers"):
            messagebox.showwarning("运行测量", "还没有标注任何标志物，无法测量。")
            self._select_step(STEP_KEYS.index("markers"))
            return
        if not (self._step_complete("manual_h") and self._step_complete("manual_v")):
            if not messagebox.askyesno(
                "运行测量",
                "横向/纵向直线未标注完整，畸变标定精度会受影响。\n仍要继续吗？",
            ):
                return
        if not self.save_all_default():
            messagebox.showwarning("运行测量", "没有可保存的标注，已取消。")
            return

        log_window, log_text = self._open_measurement_log_window()
        log_text.insert(tk.END, f"相机：{self.camera_name}\n")
        log_text.insert(tk.END, f"标志物文件：{self._step_save_path('markers')}\n")
        log_text.insert(tk.END, "开始运行测量流程...\n\n")
        log_text.configure(state="disabled")
        self.status.set("测量运行中…")
        worker = threading.Thread(target=self._run_measurement_worker,
                                  args=(log_window, log_text), daemon=True)
        worker.start()

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

    def _run_measurement_worker(self, log_window, log_text) -> None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(
                [sys.executable, "measure_marker_position.py"],
                cwd=str(self.project_dir),
                input="y\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[错误输出]\n" + result.stderr
            if result.returncode == 0:
                output += f"\n测量完成。结果目录：store/manual/{self.camera_name}\n"
                status = "测量完成"
            else:
                output += f"\n测量失败，退出码：{result.returncode}\n"
                status = "测量失败"
        except Exception as exc:
            output = f"测量启动失败：{exc}\n"
            status = "测量启动失败"

        def update_ui() -> None:
            if log_window.winfo_exists():
                self._append_measurement_log(log_text, output)
            self.status.set(status)
            self._refresh_step_list()
            self._refresh_finish_summary()

        self.after(0, update_ui)


def main() -> None:
    print("[环境检查] 环境就绪，启动标注客户端…")
    app = AnnotationClient()
    app.mainloop()


if __name__ == "__main__":
    main()
