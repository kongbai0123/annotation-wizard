"""Desktop interface for annotation by box and foreground/background prompts."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import ceil, floor
from pathlib import Path
from queue import Empty
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from project import Project
from mask_cleanup import clean_mask
from segmentation import SegmentationEngine
from video_dialog import VIDEO_TYPES, VideoImport, VideoReviewDialog


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "sam2.1-hiera-tiny"
ENGINES = {"本機 SAM 2.1": "sam2", "GrabCut": "grabcut"}
CLEANUP = {"平衡": "balanced", "輕微": "light", "關閉": "off"}
BOX_MODE, MASK_MODE = "偵測框", "精細遮罩"
PROMPT_TOOLS = (("框選 (B)", "box"), ("前景點 (P)", "positive"), ("背景點 (N)", "negative"))
REPAIR_TOOLS = (("填洞", "fill"), ("移除孤島", "island"), ("加畫", "paint"), ("擦除", "erase"))
FONT = "Microsoft JhengHei UI"
BG, PANEL, WORKSPACE, TEXT = "#101820", "#18242c", "#0a1017", "#ebf2f3"
MUTED, BORDER, ACCENT = "#a8bbc3", "#31424e", "#94e7d4"
POSITIVE, NEGATIVE, BOX_COLOR, CANDIDATE = "#a4efd6", "#ff9abb", "#83c7ff", "#ffba66"
COLORS = ("#6bddcb", "#8db1ff", "#dba6ff", "#f5d27d", "#9fe791", "#ff9ab9")


@dataclass(frozen=True)
class Preview:
    mask: np.ndarray
    label: str
    source: str
    score: float | None
    elapsed_ms: float
    revision: int
    raw_mask: np.ndarray
    base_mask: np.ndarray
    cleanup_stats: dict
    replaces_id: str | None = None
    expected_counts: list | None = None


@dataclass
class BrushStroke:
    before: np.ndarray
    working: np.ndarray
    previous: tuple[int, int]
    value: int
    radius: int


class WizardApp:
    def __init__(self, root: tk.Tk, model_dir: Path = MODEL_DIR) -> None:
        self.root = root
        self.root.title("標註精靈")
        icon = ROOT / "launcher" / "annotation.ico"
        if icon.is_file():
            try:
                self.root.iconbitmap(default=str(icon))
            except tk.TclError:
                pass
        self.root.minsize(1040, 640)
        width = min(1410, max(1050, root.winfo_screenwidth() - 76))
        height = min(900, max(660, root.winfo_screenheight() - 100))
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.project: Project | None = None
        self.directory: Path | None = None
        self.asset: dict | None = None
        self.image: Image.Image | None = None
        self.preview: Preview | None = None
        self.mask_layers: list[tuple[dict, Image.Image]] = []
        self.preview_layer: Image.Image | None = None
        self.raw_layer: Image.Image | None = None
        self.repair_history = []
        self.brush_stroke: BrushStroke | None = None
        self.manual_pixels = 0
        self.selected_annotation: str | None = None
        self.edit_box_id: str | None = None
        self.expected_bbox: list | None = None
        self.prompt_history = []
        self.box: tuple[float, float, float, float] | None = None
        self.positive: list[tuple] = []
        self.negative: list[tuple] = []
        self.revision = 0
        self.drag_box = None
        self.pan_drag = None
        self.zoom, self.pan = 1.0, (0.0, 0.0)
        self.scale, self.origin = 1.0, (0.0, 0.0)
        self.photo = None
        self.draw_job = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annotation")
        self.task: Future | None = None
        self.task_kind = None
        self.video_import: VideoImport | None = None
        self.alive = True
        self.closing = False
        self.engine = SegmentationEngine(model_dir)
        self.sam_available = all((model_dir / filename).is_file() for filename in (
            "config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"
        ))
        self.class_name = tk.StringVar()
        self.method = tk.StringVar(value="本機 SAM 2.1" if self.sam_available else "GrabCut")
        self.tool = tk.StringVar(value="box")
        self.mode = tk.StringVar(value=BOX_MODE)
        self.current_mode = BOX_MODE
        self.show_masks = tk.BooleanVar(value=True)
        self.show_raw = tk.BooleanVar(value=False)
        self.cleanup = tk.StringVar(value="平衡")
        self._theme()
        self._layout()
        self._update_controls()

    def _theme(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=(FONT, 10), background=PANEL, foreground=TEXT,
                        fieldbackground=WORKSPACE, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        style.configure("TButton", padding=(10, 7), anchor="center")
        style.configure("Primary.TButton", background=ACCENT, foreground=WORKSPACE,
                        bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT, font=(FONT, 10, "bold"))
        style.map("TButton", background=[("pressed", "#354957"), ("active", "#2a3c4a")],
                  foreground=[("disabled", "#70848e")])
        style.map("Primary.TButton", background=[("disabled", "#253b42"), ("pressed", "#78cbbb"),
                                                  ("active", "#b2f7e6")],
                  foreground=[("disabled", "#83999e")])
        style.configure("Tool.TRadiobutton", padding=(8, 5), indicatorcolor=WORKSPACE)
        style.map("Tool.TRadiobutton", indicatorcolor=[("selected", ACCENT)])
        style.configure("Repair.TRadiobutton", padding=(3, 3), indicatorcolor=WORKSPACE, font=(FONT, 9))
        style.map("Repair.TRadiobutton", indicatorcolor=[("selected", ACCENT)])
        style.map("TCombobox", fieldbackground=[("readonly", WORKSPACE)], selectbackground=[("readonly", WORKSPACE)],
                  selectforeground=[("readonly", TEXT)])
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        borderwidth=0, rowheight=29)
        style.configure("Treeview.Heading", background=PANEL, foreground=MUTED, padding=(5, 6), font=(FONT, 9, "bold"))
        style.map("Treeview", background=[("selected", "#284652")])

    def _label(self, parent, text="", *, size=10, bold=False, fg=TEXT, bg=PANEL, **kwargs):
        return tk.Label(parent, text=text, font=(FONT, size, "bold" if bold else "normal"),
                        fg=fg, bg=bg, anchor="w", justify="left", **kwargs)

    def _button(self, parent, text, command, *, primary=False):
        return ttk.Button(parent, text=text, command=command, style="Primary.TButton" if primary else "TButton")

    def _panel(self, parent, color=PANEL):
        return tk.Frame(parent, bg=color, highlightbackground=BORDER, highlightthickness=1)

    def _layout(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=20, pady=(15, 12))
        title = tk.Frame(header, bg=BG)
        title.pack(side="left")
        self._label(title, "標註精靈", size=20, bold=True, bg=BG).pack(fill="x")
        self.project_title = self._label(title, "尚未開啟專案", fg=MUTED, bg=BG)
        self.project_title.pack(fill="x")
        actions = tk.Frame(header, bg=BG)
        actions.pack(side="right")
        self.new_button = self._button(actions, "新建專案", self.new_project)
        self.open_button = self._button(actions, "開啟專案", self.open_project)
        self.export_coco_button = self._button(actions, "匯出 COCO", self.export)
        self.export_button = self._button(actions, "匯出 YOLO", lambda: self.export("yolo"), primary=True)
        for button in (self.new_button, self.open_button, self.export_coco_button, self.export_button):
            button.pack(side="left", padx=(7, 0))

        area = tk.Frame(self.root, bg=BG)
        area.pack(fill="both", expand=True, padx=20)
        area.grid_columnconfigure(0, minsize=210)
        area.grid_columnconfigure(1, weight=1, minsize=450)
        area.grid_columnconfigure(2, minsize=290)
        area.grid_rowconfigure(0, weight=1)
        assets = self._panel(area)
        assets.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        assets.configure(width=210)
        assets.pack_propagate(False)
        center = self._panel(area, WORKSPACE)
        center.grid(row=0, column=1, sticky="nsew")
        right = self._panel(area)
        right.grid(row=0, column=2, sticky="nsew", padx=(9, 0))
        right.configure(width=290)
        right.pack_propagate(False)
        self._asset_panel(assets)
        self._canvas_panel(center)
        self._annotation_panel(right)

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=20, pady=(8, 11))
        self.status = self._label(footer, "建立或開啟專案以開始標註。", fg=MUTED, bg=BG)
        self.status.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=95)
        self.video_stop = self._button(footer, "取消影片處理", self.cancel_video)
        self.root.bind("<Control-z>", lambda event: self._key(event, self._undo))
        self.root.bind("<Escape>", lambda event: self._key(event, self._discard_current))
        self.root.bind("<Return>", lambda event: self._key(event, self.save_box) if self._box_mode() else None)
        self.root.bind("<Left>", lambda event: self._key(event, lambda: self.navigate(-1)))
        self.root.bind("<Right>", lambda event: self._key(event, lambda: self.navigate(1)))
        self.root.bind("<KeyPress>", self._tool_key, add="+")

    def _asset_panel(self, parent):
        self._label(parent, "圖片", size=13, bold=True).pack(fill="x", padx=12, pady=(11, 2))
        self.asset_summary = self._label(parent, "0 張圖片", fg=MUTED)
        self.asset_summary.pack(fill="x", padx=12, pady=(0, 8))
        self.import_button = self._button(parent, "匯入圖片", self.import_images)
        self.import_button.pack(fill="x", padx=11, pady=(0, 5))
        self.video_button = self._button(parent, "影片自動標註", self.import_video)
        self.video_button.pack(fill="x", padx=11, pady=(0, 8))
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="both", expand=True, padx=(6, 3))
        self.asset_tree = ttk.Treeview(frame, columns=("name", "count"), show="headings", selectmode="browse")
        self.asset_tree.heading("name", text="檔名")
        self.asset_tree.heading("count", text="狀態")
        self.asset_tree.column("name", width=117, stretch=True)
        self.asset_tree.column("count", width=63, stretch=False)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.asset_tree.yview)
        self.asset_tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        self.asset_tree.pack(fill="both", expand=True)
        self.asset_tree.bind("<<TreeviewSelect>>", self._asset_selected)
        nav = tk.Frame(parent, bg=PANEL)
        nav.pack(fill="x", padx=10, pady=10)
        self.previous_button = self._button(nav, "上一張", lambda: self.navigate(-1))
        self.next_button = self._button(nav, "下一張", lambda: self.navigate(1))
        self.previous_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.next_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _canvas_panel(self, parent):
        tools = tk.Frame(parent, bg=PANEL)
        tools.pack(fill="x")
        self.prompt_tools = []
        for text, value in PROMPT_TOOLS:
            button = ttk.Radiobutton(tools, text=text, variable=self.tool, value=value, style="Tool.TRadiobutton")
            button.pack(side="left", padx=(8 if value == "box" else 3, 0), pady=7)
            self.prompt_tools.append(button)
        self._button(tools, "還原縮放", self.fit).pack(side="right", padx=8, pady=5)
        repairs = self.repair_toolbar = tk.Frame(parent, bg=PANEL)
        self.raw_toggle = ttk.Checkbutton(repairs, text="看原始遮罩", variable=self.show_raw,
                                          command=self._raw_view_changed)
        self.raw_toggle.pack(side="left", padx=(8, 6), pady=(0, 4))
        self.repair_tools = []
        for text, value in REPAIR_TOOLS:
            button = ttk.Radiobutton(repairs, text=text, variable=self.tool, value=value,
                                     style="Repair.TRadiobutton")
            button.pack(side="left", padx=(0, 2), pady=(0, 4))
            self.repair_tools.append(button)
        self.undo_repair_button = ttk.Button(repairs, text="撤回修整", command=self.undo_repair, padding=(4, 1))
        self.undo_repair_button.pack(side="right", padx=(0, 8), pady=(0, 4))
        self.canvas = tk.Canvas(parent, bg=WORKSPACE, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._schedule_canvas())
        self.canvas.bind("<ButtonPress-1>", self._prompt_start)
        self.canvas.bind("<B1-Motion>", self._prompt_move)
        self.canvas.bind("<ButtonRelease-1>", self._prompt_end)
        self.canvas.bind("<MouseWheel>", self._wheel)
        for button in (2, 3):
            self.canvas.bind(f"<ButtonPress-{button}>", self._pan_start)
            self.canvas.bind(f"<B{button}-Motion>", self._pan_move)
            self.canvas.bind(f"<ButtonRelease-{button}>", self._pan_end)
        bar = tk.Frame(parent, bg=PANEL)
        bar.pack(fill="x")
        self.image_meta = self._label(bar, fg=MUTED)
        self.image_meta.pack(side="left", padx=10, pady=7)
        self._label(bar, "滾輪縮放　右鍵拖曳移動", fg=MUTED, size=9).pack(side="right", padx=10)

    def _annotation_panel(self, parent):
        bottom = tk.Frame(parent, bg=PANEL)
        bottom.pack(side="bottom", fill="x", padx=11, pady=8)
        self.delete_button = self._button(bottom, "刪除所選", self.delete_annotation)
        self.review_button = self._button(bottom, "標記已審核", self.toggle_review)
        self.delete_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.review_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._label(parent, "標註設定", size=12, bold=True).pack(fill="x", padx=12, pady=(8, 4))
        modes = tk.Frame(parent, bg=PANEL)
        modes.pack(fill="x", padx=11, pady=(0, 5))
        self._label(modes, "類型", fg=MUTED, size=9).pack(side="left", padx=(0, 6))
        self.mode_picker = ttk.Combobox(modes, values=(BOX_MODE, MASK_MODE), textvariable=self.mode,
                                        state="readonly", width=13)
        self.mode_picker.pack(side="left", fill="x", expand=True)
        self.mode_picker.bind("<<ComboboxSelected>>", self._mode_changed)
        classes = tk.Frame(parent, bg=PANEL)
        classes.pack(fill="x", padx=11, pady=(0, 5))
        self._label(classes, "類別", fg=MUTED, size=9).pack(side="left", padx=(0, 6))
        self.class_picker = ttk.Combobox(classes, textvariable=self.class_name, state="readonly", width=13)
        self.class_picker.pack(side="left", fill="x", expand=True)
        self.class_picker.bind("<<ComboboxSelected>>", self._settings_changed)
        self.new_label_button = ttk.Button(classes, text="新增", command=self.add_label, padding=(8, 3))
        self.new_label_button.pack(side="left", padx=(5, 0))
        mask_options = self.mask_options = tk.Frame(parent, bg=PANEL)
        engines = tk.Frame(mask_options, bg=PANEL)
        engines.pack(fill="x", padx=11, pady=(0, 4))
        self._label(engines, "方法", fg=MUTED, size=9).pack(side="left", padx=(0, 6))
        engine_options = tuple(ENGINES) if self.sam_available else ("GrabCut",)
        self.engine_picker = ttk.Combobox(engines, values=engine_options, textvariable=self.method,
                                          state="readonly", width=13)
        self.engine_picker.pack(side="left", fill="x", expand=True)
        self.engine_picker.bind("<<ComboboxSelected>>", self._settings_changed)
        cleanup = tk.Frame(mask_options, bg=PANEL)
        cleanup.pack(fill="x", padx=11, pady=(0, 5))
        self._label(cleanup, "自動修整", fg=MUTED, size=9).pack(side="left", padx=(0, 6))
        self.cleanup_picker = ttk.Combobox(cleanup, values=tuple(CLEANUP), textvariable=self.cleanup,
                                           state="readonly", width=13)
        self.cleanup_picker.pack(side="left", fill="x", expand=True)
        self.cleanup_picker.bind("<<ComboboxSelected>>", self._cleanup_changed)
        if not self.sam_available:
            self._label(mask_options, "目前使用 GrabCut。未找到本機 SAM 2.1 模型。", fg=MUTED, size=9,
                        wraplength=260).pack(fill="x", padx=12, pady=(0, 6))

        self._label(mask_options, "前景點保留，背景點排除。", fg=MUTED, size=9).pack(fill="x", padx=12, pady=(0, 2))
        self.prompt_count = self._label(mask_options, fg=MUTED, size=9)
        self.prompt_count.pack(fill="x", padx=12, pady=(0, 4))
        hints = tk.Frame(mask_options, bg=PANEL)
        hints.pack(fill="x", padx=11, pady=(0, 4))
        self.undo_button = ttk.Button(hints, text="撤回提示", command=self.undo_prompt, padding=(6, 4))
        self.clear_button = ttk.Button(hints, text="清除提示", command=self.clear_prompts, padding=(6, 4))
        self.undo_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.clear_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.generate_button = self._button(parent, "產生候選", self.generate, primary=True)
        self.generate_button.pack(fill="x", padx=11)
        self.preview_text = self._label(parent, fg=MUTED, size=9, wraplength=260)
        self.preview_text.pack(fill="x", padx=12, pady=(5, 4))
        decision = self.decision = tk.Frame(parent, bg=PANEL)
        decision.pack(fill="x", padx=11, pady=(0, 8))
        self.accept_button = ttk.Button(decision, text="接受並儲存", command=self.accept, padding=(8, 4))
        self.discard_button = ttk.Button(decision, text="捨棄候選", command=self._discard_current, padding=(8, 4))
        self.accept_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.discard_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

        ttk.Separator(parent).pack(fill="x", padx=11, pady=4)
        saved = tk.Frame(parent, bg=PANEL)
        saved.pack(fill="x", padx=12, pady=(5, 4))
        self.annotation_heading = self._label(saved, "已儲存標註", size=12, bold=True)
        self.annotation_heading.pack(side="left")
        ttk.Checkbutton(saved, text="顯示", variable=self.show_masks, command=self._schedule_canvas).pack(side="right")
        self.repair_selected_button = ttk.Button(saved, text="修整所選", command=self.repair_selected, padding=(4, 1))
        self.repair_selected_button.pack(side="right", padx=(2, 4))
        list_frame = tk.Frame(parent, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=11)
        self.annotation_tree = ttk.Treeview(list_frame, columns=("label", "kind"), show="headings", selectmode="browse", height=1)
        self.annotation_tree.heading("label", text="物件類別")
        self.annotation_tree.heading("kind", text="類型")
        self.annotation_tree.column("label", width=155, stretch=True)
        self.annotation_tree.column("kind", width=60, stretch=False)
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.annotation_tree.yview)
        self.annotation_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.annotation_tree.pack(fill="both", expand=True)
        self.annotation_tree.bind("<<TreeviewSelect>>", self._annotation_selected)
        self._show_mode_controls()

    def _message(self, text, error=False):
        self.status.configure(text=text, fg=NEGATIVE if error else MUTED)

    def _error(self, title, exc):
        self._message(f"{title}：{exc}", error=True)
        messagebox.showerror(title, str(exc), parent=self.root)

    def _key(self, event, callback):
        if event.widget.winfo_class() not in {"Entry", "TEntry", "Text", "TCombobox", "Treeview", "Spinbox"}:
            callback()
            return "break"

    def _tool_key(self, event):
        if event.state & 0xC:
            return
        tools = {"b": "box", "p": "positive", "n": "negative"}
        tool = tools.get(event.keysym.lower())
        if tool and not self._box_mode() and not self._repairing_saved():
            return self._key(event, lambda: self.tool.set(tool))

    def _box_mode(self):
        return self.current_mode == BOX_MODE

    @staticmethod
    def _annotation_kind(annotation):
        return "bbox" if annotation.get("type") == "bbox" else "mask"

    def _selected_record(self):
        if not self.asset or not self.selected_annotation:
            return None
        return next((row for row in self.asset["annotations"] if row["id"] == self.selected_annotation), None)

    def _show_mode_controls(self):
        if self._box_mode():
            self.mask_options.pack_forget()
            self.repair_toolbar.pack_forget()
            for tool in self.prompt_tools[1:]:
                tool.pack_forget()
            self.accept_button.pack_forget()
            self.generate_button.configure(command=self.save_box)
        else:
            self.mask_options.pack(fill="x", before=self.generate_button)
            self.repair_toolbar.pack(fill="x", before=self.canvas)
            for tool in self.prompt_tools[1:]:
                tool.pack(side="left", padx=(3, 0), pady=7)
            self.accept_button.pack(side="left", fill="x", expand=True, padx=(0, 4), before=self.discard_button)
            self.generate_button.configure(command=self.generate)

    def _mode_changed(self, _event=None):
        target = self.mode.get()
        if target == self.current_mode:
            return
        if self.preview is not None or self.box or self.positive or self.negative:
            if not messagebox.askyesno("切換標註類型", "捨棄尚未儲存的草稿並切換？", parent=self.root, default="no"):
                self.mode.set(self.current_mode)
                return
        self._switch_mode(target)

    def _switch_mode(self, target):
        self.current_mode = target
        self.mode.set(target)
        self._clear_preview()
        self._clear_box_draft()
        self.positive, self.negative = [], []
        self.tool.set("box")
        self._show_mode_controls()
        self._refresh_annotations()
        self._prompt_changed()

    def _clear_box_draft(self):
        self.box = self.drag_box = None
        self.edit_box_id = self.expected_bbox = None
        self.prompt_history.clear()

    def _discard_current(self):
        if not self._box_mode():
            self.discard()
        elif self.box is not None or self.edit_box_id is not None:
            self._clear_box_draft()
            self._prompt_changed()
            self._message("已捨棄框選草稿。")

    def new_project(self):
        if self.task_kind in {"import", "export", "video"}:
            return
        parent = filedialog.askdirectory(title="選擇存放新專案的位置", parent=self.root)
        if not parent:
            return
        name = simpledialog.askstring("新建專案", "專案資料夾名稱", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name or name in {".", ".."} or any(char in name for char in '\\/:*?"<>|'):
            self._error("無法建立專案", "請輸入有效的資料夾名稱。")
            return
        directory = Path(parent) / name
        try:
            self._set_project(Project.create(directory), directory)
        except Exception as exc:
            self._error("無法建立專案", exc)

    def open_project(self):
        if self.task_kind in {"import", "export", "video"}:
            return
        directory = filedialog.askdirectory(title="選擇含 project.json 的專案資料夾", parent=self.root)
        if directory:
            self.open_path(Path(directory))

    def open_path(self, directory: Path):
        try:
            self._set_project(Project.open(directory), directory)
        except Exception as exc:
            self._error("無法開啟專案", exc)

    def _set_project(self, project, directory):
        self.project, self.directory = project, directory.resolve()
        self.project_title.configure(text=self.directory.name)
        self.class_name.set(project.labels[0] if project.labels else "")
        self.class_picker.configure(values=project.labels)
        self._select_asset(project.assets[0] if project.assets else None)
        self._refresh_asset_list()
        self._message(f"專案位置　{self.directory}")

    def import_images(self):
        if self.project is None or self.task is not None:
            return
        files = filedialog.askopenfilenames(title="選擇要匯入的圖片", parent=self.root,
                                            filetypes=[("圖片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                                                       ("所有檔案", "*.*")])
        if not files:
            return
        project = self.project
        self._message(f"正在匯入 {len(files)} 張圖片…")
        self._start_task("import", lambda: project.add_images([Path(file) for file in files]),
                         self._after_import, "無法匯入圖片")

    def _after_import(self, added):
        if self.project.assets and self.asset is None:
            self._select_asset(self.project.assets[0])
        self._refresh_asset_list()
        self._message(f"新增 {added} 張圖片")

    def import_video(self):
        if self.project is None or self.task is not None:
            return
        chosen = filedialog.askopenfilename(title="選擇影片以預覽與審查", parent=self.root, filetypes=VIDEO_TYPES)
        if not chosen:
            return
        approval = VideoReviewDialog(self.root, Path(chosen)).show()
        if approval is None:
            return
        project = self.project
        original_assets = {asset["id"] for asset in project.assets}
        job = self.video_import = VideoImport(approval)
        self._message("已確認影片，正在核對原檔和模型…")
        self.video_stop.state(["!disabled"])
        self.video_stop.pack(side="right")

        def after(result):
            if result is None:
                self._message("已取消影片處理。")
                return
            self.class_picker.configure(values=project.labels)
            if not self.class_name.get() and project.labels:
                self.class_name.set(project.labels[0])
            first_new = next((asset for asset in project.assets if asset["id"] not in original_assets), None)
            if first_new is not None and not (self.box or self.preview or self.positive or self.negative):
                self._select_asset(first_new)
            self._refresh_asset_list()
            imported, boxes, skipped = (result[key] for key in ("imported_frames", "added_boxes", "skipped_frames"))
            self._message(f"匯入 {imported:,} 張畫格、{boxes:,} 個候選框，略過 {skipped:,} 張重複畫格。"
                          "請逐張複核，再標記畫格已審核。")

        self._start_task("video", lambda: job.run(project), after, "無法匯入影片標註")
        self.root.after(120, lambda: self._poll_video(job))

    def _poll_video(self, job):
        if not self.alive or self.video_import is not job:
            return
        latest = None
        while True:
            try:
                latest = job.events.get_nowait()
            except Empty:
                break
        if latest is not None and not job.stop.is_set():
            if latest.get("event") == "commit":
                self.video_stop.state(["disabled"])
                self._message("偵測已完成，正在儲存待審畫格…")
            elif latest.get("event") == "progress" and latest.get("stage") == "verifying":
                self._message("正在核對已確認的影片和模型…")
            elif latest.get("event") == "progress":
                read, total = latest.get("frames_read", 0), latest.get("total_frames", 0)
                sampled = latest.get("frames_sampled", 0)
                counted = f"{read:,} / {total:,}" if total else f"{read:,}"
                self._message(f"正在偵測與追蹤 {counted} 畫格 · 已擷取 {sampled:,} 張")
        self.root.after(120, lambda: self._poll_video(job))

    def cancel_video(self):
        if self.video_import is not None and self.video_import.cancel():
            self.video_stop.state(["disabled"])
            self._message("正在停止影片處理…")

    def add_label(self):
        if not self.project or self.task_kind in {"import", "export", "video"} or self._repairing_saved() or self.edit_box_id:
            return
        name = simpledialog.askstring("新增物件類別", "輸入你的類別名稱", parent=self.root)
        if name is None:
            return
        try:
            label = self.project.add_label(name)
            self.class_picker.configure(values=self.project.labels)
            self.class_name.set(label)
            self._settings_changed()
            self._message(f"已新增類別「{label}」")
        except Exception as exc:
            self._error("無法新增類別", exc)

    def _refresh_asset_list(self):
        assets = self.project.assets if self.project is not None else []
        current = {a["id"] for a in assets}
        for row_id in self.asset_tree.get_children():
            if row_id not in current:
                self.asset_tree.delete(row_id)
        for index, asset in enumerate(assets):
            aid = asset["id"]
            status = "已審核" if asset.get("reviewed") else "影片待審" if asset.get("video") else f"{len(asset['annotations'])} 件"
            values = (asset["name"], status)
            if self.asset_tree.exists(aid):
                self.asset_tree.item(aid, values=values)
            else:
                self.asset_tree.insert("", index, iid=aid, values=values)
        completed = sum(bool(a.get("reviewed")) for a in assets)
        self.asset_summary.configure(text=f"{len(assets):,} 張圖片 · {completed:,} 張已審核")
        if self.asset:
            self.asset_tree.selection_set(self.asset["id"])
            self.asset_tree.see(self.asset["id"])
        self._update_controls()

    def _asset_selected(self, _event):
        selected = self.asset_tree.selection()
        if self.project is None or not selected or (self.asset and selected[0] == self.asset["id"]):
            return
        asset = next((row for row in self.project.assets if row["id"] == selected[0]), None)
        if asset:
            self._select_asset(asset)

    def _select_asset(self, asset):
        self.asset, self.image = None, None
        self._clear_preview()
        self.selected_annotation = None
        self._clear_box_draft()
        self.positive, self.negative = [], []
        self.prompt_history.clear()
        self.revision += 1
        self.zoom, self.pan = 1.0, (0.0, 0.0)
        if asset is not None:
            try:
                with Image.open(self.project.image_path(asset["id"])) as image:
                    self.image = image.convert("RGB")
                self.asset = asset
                action = "影片待審" if asset.get("video") and not asset.get("reviewed") else "檢視"
                self._message(f"{action}　{asset['name']}")
            except Exception as exc:
                self._error("無法讀取圖片", exc)
        self._refresh_annotations()
        self._update_controls()
        self._schedule_canvas()

    def navigate(self, delta):
        if not self.project or not self.project.assets:
            return
        assets = self.project.assets
        index = self._asset_index()
        target = max(0, min(len(assets)-1, index + delta))
        if target != index:
            self._select_asset(assets[target])
            self.asset_tree.selection_set(assets[target]["id"])
            self.asset_tree.see(assets[target]["id"])

    def _asset_index(self):
        if self.asset is None:
            return -1
        aid = self.asset["id"]
        return next((index for index, row in enumerate(self.project.assets) if row["id"] == aid), -1)

    def _settings_changed(self, _event=None):
        self._prompt_changed()

    def _cleanup_changed(self, _event=None):
        if not self._preview_is_current() or self.task is not None:
            self._prompt_changed()
            return
        preview = self.preview
        cleaned, stats = self._cleanup_cached_mask(preview.raw_mask)
        # Carry explicit brush and click edits across an automatic strength change.
        edits = preview.mask != preview.base_mask
        updated = np.where(edits, preview.mask, cleaned) if edits.any() else cleaned
        if not updated.any():
            self._message("改變修整強度會清空候選，請先撤回手動編修。", error=True)
            return
        self.repair_history.clear()
        self.preview = replace(preview, mask=updated, base_mask=cleaned, cleanup_stats=stats)
        self._refresh_preview_layer()
        self._update_controls()
        self._message("已更新自動修整強度。")

    def _cleanup_cached_mask(self, mask):
        result = clean_mask(mask, positive=self.positive, negative=self.negative,
                            strength=CLEANUP[self.cleanup.get()])
        return result.mask, {"removed_components": result.removed_components,
                             "filled_holes": result.filled_holes, "changed_pixels": result.changed_pixels}

    def _prompt_changed(self):
        self.revision += 1
        if self.task_kind == "inference":
            self._message("提示已更新，請重新產生候選。")
        self._update_controls()
        self._schedule_canvas()

    def _remember_prompts(self):
        self.prompt_history.append((self.box, self.positive.copy(), self.negative.copy()))

    def undo_prompt(self):
        if self.prompt_history and not self._repairing_saved():
            self.box, self.positive, self.negative = self.prompt_history.pop()
            self._prompt_changed()

    def clear_prompts(self):
        if not self._repairing_saved() and (self.box or self.positive or self.negative):
            self._remember_prompts()
            self.box = None
            self.positive, self.negative = [], []
            self._prompt_changed()

    def _image_point(self, event, *, bound=False):
        if self.image is None:
            return None
        x = (event.x - self.origin[0]) / self.scale
        y = (event.y - self.origin[1]) / self.scale
        w, h = self.image.size
        if bound:
            return min(w, max(0.0, x)), min(h, max(0.0, y))
        if 0 <= x < w and 0 <= y < h:
            return x, y

    def _prompt_start(self, event):
        point = self._image_point(event)
        if point is None:
            return
        self.canvas.focus_set()
        kind = "box" if self._box_mode() else self.tool.get()
        if kind in {"fill", "island", "paint", "erase"}:
            self._start_repair(kind, point)
            return
        if self._repairing_saved():
            return
        if kind == "box":
            self.drag_box = (point, point)
            self._schedule_canvas()
            return
        if len(self.positive) + len(self.negative) >= 64:
            self._message("每個物件最多 64 個提示點。可撤回或清除提示。", error=True)
            return
        self._remember_prompts()
        points = self.positive if kind == "positive" else self.negative
        points.append(point)
        self._prompt_changed()

    def _prompt_move(self, event):
        if self.brush_stroke:
            self._draw_brush_segment(self._image_point(event, bound=True))
            return
        if self.drag_box:
            self.drag_box = (self.drag_box[0], self._image_point(event, bound=True))
            self._schedule_canvas()

    def _prompt_end(self, event):
        if self.brush_stroke:
            self._draw_brush_segment(self._image_point(event, bound=True))
            stroke, self.brush_stroke = self.brush_stroke, None
            mask = stroke.working.view(np.bool_)
            if not mask.any():
                self.preview = replace(self.preview, mask=stroke.before)
                self._message("遮罩至少需保留一點前景。", error=True)
            elif not np.array_equal(mask, stroke.before):
                self._remember_repair(stroke.before)
                self._message("已修整筆畫，可撤回或儲存。")
            self._refresh_preview_layer()
            self._update_controls()
            return
        if not self.drag_box:
            return
        x, y = self.drag_box[0]
        other_x, other_y = self._image_point(event, bound=True)
        self.drag_box = None
        if abs(x - other_x) >= 2 and abs(y - other_y) >= 2:
            self._remember_prompts()
            self.box = (min(x, other_x), min(y, other_y), max(x, other_x), max(y, other_y))
            self._prompt_changed()
        else:
            if self._box_mode() and self.box is None:
                self._select_box_at(x, y)
            self._schedule_canvas()

    def _select_box_at(self, x, y):
        matches = [row for row in self.asset["annotations"] if row["bbox"][0] <= x <= row["bbox"][0]+row["bbox"][2]
                   and row["bbox"][1] <= y <= row["bbox"][1]+row["bbox"][3]]
        if matches:
            target = min(matches, key=lambda row: row["bbox"][2] * row["bbox"][3])
            self.annotation_tree.selection_set(target["id"])
            self.annotation_tree.see(target["id"])

    def _repairing_saved(self):
        return self.preview is not None and self.preview.replaces_id is not None

    def _repair_available(self):
        return self._preview_is_current() and self.task is None and not self.show_raw.get()

    def _raw_view_changed(self):
        self._update_controls()
        self._schedule_canvas()

    def _remember_repair(self, mask):
        packed = np.packbits(mask, axis=None)
        self.repair_history.append(packed)
        while len(self.repair_history) > 20 or sum(row.nbytes for row in self.repair_history) > 32 * 1024**2:
            self.repair_history.pop(0)

    def undo_repair(self):
        if self.repair_history and self._repair_available() and self.brush_stroke is None:
            packed = self.repair_history.pop()
            mask = np.unpackbits(packed, count=self.preview.mask.size).reshape(self.preview.mask.shape)
            self.preview = replace(self.preview, mask=mask.view(np.bool_))
            self._refresh_preview_layer()
            self._update_controls()
            self._message("已撤回一次手動修整。")

    def _undo(self):
        if self.repair_history:
            self.undo_repair()
        else:
            self.undo_prompt()

    def _start_repair(self, kind, point):
        if not self._repair_available():
            self._message("先產生候選或開啟已儲存標註，再修整遮罩。")
            return
        if kind in {"paint", "erase"}:
            mask = self.preview.mask
            center = self._brush_pixel(point)
            self.brush_stroke = BrushStroke(mask, mask.astype(np.uint8), center,
                                             int(kind == "paint"), max(1, min(100, round(5 / self.scale))))
            self._draw_brush_segment(point)
            return
        self._repair_component(kind, (int(point[0]), int(point[1])))

    def _brush_pixel(self, point):
        height, width = self.preview.mask.shape
        return min(width - 1, int(point[0])), min(height - 1, int(point[1]))

    def _draw_brush_segment(self, point):
        stroke = self.brush_stroke
        center = self._brush_pixel(point)
        cv2.line(stroke.working, stroke.previous, center, stroke.value, max(1, stroke.radius * 2))
        stroke.previous = center
        self.preview = replace(self.preview, mask=stroke.working.view(np.bool_))
        self.preview_layer = self._mask_layer(self.preview.mask)
        self._schedule_canvas()

    def _repair_component(self, kind, point):
        mask = self.preview.mask
        x, y = point
        fill = kind == "fill"
        if bool(mask[y, x]) == fill:
            self._message("請點在要填補的封閉孔洞內。" if fill else "請點在要移除的分離小塊上。", error=True)
            return
        region = ~mask if fill else mask
        _, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=4 if fill else 8)
        component = labels[y, x]
        left, top, width, height, area = stats[component]
        if fill and (left == 0 or top == 0 or left + width == mask.shape[1] or top + height == mask.shape[0]):
            self._message("此處與外部背景相連。請用加畫筆刷修補缺口。", error=True)
            return
        if not fill and area >= stats[1:, cv2.CC_STAT_AREA].max():
            self._message("此區塊是遮罩主體，請選擇分離區塊或使用擦除筆刷。", error=True)
            return
        updated = mask.copy()
        updated[labels == component] = fill
        self._remember_repair(mask)
        self.preview = replace(self.preview, mask=updated)
        self._refresh_preview_layer()
        self._update_controls()
        self._message(f"已{'填補孔洞' if fill else '移除孤島'}（{area:,} px）。")

    def fit(self):
        self.zoom, self.pan = 1.0, (0.0, 0.0)
        self._schedule_canvas()

    def _wheel(self, event):
        point = self._image_point(event)
        if point is None or not event.delta:
            return
        zoom = max(1., min(9., self.zoom * (1.16 if event.delta > 0 else 1/1.16)))
        factor = zoom / self.zoom
        self.zoom = zoom
        w, h = self.image.size
        scale = self.scale * factor
        self.pan = (event.x - (point[0]-w/2)*scale - self.canvas.winfo_width()/2,
                    event.y - (point[1]-h/2)*scale - self.canvas.winfo_height()/2)
        if zoom == 1:
            self.pan = (0.0, 0.0)
        self._schedule_canvas()

    def _pan_start(self, event):
        if self.image:
            self.pan_drag = (event.x, event.y, self.pan)

    def _pan_move(self, event):
        if self.pan_drag:
            x, y, pan = self.pan_drag
            self.pan = (pan[0]+event.x-x, pan[1]+event.y-y)
            self._schedule_canvas()

    def _pan_end(self, _event):
        self.pan_drag = None

    def _schedule_canvas(self):
        if self.alive and self.draw_job is None:
            self.draw_job = self.root.after_idle(self._render_canvas)

    def _render_canvas(self):
        self.draw_job = None
        canvas = self.canvas
        canvas.delete("all")
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if self.image is None:
            self.image_meta.configure(text="")
            placeholder = "匯入圖片以開始標註" if self.project else "建立或開啟專案"
            canvas.create_text(cw/2, ch/2, text=placeholder, font=(FONT, 15, "bold"), fill=MUTED)
            self.photo = None
            return
        iw, ih = self.image.size
        self.scale = scale = min(max(1, cw-20) / iw, max(1, ch-20) / ih) * self.zoom
        margin = 24
        px = max(-cw/2-iw*scale/2+margin, min(cw/2+iw*scale/2-margin, self.pan[0]))
        py = max(-ch/2-ih*scale/2+margin, min(ch/2+ih*scale/2-margin, self.pan[1]))
        self.pan = px, py
        self.origin = ox, oy = (cw-iw*scale)/2 + px, (ch-ih*scale)/2 + py
        x0, y0 = max(0, floor(-ox/scale)), max(0, floor(-oy/scale))
        x1, y1 = min(iw, ceil((cw-ox)/scale)), min(ih, ceil((ch-oy)/scale))
        if x1 <= x0 or y1 <= y0:
            return
        crop = x0, y0, x1, y1
        size = max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))
        display = self.image.crop(crop).resize(size, Image.Resampling.BILINEAR)
        if self.show_masks.get() and not self._box_mode():
            for annotation, layer in self.mask_layers:
                if self.preview is not None and self.preview.replaces_id == annotation["id"]:
                    continue
                active = annotation["id"] == self.selected_annotation
                label = annotation["label"]
                color = COLORS[self.project.labels.index(label) % len(COLORS)]
                self._blend_mask(display, layer, crop, color, 148 if active else 86)
        candidate_layer = self.raw_layer if self.show_raw.get() else self.preview_layer
        if candidate_layer is not None:
            self._blend_mask(display, candidate_layer, crop, CANDIDATE, 111 if self._preview_is_current() else 64)
        self.photo = ImageTk.PhotoImage(display, master=self.root)
        canvas.create_image(ox+x0*scale, oy+y0*scale, image=self.photo, anchor="nw")
        if self.show_masks.get():
            self._draw_saved_boxes(canvas, ox, oy, scale)
        self._draw_prompts(canvas, ox, oy, scale)
        video = self.asset.get("video")
        suffix = f"　·　影片 {video['time_ms'] / 1000:.3f} s" if video else ""
        self.image_meta.configure(text=f"{iw:,} × {ih:,} px　·　{scale:.0%}{suffix}")

    @staticmethod
    def _blend_mask(display, layer, crop, color, alpha):
        mask = layer.crop(crop)
        resampling = Image.Resampling.BOX if display.width < mask.width or display.height < mask.height else Image.Resampling.NEAREST
        mask = mask.resize(display.size, resampling)
        opacity = mask.point([round(value * alpha / 255) for value in range(256)])
        display.paste(color, (0, 0, *display.size), opacity)

    def _draw_prompts(self, canvas, ox, oy, scale):
        box = self.box
        if self.drag_box:
            (x, y), (other_x, other_y) = self.drag_box
            box = (x, y, other_x, other_y)
        if box:
            x0, y0, x1, y1 = box
            canvas.create_rectangle(ox+x0*scale, oy+y0*scale, ox+x1*scale, oy+y1*scale,
                                    outline=BOX_COLOR, width=2, dash=(6, 4))
        for points, color, marker in ((self.positive, POSITIVE, "+"), (self.negative, NEGATIVE, "−")):
            for point_x, point_y in points:
                x, y = ox+point_x*scale, oy+point_y*scale
                canvas.create_oval(x-7, y-7, x+7, y+7, outline=WORKSPACE, width=2, fill=color)
                canvas.create_text(x, y-1, text=marker, fill=WORKSPACE, font=(FONT, 10, "bold"))

    def _draw_saved_boxes(self, canvas, ox, oy, scale):
        labels = self.project.labels
        for annotation in self.asset["annotations"]:
            if annotation["id"] == self.edit_box_id or (not self._box_mode() and self._annotation_kind(annotation) != "bbox"):
                continue
            x, y, width, height = annotation["bbox"]
            pending = annotation.get("source") == "video_auto" and not self.asset.get("reviewed")
            color = CANDIDATE if pending else COLORS[labels.index(annotation["label"]) % len(COLORS)]
            left, top = ox + x*scale, oy + y*scale
            active = annotation["id"] == self.selected_annotation
            canvas.create_rectangle(left, top, left + width*scale, top + height*scale,
                                    outline=color, width=3 if active else 2, dash=(6, 4) if pending else ())
            if active:
                track = annotation.get("track_id")
                label = annotation["label"] if track is None else f"{annotation['label']} · #{track}"
                canvas.create_text(left+5, max(2, top-18), text=label, fill=color,
                                   anchor="nw", font=(FONT, 10, "bold"))

    def _preview_is_current(self):
        return bool(self.preview and self.preview.revision == self.revision)

    @staticmethod
    def _mask_layer(mask):
        return Image.fromarray(mask.astype(np.uint8) * 255)

    def _refresh_preview_layer(self):
        self.preview_layer = self._mask_layer(self.preview.mask)
        self.manual_pixels = int(np.count_nonzero(self.preview.mask != self.preview.base_mask))
        self._schedule_canvas()

    def _set_preview(self, preview):
        self.preview = preview
        self.raw_layer = self._mask_layer(preview.raw_mask)
        self.repair_history.clear()
        self.brush_stroke = None
        self.show_raw.set(False)
        self._refresh_preview_layer()

    def _clear_preview(self):
        self.preview = self.preview_layer = self.raw_layer = None
        self.repair_history.clear()
        self.brush_stroke = None
        self.manual_pixels = 0
        self.show_raw.set(False)
        if self.tool.get() in {"fill", "island", "paint", "erase"}:
            self.tool.set("box")

    def repair_selected(self):
        if self.task is not None or self.asset is None or self.preview is not None or self.box or not self.selected_annotation:
            return
        annotation = self._selected_record()
        if annotation is None:
            return
        if self._annotation_kind(annotation) == "bbox":
            self._switch_mode(BOX_MODE)
            x, y, width, height = annotation["bbox"]
            self.edit_box_id = annotation["id"]
            self.expected_bbox = annotation["bbox"].copy()
            self.class_name.set(annotation["label"])
            self.box = x, y, x + width, y + height
            self._prompt_changed()
            self._message("拖出新的框，再按「儲存框」即可更新原標註。")
            return
        if self._box_mode():
            self._switch_mode(MASK_MODE)
        try:
            original = self.project.decode_mask(annotation)
            self.box = self.drag_box = None
            self.positive, self.negative = [], []
            self.prompt_history.clear()
            cleaned, stats = self._cleanup_cached_mask(original)
        except Exception as exc:
            self._error("無法開啟已儲存遮罩", exc)
            return
        self.revision += 1
        self._set_preview(Preview(cleaned, annotation["label"], annotation["source"], annotation.get("score"),
                                  0., self.revision, original, cleaned, stats,
                                  replaces_id=annotation["id"], expected_counts=annotation["counts"].copy()))
        self.tool.set("fill")
        self._update_controls()
        self._message("可比對原始遮罩及修整結果。確認後按「儲存修整」。")

    def save_box(self):
        if not self._box_mode() or self.asset is None or self.task is not None or self.box is None:
            return
        label = self.class_name.get()
        if not label:
            self._message("請先新增或選擇一個物件類別。", error=True)
            return
        editing = self.edit_box_id is not None
        try:
            if editing:
                self.selected_annotation = self.project.replace_box(self.asset["id"], self.edit_box_id, self.box,
                                                                    expected_bbox=self.expected_bbox)
            else:
                self.selected_annotation = self.project.add_box(self.asset["id"], label, self.box)
        except Exception as exc:
            self._error("無法儲存偵測框", exc)
            return
        self._clear_box_draft()
        self._reload_asset_record()
        self._prompt_changed()
        self._message(f"已{'更新' if editing else '儲存'}「{label}」偵測框。")

    def generate(self):
        if self._box_mode() or self.task is not None or self.asset is None or self.image is None or self._repairing_saved():
            return
        label, engine = self.class_name.get(), ENGINES[self.method.get()]
        if not label or not (self.box or self.positive):
            self._message("請選擇類別，再框選物件或加入一個前景點。", error=True)
            return
        revision = self.revision
        project, aid, image = self.project, self.asset["id"], self.image
        box, positive, negative = self.box, self.positive.copy(), self.negative.copy()
        cleanup = CLEANUP[self.cleanup.get()]

        def is_current():
            return (self.project is project and self.asset is not None
                    and self.asset["id"] == aid and self.revision == revision)

        def apply(result):
            if not is_current():
                if self.project is project and self.asset is not None and self.asset["id"] == aid:
                    self._message("提示已更新，請重新產生候選。")
                return
            raw = result.mask if result.raw_mask is None else result.raw_mask
            stats = dict(removed_components=0, filled_holes=0, changed_pixels=0)
            stats.update(result.cleanup_stats)
            self._set_preview(Preview(result.mask, label, engine, result.score, result.elapsed_ms, revision,
                                      raw, result.mask, stats))
            self._message(f"已產生「{label}」候選。確認輪廓後可接受並儲存。")

        self._message(f"正在產生「{label}」候選…")
        self._start_task("inference", lambda: self.engine.generate(engine, image, box=box, positive=positive,
                                negative=negative, image_key=str(project.image_path(aid)), cleanup=cleanup), apply, "無法產生候選",
                         show_error=is_current)

    def accept(self):
        if self.asset is None or not self._repair_available() or self.brush_stroke is not None:
            return
        preview = self.preview
        try:
            if preview.replaces_id is None:
                self.selected_annotation = self.project.add_annotation(self.asset["id"], preview.label,
                                            preview.mask, preview.source, None if self.manual_pixels else preview.score)
            else:
                self.selected_annotation = self.project.replace_annotation_mask(self.asset["id"], preview.replaces_id,
                            preview.mask, expected_counts=preview.expected_counts)
        except Exception as exc:
            self._error("無法儲存標註", exc)
            return
        self._clear_preview()
        self.prompt_history.clear()
        self.box = None
        self.positive, self.negative = [], []
        self._reload_asset_record()
        self._prompt_changed()
        self._message(f"已儲存「{preview.label}」標註")

    def discard(self):
        if self.preview is not None:
            saved = self._repairing_saved()
            self._clear_preview()
            if self.task_kind == "inference":
                self.revision += 1
            self._message("已捨棄修整草稿。" if saved else "已捨棄候選。提示可供再次分割。")
            self._schedule_canvas()
            self._update_controls()

    def _reload_asset_record(self):
        aid = self.asset["id"]
        self.asset = next(row for row in self.project.assets if row["id"] == aid)
        self._refresh_annotations()
        self._refresh_asset_list()
        self._schedule_canvas()

    def _refresh_annotations(self):
        self.mask_layers = []
        tree = self.annotation_tree
        tree.delete(*tree.get_children())
        annotations = self.asset["annotations"] if self.asset else []
        for annotation in annotations:
            kind = self._annotation_kind(annotation)
            if kind == "mask" and not self._box_mode():
                self.mask_layers.append((annotation, self._mask_layer(self.project.decode_mask(annotation))))
            type_label = "偵測框" if kind == "bbox" else "遮罩"
            if annotation.get("source") == "video_auto" and not self.asset.get("reviewed"):
                type_label = "待複核"
            track = annotation.get("track_id")
            label = annotation["label"] if track is None else f"{annotation['label']} · #{track}"
            tree.insert("", "end", iid=annotation["id"], values=(label, type_label))
        if self.selected_annotation and tree.exists(self.selected_annotation):
            tree.selection_set(self.selected_annotation)
            tree.see(self.selected_annotation)
        else:
            self.selected_annotation = None
        self.annotation_heading.configure(text=f"已儲存標註 {len(annotations)}")

    def _annotation_selected(self, _event):
        options = self.annotation_tree.selection()
        self.selected_annotation = options[0] if options else None
        self._update_controls()
        self._schedule_canvas()

    def delete_annotation(self):
        if not self.asset or not self.selected_annotation or self.task_kind in {"import", "export", "video"} or self._repairing_saved() or self.edit_box_id:
            return
        if not messagebox.askyesno("刪除標註", "刪除選取的標註？", parent=self.root, default="no"):
            return
        try:
            self.project.delete_annotation(self.asset["id"], self.selected_annotation)
            self.selected_annotation = None
            self._reload_asset_record()
            self._message("已刪除選取的標註")
        except Exception as exc:
            self._error("無法刪除標註", exc)

    def toggle_review(self):
        if not self.asset or self.preview is not None or self.task is not None or self.box is not None:
            return
        revised = not self.asset.get("reviewed")
        if revised and self.asset.get("video"):
            existing = bool(self.asset["annotations"])
            check = "你已逐一檢查框與類別，並確認這張畫格沒有漏標嗎？審核通過後即可匯出。"
            if not existing:
                check = "這張畫格目前沒有標註。你已檢查整張畫格並確認沒有目標嗎？"
            if not messagebox.askyesno("審核影片畫格", check, parent=self.root, default="no"):
                return
        elif revised and not self.asset["annotations"]:
            confirmed = messagebox.askyesno("確認無目標", "本圖沒有標註。是否已檢查整張圖並確認沒有目標？",
                                            parent=self.root, default="no")
            if not confirmed:
                return
        try:
            self.project.set_reviewed(self.asset["id"], revised)
            self._reload_asset_record()
            self._message("已標記此圖為已審核" if revised else "已將此圖改回待審核")
        except Exception as exc:
            self._error("無法更新審核狀態", exc)

    def export(self, format_name="coco"):
        if self.project is None or self.task is not None:
            return
        name = format_name.upper()
        directory = filedialog.askdirectory(title=f"選擇空的 {name} 匯出資料夾", parent=self.root)
        if not directory:
            return
        project = self.project
        self._message("正在匯出已儲存的標註…")
        action = project.export_yolo if format_name == "yolo" else project.export_coco
        self._start_task("export", lambda: action(Path(directory)), lambda path: self._after_export(path, name),
                         f"無法匯出 {name}")

    def _after_export(self, path, name="COCO"):
        self._message(f"已匯出至 {path}")
        messagebox.showinfo(f"{name} 已匯出", str(path), parent=self.root)

    def _start_task(self, kind, action, completed, error_title, *, show_error=lambda: True):
        self.task_kind = kind
        self.task = future = self.worker.submit(action)
        self._update_controls()
        self.progress.pack(side="right", padx=(7, 0))
        self.progress.start(13)
        def collect():
            if not self.alive:
                return
            if not future.done():
                self.root.after(45, collect)
                return
            self.task = self.task_kind = None
            self.progress.stop()
            self.progress.pack_forget()
            if kind == "video":
                self.video_import = None
                self.video_stop.pack_forget()
            try:
                result = future.result()
                if not self.closing:
                    completed(result)
            except Exception as exc:
                if show_error():
                    self._error(error_title, exc)
            finally:
                if self.closing:
                    self.close()
                else:
                    self._update_controls()
        self.root.after(45, collect)

    def _update_controls(self):
        def enable(button, available):
            button.state(["!disabled" if available else "disabled"])
        has_project, has_asset = self.project is not None, self.asset is not None
        busy, disk_busy = self.task is not None, self.task_kind in {"import", "export", "video"}
        box_mode, editing_box = self._box_mode(), self.edit_box_id is not None
        has_prompts = bool(self.box or self.positive or self.negative)
        candidate_current = self._preview_is_current()
        editing_saved = self._repairing_saved()
        repair_available = self._repair_available()
        selected = self._selected_record()
        index = self._asset_index()
        enable(self.new_button, not disk_busy)
        enable(self.open_button, not disk_busy)
        enable(self.import_button, has_project and not busy)
        enable(self.video_button, has_project and not busy)
        export_possible = has_project and not busy and any(row.get("reviewed") or (row["annotations"] and not row.get("video"))
                                                            for row in self.project.assets)
        enable(self.export_button, export_possible)
        enable(self.export_coco_button, export_possible)
        enable(self.new_label_button, has_project and not disk_busy and not editing_saved and not editing_box)
        enable(self.previous_button, index > 0)
        enable(self.next_button, has_project and index < len(self.project.assets)-1)
        self.mode_picker.configure(state="disabled" if disk_busy else "readonly")
        class_editable = has_project and self.project.labels and not disk_busy and not editing_saved and not editing_box
        self.class_picker.configure(state="readonly" if class_editable else "disabled")
        self.engine_picker.configure(state="readonly" if has_asset and not editing_saved else "disabled")
        self.cleanup_picker.configure(state="readonly" if has_asset else "disabled")
        for tool in self.prompt_tools:
            enable(tool, has_asset and not editing_saved)
        for tool in self.repair_tools:
            enable(tool, repair_available)
        enable(self.raw_toggle, self.preview is not None)
        enable(self.undo_repair_button, repair_available and bool(self.repair_history))
        enable(self.undo_button, bool(self.prompt_history) and not editing_saved)
        enable(self.clear_button, has_prompts and not editing_saved)
        ready_geometry = bool(self.box) if box_mode else bool(self.box or self.positive)
        enable(self.generate_button, has_asset and not editing_saved and bool(self.class_name.get()) and ready_geometry and not busy)
        if box_mode:
            title = "儲存框"
        elif self.task_kind == "inference":
            title = "正在產生候選…"
        elif self.preview is not None:
            title = "更新候選"
        else:
            title = "產生候選"
        self.generate_button.configure(text=title)
        self.prompt_count.configure(text=f"框選 {int(self.box is not None)}　前景 {len(self.positive)}　背景 {len(self.negative)}")
        if box_mode:
            if self.box is None:
                preview_text = "選擇類別，在圖片拖出矩形，再按「儲存框」或 Enter。"
            else:
                x, y, right, bottom = self.box
                prefix = "修改中的框" if editing_box else "待儲存框"
                preview_text = f"{prefix} · {right - x:.0f} × {bottom - y:.0f} px"
        elif self.preview is None:
            preview_text = "橘色為候選預覽，接受後儲存。"
        elif not candidate_current:
            preview_text = "仍顯示上次預覽，更新候選後才可接受。"
        elif self.show_raw.get():
            preview_text = "正在看原始遮罩。取消勾選即可修整或儲存。"
        else:
            stats = self.preview.cleanup_stats
            preview_text = (f"修整 {stats['changed_pixels']:,} px · 孤島 {stats['removed_components']} · 孔洞 {stats['filled_holes']}"
                            if stats['changed_pixels'] else "自動修整未變動")
            if self.manual_pixels:
                preview_text += f" · 手動 {self.manual_pixels:,} px"
        self.preview_text.configure(text=preview_text, fg=CANDIDATE if self.preview or (box_mode and self.box) else MUTED)
        enable(self.accept_button, repair_available and self.brush_stroke is None)
        self.accept_button.configure(text="儲存修整" if editing_saved else "接受並儲存")
        enable(self.discard_button, (self.box is not None if box_mode else self.preview is not None))
        self.discard_button.configure(text="捨棄框選" if box_mode else "捨棄修整" if editing_saved else "捨棄候選")
        edit_text = "修改所選框" if selected and self._annotation_kind(selected) == "bbox" else "修整遮罩"
        self.repair_selected_button.configure(text=edit_text)
        enable(self.repair_selected_button, selected is not None and not busy and self.preview is None and self.box is None)
        enable(self.delete_button, has_asset and selected is not None and not disk_busy and not editing_saved and not editing_box)
        enable(self.review_button, has_asset and self.preview is None and not busy and not self.box)
        review_text = "標記畫格已審核" if has_asset and self.asset.get("video") else "標記已審核"
        self.review_button.configure(text="改回待審核" if has_asset and self.asset.get("reviewed") else review_text)

    def close(self):
        if self.task_kind == "video":
            self.closing = True
            if self.video_import is not None and self.video_import.cancel():
                self.video_stop.state(["disabled"])
                self._message("停止影片處理後會關閉視窗。")
            else:
                self._message("畫格儲存完成後會關閉視窗。")
            return
        if self.task_kind in {"import", "export"}:
            self.closing = True
            self._message("檔案寫入完成後會關閉視窗。")
            return
        self.alive = False
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, help="open a project directory")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="path to a local SAM 2.1 model")
    parser.add_argument("--smoke-ui", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = tk.Tk()
    app = WizardApp(root, args.model_dir)
    if args.project:
        app.open_path(args.project)
    if args.smoke_ui:
        root.after(200, app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
