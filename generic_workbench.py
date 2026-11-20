"""Run and review configurable detection jobs in the main Tk window."""

from collections import Counter
import json
from pathlib import Path
from time import time_ns
import tkinter as tk
from tkinter import filedialog, font as tkfont, ttk

from PIL import Image, ImageTk

from batch_audit_view import BatchAuditView
from batch_input_summary import build_input_rows, input_class_counts
from batch_workflow import record_correction
from generic_confidence import candidate_confidence
from generic_job import GenericJob, create_job, parse_classes
from generic_results_view import GenericResultsView
from generic_runner import digest, infer_job, job_progress, latest_training, train_job


BG, PANEL, TEXT, MUTED = "#101820", "#18242c", "#ebf2f3", "#a8bbc3"
FONT = "Microsoft JhengHei UI"
REASONS = {
    "no_candidate": "沒有適用候選", "no_accepted_boxes": "未確認到可採用框", "inventory_uncertain": "整張影像有未確認目標",
    "scan_uncertain": "整張影像有未確認目標", "unmatched_target": "疑似漏框", "conflicting_candidates": "候選互相衝突",
    "candidate_uncertain": "框位或類別待確認", "bad_extent": "框選範圍不合適", "empty_unverified": "空畫格需確認",
    "competing_extents": "同物件有範圍不同的競爭框", "conflicting_classes": "高度重疊的框有類別衝突",
    "unmatched_scene_target": "全圖掃描有疑似目標缺框", "uncertain_review": "兩次覆核的證據不足",
    "uncertain_scene_scan": "無法確認全圖目標是否完整", "invalid_scene_scan": "全圖目標清單無法解析",
    "invalid_proposal": "候選分數或座標無效", "low_score_proposal": "MM 分數低於候選門檻",
    "different_object": "候選為其他物件或其他類別", "duplicate_proposal": "已由同類重複框涵蓋",
    "better_supported_extent": "已採用另一個範圍更合適的候選",
    "manual_review_pending": "已保留人工框，請確認這張圖的作業類別是否全部標好",
}
VERDICTS = {"accepted": "採用", "rejected": "拒絕", "unresolved": "待審", "pending": "尚未裁決"}
FRAME_STATES = {"pending": "待審", "ai_ready": "AI 採用", "approved": "已複核", "excluded": "已扣留",
                "awaiting_inference": "未判斷", "independent_test": "獨立驗收"}
REVIEW_FILTERS = {"全部影像": None, "待審": "pending", "AI 採用": "ai_ready", "已複核": "approved", "已扣留": "excluded"}
INDEPENDENT_FILTER = "獨立驗收影像"
SHADOW_FILTER = "影子待審"
SHADOW_DISPOSITION = {"auto_candidate": "暫列自動候選", "review": "待人工複核", "held": "扣留"}
VOTE_OPTIONS = {"A": "採用", "B": "框位有誤", "C": "其他物件", "D": "證據不足"}
PANE_RATIO = (2, 2, 6)


def percentage(value):
    return f"{value:.1%}" if isinstance(value, (float, int)) and not isinstance(value, bool) else "—"


def review_priority(row):
    score = (row.get("confidence") or {}).get("score")
    return row["status"] != "pending", -1 if score is None else score


class BoxCanvas(tk.Canvas):
    def __init__(self, parent, changed):
        super().__init__(parent, bg="#0a1017", highlightthickness=0, width=1, height=145)
        self.changed = changed
        self.picture = self.photo = self.start = None
        self.boxes, self.selected, self.anchor, self.overlay = [], None, None, ()
        self.label, self.editable = "", False
        self.scale, self.offset = 1, (0, 0)
        self.bind("<Configure>", lambda _event: self.render())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

    def show(self, picture, boxes, *, editable=False, overlay=()):
        self.picture, self.boxes, self.selected, self.editable = picture, boxes, None, editable
        self.start, self.anchor, self.overlay = None, None, overlay
        self.render()

    def render(self):
        self.delete("all")
        if self.picture is None:
            self.create_text(20, 20, anchor="nw", text="選擇畫格以檢視", fill=MUTED)
            return
        width, height = self.picture.size
        available_width, available_height = self.winfo_width(), self.winfo_height()
        self.scale = max(.001, min(available_width / width, available_height / height))
        shown = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        self.offset = ((available_width - shown[0]) / 2, (available_height - shown[1]) / 2)
        self.photo = ImageTk.PhotoImage(self.picture.resize(shown, Image.Resampling.BILINEAR), master=self)
        self.create_image(*self.offset, image=self.photo, anchor="nw")
        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self._canvas_box(box["xyxy"])
            manual = box.get("source") == "manual"
            color = "#ffe194" if index == self.selected else "#a6beff" if manual else "#8de7d2"
            self.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if manual else 2)
            label = f"人工 · {box['label']}" if manual else box["label"]
            self.create_text(x1 + 3, y1 + 2, anchor="nw", text=label, fill=color, font=(FONT, 9, "bold"))
        for box in self.overlay:
            x1, y1, x2, y2 = self._canvas_box(box["xyxy"])
            self.create_rectangle(x1, y1, x2, y2, outline="#ffbd88", width=2, dash=(4, 3))
            self.create_text(x1 + 3, y2 - 2, anchor="sw", text=box["label"], fill="#ffbd88", font=(FONT, 9, "bold"))

    def _canvas_box(self, xyxy):
        return tuple(value * self.scale + self.offset[index % 2] for index, value in enumerate(xyxy))

    def _image_point(self, event):
        return tuple(max(0, min(limit, (value - offset) / self.scale))
                     for value, offset, limit in zip((event.x, event.y), self.offset, self.picture.size, strict=True))

    def _press(self, event):
        if self.editable and self.picture is not None:
            self.start = self._image_point(event)
            if self.selected is not None:
                x1, y1, x2, y2 = self._canvas_box(self.boxes[self.selected]["xyxy"])
                corners = ((x1, y1, 2, 3), (x1, y2, 2, 1), (x2, y1, 0, 3), (x2, y2, 0, 1))
                for x, y, opposite_x, opposite_y in corners:
                    if abs(event.x - x) <= 8 and abs(event.y - y) <= 8:
                        box = self.boxes[self.selected]["xyxy"]
                        self.start = box[opposite_x], box[opposite_y]
                        self.anchor = self.selected
                        break

    def _drag(self, event):
        if self.start is not None:
            self.delete("draft")
            self.create_rectangle(*self._canvas_box((*self.start, *self._image_point(event))),
                                  outline="#ffe194", width=2, dash=(4, 3), tags="draft")

    def _release(self, event):
        if self.start is None:
            return
        start, end = self.start, self._image_point(event)
        anchor, self.start, self.anchor = self.anchor, None, None
        xl, xr = sorted((start[0], end[0]))
        yt, yb = sorted((start[1], end[1]))
        if anchor is not None:
            if xr - xl >= 2 and yb - yt >= 2:
                self.boxes[anchor]["xyxy"] = [xl, yt, xr, yb]
                self.changed(True)
        elif xr - xl >= 2 and yb - yt >= 2 and self.label:
            self.boxes.append({"label": self.label, "xyxy": [xl, yt, xr, yb], "source": "manual"})
            self.selected = len(self.boxes) - 1
            self.changed(True)
        else:
            matching = [(index, (x2 - x1) * (y2 - y1)) for index, box in enumerate(self.boxes)
                        for x1, y1, x2, y2 in [box["xyxy"]] if x1 <= end[0] <= x2 and y1 <= end[1] <= y2]
            self.selected = min(matching, key=lambda entry: entry[1])[0] if matching else None
            self.changed(False)
        self.render()


class GenericWorkbench(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.project = self.job = None
        self.frame_ids, self.input_ids = (), ()
        self.reference = None
        self.dirty = False
        self.class_editor_open = False
        self.class_in_edit = None
        self.class_overrides = {}
        self.class_specs = {}
        self.input_assets, self.job_rows, self.scope_rows = {}, {}, []
        self.enabled_classes = None
        self.preview_open = False
        self.audit_open = False
        self.panes_initialized = False
        self.pane_layout_callback = None
        self.editing_frame = None
        self.evaluation_rows = []
        self.selected_frame = None
        self.mode = "ai"
        self.class_name = tk.StringVar(master=self)
        self.only_pending = tk.BooleanVar(master=self, value=False)
        self.review_filter = tk.StringVar(master=self, value="全部影像")
        self.selected_input_text = tk.StringVar(master=self)
        self.auto_train = tk.BooleanVar(master=self, value=False)
        self.shadow_sam = tk.BooleanVar(master=self, value=False)
        self.status = tk.StringVar(master=self)
        self._build()

    def _label(self, parent, text="", *, variable=None, wrap=0, size=10, fg=TEXT, bg=BG, bold=False):
        return tk.Label(parent, text=text, textvariable=variable, bg=bg, fg=fg, anchor="w", justify="left",
                        font=(FONT, size, "bold" if bold else "normal"), wraplength=wrap)

    def _build(self):
        self.page_bar = bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", pady=(0, 6))
        self.page_title = self._label(bar, "AI 批量標註", size=17, bold=True)
        self.page_title.pack(side="left")
        self.import_still = ttk.Button(bar, text="匯入圖片", command=lambda: self.app.import_images(return_to_ai=True))
        self.import_video = ttk.Button(bar, text="匯入影片畫格", command=lambda: self.app.import_video(keyframes=True))
        self.import_video.pack(side="right")
        self.import_still.pack(side="right", padx=5)
        self.status_label = self._label(self, variable=self.status, fg=MUTED, wrap=1050)
        self.status_label.pack(fill="x", pady=(0, 6))
        style = ttk.Style(self)
        style.configure("BatchCompact.TButton", padding=(3, 3))
        style.configure("BatchCompact.Primary.TButton", padding=(6, 4))
        self.batch = tk.Frame(self, bg=BG)
        self.batch.pack(fill="both", expand=True)
        self.results = tk.Frame(self, bg=BG, padx=12, pady=10)
        self._batch_toolbar()
        self._build_class_editor()
        self.panes = tk.PanedWindow(self.batch, orient="horizontal", bg="#30414c", bd=0, sashwidth=8, sashpad=0,
                                    relief="flat", sashrelief="raised", showhandle=False, opaqueresize=True)
        self.panes.pack(fill="both", expand=True)
        self.input_pane = tk.Frame(self.panes, bg=PANEL, padx=7, pady=6, width=240)
        self.class_pane = tk.Frame(self.panes, bg=PANEL, padx=5, pady=6, width=120)
        self.metrics_pane = tk.Frame(self.panes, bg=PANEL, padx=7, pady=6, width=840)
        for pane, minimum in zip((self.input_pane, self.class_pane, self.metrics_pane), (125, 96, 360), strict=True):
            self.panes.add(pane, minsize=minimum, stretch="always")
        self.panes.bind("<Configure>", self._initialize_panes)
        self._build_inputs()
        self._build_classes()
        self._build_metrics()
        self._results_tab()

    def _initialize_panes(self, event):
        if self.panes_initialized or self.pane_layout_callback is not None or event.width < 640:
            return
        self.pane_layout_callback = self.after_idle(self._apply_default_panes)

    def _apply_default_panes(self):
        self.pane_layout_callback = None
        width = self.panes.winfo_width()
        if not self.panes.winfo_ismapped() or width < 640:
            return
        usable = width - 2 * int(self.panes.cget("sashwidth"))
        counts = [round(usable * part / sum(PANE_RATIO)) for part in PANE_RATIO[:2]]
        first, second = counts[0], sum(counts) + int(self.panes.cget("sashwidth"))
        self.panes.sash_place(0, first, 0)
        self.panes.sash_place(1, second, 0)
        self.panes_initialized = True

    def show_mode(self, mode):
        self.mode = mode
        training = mode == "train"
        self.page_title.configure(text="訓練與驗收" if training else "AI 批量標註")
        if training:
            self.import_still.pack_forget()
            self.import_video.pack_forget()
            self.batch.pack_forget()
            self.results.pack(fill="both", expand=True)
        else:
            self.import_video.pack(side="right")
            self.import_still.pack(side="right", padx=5)
            self.results.pack_forget()
            self.batch.pack(fill="both", expand=True)
        self._fit_workspace_chrome()

    def _fit_workspace_chrome(self):
        if self.mode == "ai" and self.preview_open:
            self.page_bar.pack_forget()
            self.batch_actions.pack_forget()
        else:
            self.page_bar.pack(fill="x", pady=(0, 6), before=self.status_label)
            self.batch_actions.pack(fill="x", pady=(0, 6), before=self.panes)

    def _batch_toolbar(self):
        self.batch_actions = previous = tk.Frame(self.batch, bg=BG)
        previous.pack(fill="x", pady=(0, 6))
        self._label(previous, "既有作業", fg=MUTED).pack(side="left", padx=(0, 8))
        self.jobs_picker = ttk.Combobox(previous, state="readonly")
        self.jobs_picker.pack(side="left", fill="x", expand=True)
        self.jobs_picker.bind("<<ComboboxSelected>>", self._open_job)
        self.start_button = ttk.Button(previous, text="開始離線 AI 標註", style="Primary.TButton", command=self._start)
        self.start_button.pack(side="right")
        self.resume_button = ttk.Button(previous, text="續跑作業", command=self._run_inference)
        self.resume_button.pack(side="right", padx=5)
        self.shadow_sam_option = ttk.Checkbutton(previous, text="SAM 框界抽查", variable=self.shadow_sam)
        self.shadow_sam_option.pack(side="right", padx=5)

    def _build_inputs(self):
        host = self.input_pane
        header = tk.Frame(host, bg=PANEL)
        header.pack(fill="x")
        self._label(header, "本次輸入", bold=True, bg=PANEL).pack(side="left")
        self.select_all_button = ttk.Button(header, text="全選", style="BatchCompact.TButton",
                                            command=lambda: self.select_inputs(self.input_ids))
        self.select_all_button.pack(side="right")
        self._label(host, variable=self.selected_input_text, fg=MUTED, bg=PANEL).pack(fill="x", pady=(5, 5))
        listing = tk.Frame(host, bg=PANEL)
        listing.pack(fill="both", expand=True)
        self.inputs = tk.Listbox(listing, width=1, selectmode="extended", exportselection=False, bg="#101b24",
                                 fg=TEXT, selectbackground="#285750", borderwidth=0, font=(FONT, 10))
        scroll = ttk.Scrollbar(listing, command=self.inputs.yview)
        self.inputs.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.inputs.pack(side="left", fill="both", expand=True)
        self.inputs.bind("<<ListboxSelect>>", self._inputs_changed)
        self._label(host, "Ctrl／Shift 多選", fg=MUTED, bg=PANEL, size=9).pack(fill="x", pady=(5, 0))
        self.reference_button = ttk.Button(host, text="附獨立驗收清冊", style="BatchCompact.TButton", command=self._reference)
        self.reference_button.pack(fill="x", pady=(6, 0))
        self.reference_text = self._label(host, fg=MUTED, bg=PANEL, size=9, wrap=100)
        self.reference_text.pack(fill="x")

    def _build_classes(self):
        host = self.class_pane
        self._label(host, "物件類別", bold=True, bg=PANEL).pack(fill="x")
        tools = tk.Frame(host, bg=PANEL)
        tools.pack(fill="x", pady=(4, 3))
        self.add_class_button = ttk.Button(tools, text="新增", style="BatchCompact.TButton", width=4,
                                          command=lambda: self._show_class_editor())
        self.edit_class_button = ttk.Button(tools, text="編輯", style="BatchCompact.TButton", width=4,
                                           command=self._edit_selected_class)
        self.add_class_button.pack(side="left", fill="x", expand=True)
        self.edit_class_button.pack(side="left", fill="x", expand=True, padx=(2, 0))
        self.class_total = self._label(host, fg=MUTED, bg=PANEL, size=9)
        self.class_total.pack(fill="x", pady=(3, 4))
        listing = tk.Frame(host, bg=PANEL)
        listing.pack(fill="both", expand=True)
        self.class_table = ttk.Treeview(listing, columns=("use", "label", "count"), show="headings",
                                        selectmode="browse", style="GenericReview.Treeview")
        for column, title, anchor in (("use", "使用", "center"), ("label", "類別", "w"), ("count", "框數", "e")):
            self.class_table.heading(column, text=title, anchor=anchor)
            self.class_table.column(column, stretch=column == "label", anchor=anchor)
        scroll = ttk.Scrollbar(listing, command=self.class_table.yview)
        horizontal = ttk.Scrollbar(listing, orient="horizontal", command=self.class_table.xview)
        self.class_table.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        self.class_table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        listing.rowconfigure(0, weight=1)
        listing.columnconfigure(0, weight=1)
        style_font = ttk.Style(self).lookup("GenericReview.Treeview", "font") or "TkDefaultFont"
        self.class_table_font = tkfont.Font(root=self, font=style_font)
        self.class_table.bind("<<TreeviewSelect>>", lambda _event: self._update_state())
        self.class_table.bind("<ButtonRelease-1>", self._choose_class)
        self.class_table.bind("<space>", lambda _event: self._toggle_selected_class())
        self._label(host, "勾選本批類別", fg=MUTED, bg=PANEL, size=9).pack(fill="x", pady=(5, 0))

    def _build_class_editor(self):
        self.class_editor = tk.Frame(self.batch, bg=PANEL, padx=8, pady=5)
        self.class_editor_title = self._label(self.class_editor, bg=PANEL, bold=True)
        self.class_editor_title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 3))
        self.class_fields = {name: tk.StringVar(master=self) for name in ("label", "prompt", "description")}
        for index, (key, label) in enumerate(zip(self.class_fields, ("類別名稱", "MM 英文提示詞", "Qwen 判斷規則"), strict=True)):
            self._label(self.class_editor, label, fg=MUTED, bg=PANEL, size=9).grid(row=1, column=index, sticky="w")
            field = ttk.Entry(self.class_editor, textvariable=self.class_fields[key])
            field.grid(row=2, column=index, padx=(0, 6), sticky="ew")
            self.class_editor.columnconfigure(index, weight=1 if index < 2 else 2)
            if key == "label":
                self.class_label_entry = field
        ttk.Button(self.class_editor, text="儲存", style="BatchCompact.TButton",
                   command=self._save_class).grid(row=2, column=3, padx=(0, 5))
        ttk.Button(self.class_editor, text="取消", style="BatchCompact.TButton",
                   command=self._hide_class_editor).grid(row=2, column=4)

    def _build_metrics(self):
        sidebar = self.metrics_pane
        self.metrics_filters = filters = tk.Frame(sidebar, bg=PANEL)
        filters.pack(fill="x")
        self.metrics_title = self._label(filters, "指標", bold=True, bg=PANEL)
        self.metrics_title.pack(side="left", padx=(0, 10))
        self.preview_button = ttk.Button(filters, text="開啟圖像檢查", width=0, style="BatchCompact.TButton",
                                         command=self._toggle_preview)
        self.preview_button.pack(side="right")
        self.audit_button = ttk.Button(filters, text="批次驗收", style="BatchCompact.TButton",
                                       command=self._toggle_audit)
        self.audit_button.pack(side="right", padx=(4, 0))
        self.filter_picker = ttk.Combobox(filters, textvariable=self.review_filter, state="readonly", width=14,
                                         values=tuple(REVIEW_FILTERS))
        self.filter_picker.pack(side="right", padx=6)
        self.filter_picker.bind("<<ComboboxSelected>>", self._choose_filter)
        self.queue_summary = self._label(sidebar, fg=MUTED, bg=PANEL)
        self.queue_summary.pack(fill="x", pady=5)
        self.metrics_listing = listing = tk.Frame(sidebar, bg=PANEL)
        listing.pack(fill="both", expand=True)
        style = ttk.Style(self)
        style.configure("GenericReview.Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=27)
        style.map("GenericReview.Treeview", background=[("selected", "#285750")], foreground=[("selected", TEXT)])
        columns = (("name", "影像", 136), ("state", "狀態", 70), ("score", "整張信心", 76),
                   ("mm", "MM", 55), ("qwen", "Qwen", 58), ("human", "人工框", 60),
                   ("ai", "AI 框", 48), ("reason", "待審原因", 180))
        self.queue = ttk.Treeview(listing, columns=[column for column, _, _ in columns], show="headings",
                                  selectmode="browse", height=5, style="GenericReview.Treeview")
        for column, title, width in columns:
            anchor = "w" if column in {"name", "reason"} else "center"
            self.queue.heading(column, text=title, anchor=anchor)
            self.queue.column(column, width=width, minwidth=width // 2, stretch=column in {"name", "reason"}, anchor=anchor)
        self.queue.tag_configure("pending", foreground="#ffd68b")
        self.queue.tag_configure("excluded", foreground=MUTED)
        self.queue.tag_configure("approved", foreground="#96e5cb")
        scrollbar = ttk.Scrollbar(listing, orient="vertical", command=self.queue.yview)
        horizontal = ttk.Scrollbar(listing, orient="horizontal", command=self.queue.xview)
        self.queue.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        self.queue.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        listing.rowconfigure(0, weight=1)
        listing.columnconfigure(0, weight=1)
        self.queue.bind("<<TreeviewSelect>>", self._select_frame)
        self.metrics_navigation = navigation = tk.Frame(sidebar, bg=PANEL)
        navigation.pack(fill="x", pady=(6, 0))
        self.prev_frame = ttk.Button(navigation, text="上一張", width=0, style="BatchCompact.TButton",
                                     command=lambda: self._move_frame(-1))
        self.next_frame = ttk.Button(navigation, text="下一張", width=0, style="BatchCompact.TButton",
                                     command=lambda: self._move_frame(1))
        self.prev_frame.pack(side="left")
        self.queue_position = self._label(navigation, fg=MUTED, bg=PANEL)
        self.queue_position.pack(side="left", padx=6)
        self.next_frame.pack(side="left")
        self.metrics_summary = summary = tk.Frame(sidebar, bg=PANEL)
        summary.pack(fill="x", pady=(5, 3))
        self.case_header = case = tk.Frame(summary, bg=PANEL)
        case.pack(fill="x")
        self.case_text = self._label(case, wrap=760, bold=True, bg=PANEL)
        self.case_text.pack(side="left", fill="x", expand=True)
        self.preview_score = self._label(summary, fg="#96e5cb", bg=PANEL, size=9)
        self.confidence_text = self._label(summary, wrap=760, fg="#96e5cb", bg=PANEL)
        self.confidence_text.pack(fill="x", pady=(3, 2))
        self.decision_text = self._label(summary, wrap=760, fg=MUTED, bg=PANEL)
        self.decision_text.pack(fill="x", pady=(0, 4))
        self.preview = detail = tk.Frame(sidebar, bg=BG, padx=5, pady=5)
        style.configure("BatchPreview.TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 3))
        style.configure("BatchPreview.TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(10, 3))
        style.map("BatchPreview.TNotebook.Tab", background=[("selected", "#29454f")],
                  foreground=[("selected", TEXT), ("disabled", "#63747e")])
        self.preview_tabs = ttk.Notebook(detail, style="BatchPreview.TNotebook")
        self.image_tab = tk.Frame(self.preview_tabs, bg=BG)
        self.details_tab = tk.Frame(self.preview_tabs, bg=BG)
        self.preview_tabs.add(self.image_tab, text="圖像")
        self.preview_tabs.add(self.details_tab, text="裁決明細", state="disabled")
        self.preview_tabs.enable_traversal()
        self.preview_body = body = tk.Frame(detail, bg=BG)
        self.canvas = BoxCanvas(body, self._canvas_changed)
        self.preview_notice = self._label(detail, fg="#ffd68b", size=9)
        self.candidate_panel = tk.Frame(body, bg=PANEL, padx=5, pady=4, width=260)
        self.candidate_controls = candidates = tk.Frame(self.candidate_panel, bg=PANEL)
        self.candidate_picker = ttk.Combobox(candidates, state="readonly")
        self.candidate_picker.pack(fill="x")
        self.candidate_picker.bind("<<ComboboxSelected>>", self._candidate_selected)
        self.adopt_button = ttk.Button(candidates, text="採用此候選", style="BatchCompact.TButton",
                                       command=self._adopt_candidate)
        self.adopt_button.pack(fill="x", pady=(3, 0))
        self.detail_body = tk.Frame(self.candidate_panel, bg=PANEL)
        self.detail_body.pack(fill="both", expand=True)
        detail_scroll = ttk.Scrollbar(self.detail_body)
        self.detail_text = tk.Text(self.detail_body, bg=PANEL, fg=TEXT, wrap="word", width=1, height=1, padx=4,
                                   spacing3=3,
                                   relief="flat", highlightthickness=0, state="disabled", font=(FONT, 9),
                                   yscrollcommand=detail_scroll.set)
        self.detail_text.tag_configure("heading", foreground="#96e5cb", font=(FONT, 9, "bold"))
        detail_scroll.configure(command=self.detail_text.yview)
        detail_scroll.pack(side="right", fill="y")
        self.detail_text.pack(side="left", fill="both", expand=True)
        self.candidate_notes = ""
        self.candidate_panel.pack_propagate(False)
        sidebar.bind("<Configure>", self._wrap_review_detail)
        self.preview_actions = actions = tk.Frame(detail, bg=BG)
        actions.pack(side="bottom", fill="x", pady=(4, 0))
        self.preview_tabs.pack(fill="both", expand=True)
        self.canvas.pack(fill="both", expand=True)
        body.pack(in_=self.image_tab, fill="both", expand=True)
        self.preview_tabs.bind("<<NotebookTabChanged>>", self._preview_tab_changed)
        self.edit_tools = tk.Frame(actions, bg=BG)
        self.submit_tools = tk.Frame(actions, bg=BG)
        self.narrow_actions = False
        self.edit_tools.pack(side="left")
        self.submit_tools.pack(side="right")
        self.label_picker = ttk.Combobox(self.edit_tools, textvariable=self.class_name, state="readonly", width=13)
        self.label_picker.pack(side="left")
        self.label_picker.bind("<<ComboboxSelected>>", self._change_label)
        self.correct_button = ttk.Button(self.edit_tools, text="更正此張", width=0, style="BatchCompact.TButton",
                                         command=self._start_correction)
        self.delete_button = ttk.Button(self.edit_tools, text="刪框", width=0, style="BatchCompact.TButton",
                                        command=self._delete_box)
        self.discard_button = ttk.Button(self.edit_tools, text="捨棄修正", width=0, style="BatchCompact.TButton",
                                         command=lambda: self._load_frame(self.selected_frame))
        self.exclude_button = ttk.Button(self.submit_tools, text="扣留整張", width=0, style="BatchCompact.TButton",
                                         command=lambda: self._submit(exclude=True))
        self.confirm_button = ttk.Button(self.submit_tools, text="確認整張並到下一張", width=0, command=self._submit,
                                         style="BatchCompact.Primary.TButton")
        self.correct_button.pack(side="left", padx=5)
        self.delete_button.pack(side="left", padx=(0, 5))
        self.discard_button.pack(side="left")
        self.exclude_button.pack(side="right", padx=5)
        self.confirm_button.pack(side="right")
        self.batch_audit = BatchAuditView(sidebar, canvas_class=BoxCanvas, label_factory=self._label, background=PANEL)
        self.review_actor = ttk.Entry(filters, textvariable=self.batch_audit.actor, width=10)
        self.review_actor.pack(side="right", padx=(4, 0))
        self.actor_label = self._label(filters, "覆核者", bg=PANEL, fg=MUTED, size=9)
        self.actor_label.pack(side="right")

    def _toggle_audit(self):
        if self.dirty or self.class_editor_open or self.app.task is not None:
            return
        self._set_audit(not self.audit_open)

    def _set_audit(self, shown):
        if shown == self.audit_open:
            return
        self.audit_open = shown
        self.audit_button.configure(text="返回指標" if shown else "批次驗收")
        sections = (self.filter_picker, self.preview_button, self.review_actor, self.actor_label, self.queue_summary,
                    self.metrics_listing, self.metrics_navigation, self.metrics_summary)
        if shown:
            self._set_preview(False)
            for widget in sections:
                widget.pack_forget()
            self.batch_audit.pack(fill="both", expand=True, pady=(4, 0))
            self.batch_audit.load(self.job)
        else:
            self.batch_audit.pack_forget()
            self.preview_button.pack(side="right", before=self.audit_button)
            self.filter_picker.pack(side="right", padx=6)
            self.review_actor.pack(side="right", padx=(4, 0))
            self.actor_label.pack(side="right")
            self.queue_summary.pack(fill="x", pady=5)
            self.metrics_listing.pack(fill="both", expand=True)
            self.metrics_navigation.pack(fill="x", pady=(6, 0))
            self.metrics_summary.pack(fill="x", pady=(5, 3))

    def _wrap_review_detail(self, event):
        width = max(180, event.width - 14)
        for label in (self.confidence_text, self.decision_text):
            label.configure(wraplength=width)
        self.preview_notice.configure(wraplength=width)
        candidate_width = min(280, max(125, int(width * 0.35)))
        self.candidate_panel.configure(width=candidate_width)
        self._fit_preview_layout(width)
        narrow = width < self.edit_tools.winfo_reqwidth() + self.submit_tools.winfo_reqwidth()
        if narrow != self.narrow_actions:
            self.narrow_actions = narrow
            self.edit_tools.pack_forget()
            self.submit_tools.pack_forget()
            if narrow:
                self.edit_tools.pack(side="top", anchor="w")
                self.submit_tools.pack(side="top", anchor="e", pady=(5, 0))
            else:
                self.edit_tools.pack(side="left")
                self.submit_tools.pack(side="right")

    def _fit_preview_layout(self, width):
        self.metrics_title.configure(text="圖像檢查" if self.preview_open else "指標")
        wide = width >= 560
        self.metrics_navigation.pack_forget()
        self.metrics_summary.pack_forget()
        self.preview_score.pack_forget()
        if self.preview_open:
            self.metrics_listing.pack_forget()
            self.queue_summary.pack_forget()
            self.confidence_text.pack_forget()
            self.decision_text.pack_forget()
            if width >= 455:
                self.metrics_navigation.pack(in_=self.metrics_filters, side="left")
            else:
                self.metrics_navigation.pack(fill="x", before=self.preview, pady=(4, 0))
            self.metrics_summary.pack(fill="x", before=self.preview, pady=(5, 3))
            if wide:
                self.preview_score.pack(in_=self.case_header, side="left", padx=(6, 0))
            else:
                self.preview_score.pack(fill="x", pady=(1, 0), after=self.case_header)
        else:
            self.metrics_listing.pack_forget()
            self.queue_summary.pack(fill="x", pady=5, after=self.metrics_filters)
            self.metrics_summary.pack(side="bottom", fill="x", pady=(5, 3))
            self.metrics_navigation.pack(side="bottom", fill="x", pady=(6, 0))
            self.metrics_listing.pack(fill="both", expand=True)
            self.confidence_text.pack(fill="x", pady=(3, 2))
            self.decision_text.pack(fill="x", pady=(0, 4))
        occupied = 14
        if self.preview_open and wide:
            occupied += self.preview_score.winfo_reqwidth() + 6
        self.case_text.configure(wraplength=max(120, width - occupied))

    def _results_tab(self):
        sources = tk.Frame(self.results, bg=BG)
        sources.pack(fill="x", pady=(0, 7))
        self._label(sources, "訓練作業", fg=MUTED).pack(side="left", padx=(0, 8))
        self.training_jobs = ttk.Combobox(sources, state="readonly")
        self.training_jobs.pack(side="left", fill="x", expand=True)
        self.training_jobs.bind("<<ComboboxSelected>>", self._open_job)
        self.manual_train_button = ttk.Button(sources, text="用專案人工標註建立作業", command=self._manual_training)
        self.manual_train_button.pack(side="left", padx=(6, 0))
        actions = tk.Frame(self.results, bg=BG)
        actions.pack(fill="x")
        self.train_button = ttk.Button(actions, text="匯出並從零訓練 YOLO", style="Primary.TButton", command=self._train)
        self.train_button.pack(side="right")
        self.eval_button = ttk.Button(actions, text="查看獨立驗收框", command=self._show_evaluation, state="disabled")
        self.eval_button.pack(side="right", padx=5)
        self._label(actions, "輪數").pack(side="left")
        self.epochs = tk.StringVar(master=self, value="20")
        ttk.Spinbox(actions, from_=1, to=400, increment=1, width=6, textvariable=self.epochs).pack(side="left", padx=6)
        ttk.Checkbutton(actions, text="例外處理完成後自動訓練", variable=self.auto_train).pack(side="left", padx=5)
        self.result_view = GenericResultsView(self.results)
        self.result_view.pack(fill="both", expand=True, pady=(9, 4))

    def attach(self, project):
        if self.project is not project:
            self._set_audit(False)
            self.batch_audit.load(None)
            self.project, self.job, self.reference = project, None, None
            self.dirty = False
            self.class_overrides.clear()
            self.enabled_classes = None
            self._hide_class_editor(refresh=False)
            self._set_preview(False, load=False)
            self.selected_frame = None
            self.evaluation_rows = []
            self.only_pending.set(False)
            self.review_filter.set("全部影像")
            self.inputs.selection_clear(0, "end")
            self.reference_text.configure(text="")
        self._refresh()

    def _refresh(self, *, preferred_frame=None):
        selected = {self.input_ids[index] for index in self.inputs.curselection() if index < len(self.input_ids)}
        assets = self.project.assets if self.project else []
        self.input_assets = {asset["id"]: asset for asset in assets}
        self.input_ids = tuple(asset["id"] for asset in assets)
        self.inputs.delete(0, "end")
        for index, asset in enumerate(assets):
            self.inputs.insert("end", asset["name"])
            if asset["id"] in selected:
                self.inputs.selection_set(index)
        store = self.project.directory / "ai-jobs" if self.project else None
        paths = (path.parent for path in store.glob("*/manifest.json")) if store else ()
        choices = sorted((self._job_choice(path) for path in paths), reverse=True)
        self.job_locations = [path for _, _, path in choices]
        self.job_choices = {path: title for _, title, path in choices}
        for picker in (self.jobs_picker, self.training_jobs):
            picker.configure(values=tuple(self.job_choices.values()))
        self._set_job_choice()
        if self.job:
            self.job.refresh()
            summary = self.job.summary()
            self.status.set(f"本批 {summary['frames']} 張 · 待審 {summary['needs_review']} 張 · 已複核 {summary['approved']} 張")
        else:
            self.status.set(f"已匯入 {len(assets)} 張，選資料與類別後可批次標註。" if self.project else "請先建立或開啟專案。")
        self.job_rows = {row["id"]: row for row in self.job.frame_rows()} if self.job else {}
        self._filter_frames(preferred=preferred_frame)
        if self.audit_open:
            self.batch_audit.load(self.job)
        self._update_state()
        self._render_results()

    def select_inputs(self, identifiers):
        if self.dirty or self.class_editor_open or self.app.task is not None:
            return
        self._mark_inputs(identifiers)
        self._inputs_changed()

    def _mark_inputs(self, identifiers):
        wanted = set(identifiers)
        self.inputs.selection_clear(0, "end")
        for index, ident in enumerate(self.input_ids):
            if ident in wanted:
                self.inputs.selection_set(index)

    def _select_job_inputs(self):
        self._mark_inputs(frame["id"] for frame in self.job.manifest["frames"])

    def focus_asset(self, identifier):
        if self.dirty or self.class_editor_open or self.app.task is not None:
            return
        current = [self.job] if self.job else []
        remaining = (path for path in self.job_locations if self.job is None or path != self.job.directory)
        for job in current:
            if self._focus_job_frame(job, identifier):
                return
        for path in remaining:
            try:
                job = GenericJob.open(path)
                if self._focus_job_frame(job, identifier):
                    return
            except (OSError, ValueError, RuntimeError):
                continue
        self.select_inputs((identifier,))
        self.status.set("已選好這張圖，可開啟圖像檢查或開始新批次。")

    def _focus_job_frame(self, job, identifier):
        if (job.manifest.get("purpose") == "manual_training"
                or not any(frame["id"] == identifier for frame in job.manifest["frames"])):
            return False
        if self.job is None or self.job.directory != job.directory:
            self.auto_train.set(False)
        self.job = job
        self._select_job_inputs()
        self._use_job_classes(job)
        self.only_pending.set(False)
        self.review_filter.set("全部影像")
        self.evaluation_rows = []
        self._set_preview(True, load=False)
        self._refresh(preferred_frame=identifier)
        return True

    def _selected_input_ids(self):
        picked = tuple(self.input_ids[index] for index in self.inputs.curselection())
        self.selected_input_text.set(f"已選 {len(picked)}／{len(self.input_ids)} 張" if picked else f"全部 {len(self.input_ids)} 張")
        return picked or self.input_ids

    def _inputs_changed(self, _event=None):
        if self.dirty or self.class_editor_open or self.app.task is not None:
            return
        if self.review_filter.get() == INDEPENDENT_FILTER:
            self.review_filter.set("全部影像")
            self.only_pending.set(False)
        selection = self.inputs.curselection()
        active = self.inputs.index("active") if selection else -1
        preferred = self.input_ids[active] if active in selection else None
        self._filter_frames(preferred=preferred)

    def set_batch_classes(self, classes, *, redraw=True):
        """Set classes for a new run without changing any existing job."""
        definitions = parse_classes(classes) if isinstance(classes, str) else classes
        self.class_overrides = {spec["label"]: dict(spec) for spec in definitions}
        self.enabled_classes = set(self.class_overrides)
        if redraw:
            self._show_class_totals(self.scope_rows)
            self._update_state()

    def _use_job_classes(self, job):
        stored = self.project.batch_class_settings
        current = ({"label": spec["label"], **stored[spec["label"]]} if spec["label"] in stored else spec
                   for spec in job.manifest["config"]["classes"])
        self.set_batch_classes(current, redraw=False)

    def _show_class_totals(self, rows):
        counts = input_class_counts(rows)
        stored = self.project.batch_class_settings if self.project else {}
        specs = {label: {"label": label, **stored.get(label, {})}
                 for label in self.project.labels} if self.project else {}
        specs.update(self.class_overrides)
        for label in counts:
            specs.setdefault(label, {"label": label})
        self.class_specs = specs
        self._size_class_columns(counts)
        enabled = self._enabled_labels()
        selected = self.class_table.selection()
        current = self.class_table.get_children()
        if current:
            self.class_table.delete(*current)
        for name in specs:
            self.class_table.insert("", "end", iid=name, values=("✓" if name in enabled else "·", name, counts[name]))
        if selected and selected[0] in specs:
            self.class_table.selection_set(selected[0])
        self.class_total.configure(text=f"{sum(counts.values())} 框 · {len(counts)} 類\n本批選 {len(enabled)} 類")

    def _size_class_columns(self, counts):
        measure = self.class_table_font.measure
        use_width = max(40, measure("使用") + 12)
        count_width = max(43, measure("框數") + 12, measure(str(max(counts.values(), default=0))) + 14)
        name_width = max(measure("類別"), max(map(measure, self.class_specs), default=0)) + 12
        self.class_table.column("use", width=use_width, minwidth=use_width)
        available = self.class_table.winfo_width() - use_width - count_width - 2
        self.class_table.column("label", width=max(available, name_width), minwidth=name_width)
        self.class_table.column("count", width=count_width, minwidth=count_width)

    def _enabled_labels(self):
        if self.enabled_classes is None:
            return set(self.class_specs)
        return self.enabled_classes & self.class_specs.keys()

    def _choose_class(self, event):
        if self.class_table.identify_column(event.x) == "#1":
            label = self.class_table.identify_row(event.y)
            if label:
                self._toggle_class(label)

    def _toggle_selected_class(self):
        if selected := self.class_table.selection():
            self._toggle_class(selected[0])
        return "break"

    def _toggle_class(self, label):
        if not self.project or self.dirty or self.class_editor_open or self.app.task is not None:
            return
        self.enabled_classes = self._enabled_labels() ^ {label}
        self._show_class_totals(self.scope_rows)
        self._update_state()

    def _edit_selected_class(self):
        if selected := self.class_table.selection():
            self._show_class_editor(selected[0])

    def _show_class_editor(self, label=None):
        if not self.project or self.dirty or self.class_editor_open or self.app.task is not None:
            return
        self.class_in_edit = label
        self.class_editor_open = True
        record = self.class_specs.get(label, {})
        for field, variable in self.class_fields.items():
            variable.set(record.get(field, ""))
        self.class_label_entry.configure(state="readonly" if label is not None else "normal")
        self.class_editor_title.configure(text="編輯類別設定 · 套用於之後的新批次" if label is not None else "新增共用類別")
        self.class_editor.pack(fill="x", pady=(0, 6), before=self.panes)
        self.class_label_entry.focus_set()
        self._update_state()

    def _hide_class_editor(self, *, refresh=True):
        self.class_editor_open = False
        self.class_editor.pack_forget()
        if refresh:
            self._update_state()

    def _save_class(self):
        if not self.class_editor_open or not self.project or self.app.task is not None:
            return
        entry = {field: variable.get() for field, variable in self.class_fields.items()}
        label = self.class_in_edit or entry["label"].strip()
        if self.class_in_edit is None and any(name.casefold() == label.casefold() for name in self.class_specs):
            self.status.set("已有這個類別，請選取後按「編輯」。")
            return
        try:
            label = self.project.save_batch_class(label, prompt=entry["prompt"], description=entry["description"])
            self.class_overrides[label] = {"label": label, **self.project.batch_class_settings[label]}
            if self.enabled_classes is not None and self.class_in_edit is None:
                self.enabled_classes.add(label)
            self._hide_class_editor(refresh=False)
            self.app.refresh_annotation_project()
            self._refresh()
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set(str(error))
            return
        self.class_table.selection_set(label)
        self.status.set(f"已儲存共用類別「{label}」。設定會套用於之後的新批次。")

    @staticmethod
    def _job_choice(path):
        try:
            manifest = GenericJob.open(path).manifest
            labels = "/".join(item["label"] for item in manifest["config"]["classes"])
            label = labels if len(labels) <= 45 else labels[:42] + "…"
            origin = "人工標註 · " if manifest.get("purpose") == "manual_training" else ""
            return manifest["created"], f"{origin}{label} · {len(manifest['frames'])} 張 · {path.name[:8]}", path
        except (OSError, ValueError, TypeError, KeyError):
            return "", f"{path.name[:8]} · 作業資料無法讀取", path

    def _set_job_choice(self):
        current = self.job.directory if self.job else None
        for picker in (self.jobs_picker, self.training_jobs):
            picker.set(self.job_choices.get(current, "選擇作業"))

    def _update_state(self):
        available = self.project is not None and self.app.task is None and not self.dirty and not self.class_editor_open
        for button in (self.import_still, self.import_video, self.reference_button):
            button.state(["!disabled" if available else "disabled"])
        self.shadow_sam_option.state(["!disabled" if available else "disabled"])
        state = self.job.state if self.job else {}
        for button, enabled in ((self.select_all_button, available and bool(self.input_ids)),
                                (self.start_button, available and bool(self.input_ids) and bool(self._enabled_labels())),
                                (self.add_class_button, available),
                                (self.edit_class_button, available and bool(self.class_table.selection())),
                                (self.manual_train_button, available and bool(self.input_ids)),
                                (self.train_button, available and state.get("status") == "review_ready"),
                                (self.audit_button, available and state.get("status") == "review_ready"),
                                (self.resume_button, available and state.get("status") == "prepared")):
            button.state(["!disabled" if enabled else "disabled"])
        for picker in (self.jobs_picker, self.training_jobs):
            picker.configure(state="readonly" if available and self.job_locations else "disabled")
        self.inputs.configure(state="normal" if available else "disabled")
        self.filter_picker.configure(state="disabled" if self.dirty or self.class_editor_open else "readonly")
        position = self.queue.index(self.selected_frame) if self.selected_frame in self.frame_ids else -1
        self.queue_position.configure(text=f"{position + 1}／{len(self.frame_ids)}")
        can_switch = not self.dirty and not self.class_editor_open
        self.prev_frame.state(["!disabled" if can_switch and position > 0 else "disabled"])
        self.next_frame.state(["!disabled" if can_switch and 0 <= position < len(self.frame_ids) - 1 else "disabled"])
        row = getattr(self, "rows", {}).get(self.selected_frame, {})
        self.preview_button.state(["!disabled" if row or self.preview_open else "disabled"])
        idle_frame = self.preview_open and self.canvas.picture is not None and self.app.task is None and not self.class_editor_open
        can_review = idle_frame and state.get("status") == "review_ready"
        editable = can_review and (row.get("status") == "pending" or self.editing_frame is not None)
        self.canvas.editable = editable
        self.label_picker.configure(state="readonly" if editable else "disabled")
        has_candidates = bool(getattr(self, "candidates", ()))
        self.candidate_picker.configure(state="readonly" if has_candidates else "disabled")
        has_decision = (row.get("status") != "independent_test" and
                        (has_candidates or self.canvas.boxes or self.decision_text.cget("text")
                         or self.confidence_text.cget("text")))
        details_title = f"裁決明細（{len(self.candidates)}）" if has_candidates else "裁決明細"
        self.preview_tabs.tab(self.details_tab, text=details_title, state="normal" if has_decision else "disabled")
        correctable = row.get("status") in {"ai_ready", "approved", "excluded"}
        for button, enabled in ((self.correct_button, can_review and correctable and self.editing_frame is None),
                                (self.adopt_button, editable and self.candidate_picker.current() >= 0),
                                (self.delete_button, editable and self.canvas.selected is not None),
                                (self.discard_button, editable and self.dirty), (self.exclude_button, editable),
                                (self.confirm_button, editable)):
            button.state(["!disabled" if enabled else "disabled"])

    def _reference(self):
        selected = filedialog.askopenfilename(parent=self.app.root, title="選擇獨立驗收清冊", filetypes=[("JSON", "*.json")])
        if selected:
            self.reference = Path(selected)
            self.reference_text.configure(text=self.reference.name)

    def _start(self):
        if self.project is None or self.app.task is not None or self.dirty or self.class_editor_open:
            return
        try:
            enabled = self._enabled_labels()
            classes = [spec for label, spec in self.class_specs.items() if label in enabled]
            ids = [self.input_ids[index] for index in self.inputs.curselection()] or None
            self.job = create_job(self.project, classes, asset_ids=ids, holdout=self.reference,
                                  settings={"shadow_sam": self.shadow_sam.get()})
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set(str(error))
            return
        self._run_inference()

    def _run_inference(self):
        job = self.job
        if job is None or self.app.task is not None or self.dirty or self.class_editor_open:
            return
        self.status.set("在斷網沙箱執行 MM 找框、Qwen 裁決及局部重查。完成後可檢視例外。")
        started = time_ns()
        self.app._start_task("generic_ai", lambda: infer_job(job), self._after_inference, "離線 AI 標註失敗",
                             on_error=self._task_error)
        self._watch_progress(job, "infer", self.app.task, started)
        self._update_state()

    def _after_inference(self, job):
        self.job = job
        self.selected_frame = None
        self.evaluation_rows = []
        self.only_pending.set(False)
        self.review_filter.set("全部影像")
        self._select_job_inputs()
        self._refresh()
        self._maybe_train()

    def _task_error(self, error):
        self.status.set(f"這次作業未完成。{error}")
        self._update_state()

    def _watch_progress(self, job, stage, task, started):
        if task is None or self.app.task is not task or self.job is not job:
            return
        if progress := job_progress(job, stage, started):
            self.status.set(progress)
        self.after(1000, self._watch_progress, job, stage, task, started)

    def _open_job(self, event=None):
        if self.dirty or self.class_editor_open or self.app.task is not None:
            self._set_job_choice()
            return
        picker = self.training_jobs if event is not None and event.widget is self.training_jobs else self.jobs_picker
        index = picker.current()
        if index < 0:
            return
        try:
            self.job = GenericJob.open(self.job_locations[index])
            self._select_job_inputs()
            self._use_job_classes(self.job)
            self.auto_train.set(False)
            self.selected_frame = None
            self.evaluation_rows = []
            self.only_pending.set(False)
            self.review_filter.set("全部影像")
            self._refresh()
            if self.job.state.get("status") == "prepared" and self.mode == "ai":
                self.status.set("已選擇未完成的作業。按「續跑已選作業」開始。")
        except (OSError, ValueError, RuntimeError) as error:
            self._set_job_choice()
            self.status.set(str(error))

    def _choose_filter(self, _event=None):
        self.only_pending.set(self.review_filter.get() == "待審")
        self._filter_frames()

    def _filter_frames(self, *, preferred=None):
        if self.dirty:
            return
        assets = [self.input_assets[ident] for ident in self._selected_input_ids()]
        job_labels = [item["label"] for item in self.job.manifest["config"]["classes"]] if self.job else ()
        self.scope_rows = build_input_rows(assets, self.job_rows, job_labels)
        self._show_class_totals(self.scope_rows)
        values = tuple(REVIEW_FILTERS)
        if any(row.get("shadow_qc") and row["status"] in {"ai_ready", "pending"} for row in self.scope_rows):
            values += (SHADOW_FILTER,)
        if self.evaluation_rows:
            values += (INDEPENDENT_FILTER,)
        self.filter_picker.configure(values=values)
        choice = "待審" if self.only_pending.get() else self.review_filter.get()
        if choice == INDEPENDENT_FILTER and self.evaluation_rows:
            self.queue_summary.configure(text=f"獨立驗收 {len(self.evaluation_rows)} 張")
            self._populate_queue(self.evaluation_rows, preferred)
            return
        if choice == SHADOW_FILTER and choice in values:
            rows = [row for row in self.scope_rows if row["status"] in {"ai_ready", "pending"} and row.get("shadow_qc")
                    and row["shadow_qc"]["disposition"] in {"review", "held"}]
            self.queue_summary.configure(text=f"影子待審 {len(rows)} 張")
            self._populate_queue(sorted(rows, key=review_priority), preferred)
            return
        if choice not in REVIEW_FILTERS:
            choice = "全部影像"
        self.review_filter.set(choice)
        rows = self.scope_rows
        state = REVIEW_FILTERS[choice]
        visible = [row for row in rows if state is None or row["status"] == state]
        pending = sum(row["status"] == "pending" for row in rows)
        self.queue_summary.configure(text=f"顯示 {len(visible)}／{len(rows)} 張 · 待審 {pending} 張")
        self._populate_queue(sorted(visible, key=review_priority), preferred)

    def _populate_queue(self, rows, preferred):
        target = preferred if preferred is not None else self.selected_frame
        self.rows = {row["id"]: row for row in rows}
        self.frame_ids = tuple(self.rows)
        if children := self.queue.get_children():
            self.queue.delete(*children)
        for row in rows:
            state = row["status"]
            confidence = {} if state == "independent_test" else row.get("confidence") or {}
            name = row.get("name") or Path(row["image"]).name
            reasons = self._reason_text(row.get("exceptions", ())) if state == "pending" else ""
            if self.review_filter.get() == SHADOW_FILTER:
                reasons = self._reason_text(row["shadow_qc"]["flags"])
            self.queue.insert("", "end", iid=row["id"], tags=(state,), values=(
                name, FRAME_STATES.get(state, state), percentage(confidence.get("score")),
                percentage(confidence.get("mm_score")), percentage(confidence.get("qwen_score")),
                row.get("manual_box_count", "—"), row.get("ai_box_count", "—"), reasons,
            ))
        self._pick_frame(target if target in self.rows else next(iter(self.frame_ids), None))

    def _pick_frame(self, target):
        if target:
            self.queue.selection_set(target)
            self.queue.focus(target)
            self.queue.see(target)
        self._load_frame(target)

    def _move_frame(self, offset):
        if self.dirty or self.class_editor_open or self.selected_frame not in self.rows:
            return
        index = self.queue.index(self.selected_frame) + offset
        if 0 <= index < len(self.frame_ids):
            self._pick_frame(self.frame_ids[index])

    def _select_frame(self, _event=None):
        chosen = self.queue.selection()
        if not chosen:
            return
        target = chosen[0]
        if target == self.selected_frame:
            return
        if self.dirty or self.class_editor_open:
            self.status.set("請先完成或取消目前的編輯。")
            if self.selected_frame:
                self.queue.selection_set(self.selected_frame)
            return
        self._load_frame(target)

    def _toggle_preview(self):
        self._set_preview(not self.preview_open)

    def _set_preview(self, shown, *, load=True):
        self.preview_open = shown
        self.preview_button.configure(text="返回指標" if shown else "開啟圖像檢查")
        self._fit_workspace_chrome()
        if shown:
            self.preview.pack(fill="both", expand=True, pady=(2, 0))
            if load:
                self._load_preview_picture(self.rows.get(self.selected_frame))
        else:
            self.preview.pack_forget()
            self.canvas.picture = self.canvas.photo = None
        self._fit_preview_layout(self.metrics_pane.winfo_width() - 14)
        if load:
            self._update_state()

    def _load_preview_picture(self, row):
        if not self.preview_open or row is None:
            return
        try:
            if row.get("status") == "independent_test":
                source = (row["dataset"] / row["image"]).resolve(strict=True)
                if not source.is_relative_to(row["dataset"] / "images") or digest(source) != row["image_sha256"]:
                    raise ValueError("畫格的路徑或雜湊不符。")
            else:
                source = self.project.image_path(row["id"])
            with Image.open(source) as image:
                self.canvas.picture = image.convert("RGB")
        except (OSError, KeyError, ValueError) as error:
            self.canvas.picture = None
            self.canvas.render()
            self._set_preview_notice(f"無法讀取影像。{error}")
            return
        self.canvas.render()
        if row.get("status") == "independent_test":
            self._set_preview_notice(f"YOLO {len(self.canvas.boxes)} · 人工參考 {len(self.canvas.overlay)}（橘色虛線）")
        else:
            self._update_state()
            self._canvas_changed(False)

    def _load_frame(self, frame_id):
        self.selected_frame = frame_id
        self.dirty = False
        self.editing_frame = None
        self.candidates = []
        self.preview_tabs.select(self.image_tab)
        self.candidate_picker.configure(values=[])
        self.candidate_picker.set("")
        self.candidate_notes = ""
        self.preview_score.configure(text="")
        self.confidence_text.configure(text="")
        self.decision_text.configure(text="")
        self._set_preview_notice("")
        row = self.rows.get(frame_id)
        if row and row.get("status") == "independent_test":
            self._load_evaluation_frame(row)
            return
        if row is None:
            self.case_text.configure(text="目前無可顯示的影像")
            self.canvas.show(None, [])
            self._update_state()
            return
        labels = [item["label"] for item in self.job.manifest["config"]["classes"]] if frame_id in self.job_rows else list(self.class_specs)
        self.label_picker.configure(values=labels)
        if self.class_name.get() not in labels:
            self.class_name.set(labels[0] if labels else "")
        self.canvas.label = self.class_name.get()
        state = row["status"]
        self.case_text.configure(text=f"{row['name']} · {FRAME_STATES.get(state, state)}")
        self._show_frame_decision(row)
        boxes = [{**box, "xyxy": list(box["xyxy"])} for box in row.get("boxes", ())]
        self.candidates = [candidate for candidate in row.get("candidates", ()) if candidate.get("xyxy")]
        self.candidate_picker.configure(values=[self._candidate_text(candidate, index)
                                                for index, candidate in enumerate(self.candidates, 1)])
        if self.candidates:
            self.candidate_picker.set("選一個候選查看框位和裁決理由")
        else:
            self.candidate_picker.set("這張影像沒有候選可檢視")
        self.canvas.show(None, boxes)
        self._canvas_changed(False)
        self._load_preview_picture(row)

    @staticmethod
    def _reason_text(exceptions):
        codes = (REASONS.get(item["code"], item.get("reason") or item.get("message") or item["code"])
                 for item in exceptions)
        return "、".join(dict.fromkeys(codes))

    def _show_frame_decision(self, row):
        confidence = row.get("confidence") or {}
        score = percentage(confidence.get("score"))
        status = confidence.get("status")
        suffix = {"review": " · 待審，分數僅涵蓋已採用框",
                  "no_accepted": " · 無採用框，整張是否為空仍待確認",
                  "unavailable": " · 尚無足夠模型證據"}.get(status, "")
        self.confidence_text.configure(text=(
            f"原始 AI 分數 {score} · MM {percentage(confidence.get('mm_score'))} · "
            f"Qwen 採用支持 {percentage(confidence.get('qwen_score'))}{suffix}") if confidence else "")
        self.preview_score.configure(text=(
            f"AI 原分 {score} · MM {percentage(confidence.get('mm_score'))} · Qwen {percentage(confidence.get('qwen_score'))}"
            if confidence else ""))
        candidates = row.get("candidates", ())
        counts = Counter(candidate.get("verdict") for candidate in candidates)
        details = []
        if "candidates" in row:
            details.append(f"候選 {len(candidates)} · 採用 {counts['accepted']} · 拒絕 {counts['rejected']} · "
                           f"未定 {len(candidates) - counts['accepted'] - counts['rejected']} · "
                           f"局部重查 {len(row.get('retries', ()))} 區")
        if any(item.get("code") == "manual_review_pending" for item in row.get("exceptions", ())):
            details.append(REASONS["manual_review_pending"])
        if codes := self._reason_text(row.get("ai_exceptions", row.get("exceptions", ()))):
            details.append("原始 AI 送審原因  " + codes)
        if qc := row.get("shadow_qc"):
            coverage = qc.get("evidence_coverage", {})
            details.append(f"影子 QC  {SHADOW_DISPOSITION[qc['disposition']]} · 重查 {qc['retry_count']} 次"
                           f" · SAM 框界 {coverage.get('sam_boxes_checked', 0)} 框")
            if reasons := self._reason_text(qc["flags"]):
                details.append(reasons)
        self.decision_text.configure(text="\n".join(details))

    def _canvas_changed(self, dirty):
        self.dirty |= dirty
        selected = self.canvas.selected
        if selected is not None:
            self.class_name.set(self.canvas.boxes[selected]["label"])
            self.canvas.label = self.class_name.get()
        self._set_preview_notice("尚未提交" if self.dirty else "")
        if self.preview_tabs.select() == str(self.details_tab):
            self._render_decision_details()
        self._update_state()

    def _set_preview_notice(self, text):
        self.preview_notice.configure(text=text)
        if text:
            self.preview_notice.pack(side="bottom", fill="x", after=self.preview_actions, pady=(3, 0))
        else:
            self.preview_notice.pack_forget()

    def _start_correction(self):
        if self.correct_button.instate(["disabled"]):
            return
        self.editing_frame = self.selected_frame
        self.canvas.editable = True
        self._canvas_changed(True)

    def _delete_box(self):
        if self.canvas.editable and self.canvas.selected is not None:
            self.canvas.boxes.pop(self.canvas.selected)
            self.canvas.selected = None
            self._canvas_changed(True)
            self.canvas.render()

    def _change_label(self, _event):
        self.canvas.label = self.class_name.get()
        if self.canvas.editable and self.canvas.selected is not None:
            self.canvas.boxes[self.canvas.selected]["label"] = self.canvas.label
            self._canvas_changed(True)
            self.canvas.render()

    @staticmethod
    def _candidate_text(candidate, index):
        verdict = candidate.get("decision", candidate.get("verdict", ""))
        if isinstance(verdict, dict):
            verdict = verdict.get("status") or verdict.get("reason") or "待確認"
        reason = REASONS.get(candidate.get("reason_code"), candidate.get("reason_code") or "")
        outcome = VERDICTS.get(verdict, verdict)
        return f"候選 {index} · {candidate.get('label', '')} · {outcome} {reason}".rstrip()

    def _candidate_selected(self, _event=None):
        index = self.candidate_picker.current()
        if not 0 <= index < len(self.candidates) or self.preview_tabs.select() != str(self.details_tab):
            return
        candidate = self.candidates[index]
        scores = candidate_confidence(candidate)
        votes = [
            f"{order}順序 {VOTE_OPTIONS.get(vote.get('winner'), '未定')}（分差 {percentage(vote.get('margin'))}、"
            f"四個選項機率合計 {percentage(vote.get('allowed_token_mass'))}）"
            for order, vote in zip(("原", "反"), candidate.get("votes", ())) if isinstance(vote, dict)
        ]
        evidence = " · ".join(votes) if votes else "沒有雙順序覆核證據"
        self.candidate_notes = (f"{self._candidate_text(candidate, index + 1)}\n"
                                f"MM {percentage(scores['mm_score'])} · Qwen 採用支持 "
                                f"{percentage(scores['qwen_score'])}\n{evidence}")
        displayed = [{"label": candidate["label"], "xyxy": candidate["xyxy"]}]
        qc = self.rows.get(self.selected_frame, {}).get("shadow_qc") or {}
        checked = next((item for item in qc.get("checks", ()) if item["candidate_id"] == candidate["id"]), None)
        if checked:
            if checked.get("sam_status") == "measured":
                self.candidate_notes += f"\nSAM 外接框 IoU {percentage(checked['sam_box_iou'])}"
                displayed.append({"label": "SAM", "xyxy": checked["sam_box"]})
            elif checked.get("sam_status") == "failed":
                self.candidate_notes += "\nSAM 未取得有效遮罩"
            flags = [item for item in qc.get("flags", ()) if item.get("candidate_id") == candidate["id"]]
            if flags:
                self.candidate_notes += "\n" + self._reason_text(flags)
        self._render_decision_details()
        self.canvas.overlay = displayed
        self.canvas.render()
        self._update_state()

    def _preview_tab_changed(self, _event=None):
        details = self.preview_tabs.select() == str(self.details_tab)
        self.preview_body.pack_forget()
        page = self.details_tab if details else self.image_tab
        self.preview_body.pack(in_=page, fill="both", expand=True)
        if details:
            self.candidate_panel.pack(side="right", fill="y", padx=(8, 0), before=self.canvas)
            if self.candidates:
                self.candidate_controls.pack(fill="x", pady=(0, 6), before=self.detail_body)
            else:
                self.candidate_controls.pack_forget()
            if self.candidate_picker.current() >= 0:
                self._candidate_selected()
            else:
                self._render_decision_details()
        else:
            self.candidate_panel.pack_forget()
            if getattr(self, "rows", {}).get(self.selected_frame, {}).get("status") != "independent_test":
                self.canvas.overlay = ()
            self.canvas.render()

    def _render_decision_details(self):
        sections = []
        if self.candidate_notes:
            sections.append(("所選候選框", self.candidate_notes))
        manual = sum(box.get("source") == "manual" for box in self.canvas.boxes)
        status = "目前標註（草稿）" if self.dirty else "目前標註"
        sections.append((status, f"人工 {manual} · AI {len(self.canvas.boxes) - manual}"))
        frame = "\n".join(text for label in (self.confidence_text, self.decision_text)
                          if (text := label.cget("text")))
        if frame:
            sections.append(("影像裁決紀錄", frame))
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        for index, (heading, text) in enumerate(sections):
            if index:
                self.detail_text.insert("end", "\n\n")
            self.detail_text.insert("end", heading + "\n", "heading")
            self.detail_text.insert("end", text)
        self.detail_text.configure(state="disabled")
        self.detail_text.yview_moveto(0)

    def _adopt_candidate(self):
        index = self.candidate_picker.current()
        if self.canvas.editable and index >= 0:
            candidate = self.candidates[index]
            box = {"label": candidate["label"], "xyxy": list(candidate["xyxy"])}
            match = next((position for position, existing in enumerate(self.canvas.boxes)
                          if existing["label"] == box["label"] and existing["xyxy"] == box["xyxy"]), None)
            if match is None:
                box["source"] = "manual"
                match = len(self.canvas.boxes)
                self.canvas.boxes.append(box)
                self.dirty = True
            self.canvas.selected = match
            self._canvas_changed(False)
            self.canvas.render()

    def _submit(self, *, exclude=False):
        if not self.canvas.editable or not self.selected_frame or self.app.task is not None:
            return
        index = self.queue.index(self.selected_frame)
        ordered = self.frame_ids[index + 1:] + self.frame_ids[:index]
        target = (next(iter(ordered), self.selected_frame) if self.review_filter.get() == SHADOW_FILTER else
                  next((ident for ident in ordered if self.rows[ident]["status"] == "pending"), self.selected_frame))
        try:
            revision = self.rows[self.selected_frame].get("project_annotation_revision")
            self.job.review(self.selected_frame, None if exclude else self.canvas.boxes, exclude=exclude,
                            expected_project_revision=revision)
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set(str(error))
            return
        self.dirty = False
        self.editing_frame = None
        evidence_note = ""
        if not exclude and self.batch_audit.actor.get().strip():
            try:
                event = record_correction(self.job, self.selected_frame, actor=self.batch_audit.actor.get())
                if event is not None:
                    evidence_note = " 已記錄為下一批的開發證據。"
            except (OSError, ValueError, KeyError) as error:
                evidence_note = f" 開發證據未寫入。{error}"
        try:
            self.app.refresh_annotation_project()
        except (OSError, ValueError) as error:
            self._refresh(preferred_frame=target)
            self.status.set(f"已儲存複核，但單張工作區讀取失敗。{error}")
            return
        self._refresh(preferred_frame=target)
        self.status.set("已扣留這張畫格。" if exclude else f"已儲存整張複核結果。{evidence_note}")
        self._maybe_train()

    def _maybe_train(self):
        if self.auto_train.get() and self.job:
            summary = self.job.summary()
            if summary["needs_review"] == 0 and all(summary["boxes_by_class"].values()):
                self._train()

    def _manual_training(self):
        if self.project is None or self.app.task is not None or self.dirty or self.class_editor_open:
            return
        from generic_job import create_manual_training_job

        try:
            self.job = create_manual_training_job(self.project, self.project.labels, holdout=self.reference)
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set(str(error))
            return
        self._refresh()
        labels = "、".join(item["label"] for item in self.job.manifest["config"]["classes"])
        self.status.set(f"人工標註作業已建立 · 類別 {labels}。選擇輪數後即可訓練。")

    def _train(self):
        if self.app.task is not None or self.dirty or self.class_editor_open or self.job is None:
            return
        job = self.job
        try:
            epochs = int(self.epochs.get())
            if not 1 <= epochs <= 400:
                raise ValueError
        except ValueError:
            self.status.set("訓練輪數請輸入 1 到 400 的整數。")
            return
        self.status.set("正在建立訓練資料，並在斷網沙箱用隨機初始權重訓練 YOLO。")
        started = time_ns()
        self.app._start_task("generic_ai", lambda: train_job(job, epochs), self._after_training, "離線訓練失敗",
                             on_error=self._task_error)
        self._watch_progress(job, "train", self.app.task, started)
        self._update_state()

    def _after_training(self, job):
        self.job = job
        if not self.app.show_generic_training():
            self._refresh()

    def _show_evaluation(self):
        if self.dirty or self.class_editor_open or self.app.task is not None:
            self.status.set("請先完成目前的修正或作業。")
            return
        try:
            report, training, dataset = latest_training(self.job)
            evaluation = report["independent_evaluation"]
            source = (training / evaluation["frames_report"]).resolve(strict=True)
            if not source.is_relative_to(training) or digest(source) != evaluation["frames_report_sha256"]:
                raise ValueError("逐張驗收檔案的雜湊或路徑不符。")
            files = {row["image"]: row for row in json.loads((dataset / "dataset.json").read_bytes())["frames"]
                     if row["split"] == "test"}
            results = json.loads(source.read_bytes())["frames"]
            if {row["image"] for row in results} != files.keys() or len(results) != len(files):
                raise ValueError("驗收影像和資料名單不一致。")
            self.evaluation_rows = [{**row, "status": "independent_test", "dataset": dataset,
                                     "image_sha256": files[row["image"]]["image_sha256"],
                                     "confidence": evaluation["confidence"]} for row in results]
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.set(f"無法讀取獨立驗收結果。{error}")
            return
        if not self.app.show_generic_ai():
            return
        self.only_pending.set(False)
        self.review_filter.set(INDEPENDENT_FILTER)
        self._set_preview(True, load=False)
        self._filter_frames(preferred=self.evaluation_rows[0]["id"] if self.evaluation_rows else None)
        self.status.set("用影像清單的篩選選單可返回 AI 標註結果。")

    def _load_evaluation_frame(self, row):
        predictions = [box for box in row["predictions"] if box["score"] >= row["confidence"]]
        self.case_text.configure(text=f"獨立驗收 · {row['shot']}")
        self.confidence_text.configure(text=f"YOLO 預測顯示門檻 {percentage(row['confidence'])}")
        self.preview_score.configure(text=f"YOLO 門檻 {percentage(row['confidence'])}")
        self.canvas.show(None, predictions, overlay=row["reference"])
        self._load_preview_picture(row)
        self._update_state()

    def _render_results(self):
        report = training = dataset = error = None
        if self.job:
            try:
                report, training, dataset = latest_training(self.job)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        has_evaluation = report and report.get("independent_evaluation")
        self.eval_button.state(["!disabled" if has_evaluation else "disabled"])
        self.result_view.render(self.job, report, training, dataset, error=error)
