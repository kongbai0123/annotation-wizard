"""Structured batch decisions and separately measured YOLO results."""

from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
import tkinter as tk
from tkinter import ttk


BG, PANEL, TEXT, MUTED = "#101820", "#18242c", "#ebf2f3", "#a8bbc3"
EDGE, STRIPE = "#344651", "#20303a"
FONT = "Microsoft JhengHei UI"
REVIEW_REASONS = {
    "bad_extent": "候選框範圍不符",
    "conflicting_classes": "重疊候選的類別衝突",
    "competing_extents": "同類候選的框界難以取捨",
    "empty_unverified": "空畫格需確認",
    "invalid_proposal": "無效候選分數或座標",
    "invalid_scene_scan": "整圖掃描結果無法讀取",
    "no_accepted_boxes": "沒有通過覆核的框",
    "uncertain_review": "Qwen 兩次判讀證據不足",
    "uncertain_scene_scan": "整圖目標可能不完整",
    "unmatched_scene_target": "整圖掃描有疑似漏框",
}
TRAINING_STATES = {
    "completed": "訓練完成",
    "running": "訓練中",
    "failed": "訓練未完成",
    "blocked_no_accepted_positive": "有類別缺少可入訓正例",
}
INITIALIZATIONS = {"random": "從零隨機初始化"}
LOSS_NAMES = {"train/box_loss": "框位置損失", "train/cls_loss": "類別損失", "train/l1_loss": "L1 損失",
              "train/dfl_loss": "DFL 損失"}


def percent(value):
    return f"{value:.1%}" if type(value) in (int, float) and isfinite(value) else "—"


def number(value, digits=3):
    return f"{value:.{digits}f}" if type(value) in (int, float) and isfinite(value) else "—"


def yes_no(value):
    return {True: "有", False: "無"}.get(value, "—")


def display_label(label):
    return "背景" if label == "background" else label


def decision_details(inference, reviewed_ids):
    """Count actual entries and distinct affected images from sealed batch output."""
    if not inference:
        return None
    frames = inference.get("frames")
    if frames is None:
        return None
    reasons, affected, remaining, descriptions = Counter(), defaultdict(set), defaultdict(set), {}
    candidates, proposals, retries = Counter(), Counter(), 0
    initial_review = coverage = 0
    for row in frames:
        frame_id = row["id"]
        boxes = row.get("boxes", ())
        entries = row.get("candidates")
        exceptions = row.get("exceptions", ())
        if not boxes and not exceptions:
            exceptions = [{"code": "empty_unverified"}]
        initial_review += bool(exceptions)
        if entries is not None:
            coverage += bool(entries)
            candidates.update(candidate.get("verdict", "unknown") for candidate in entries)
        proposals.update(box["label"] for box in boxes)
        retries += len(row.get("retries", ()))
        for exception in exceptions:
            code = exception["code"]
            descriptions.setdefault(code, exception.get("reason") or code)
            reasons[code] += 1
            affected[code].add(frame_id)
            if frame_id not in reviewed_ids:
                remaining[code].add(frame_id)
    return {
        "initial_review": initial_review,
        "candidate_frames": coverage if all(row.get("candidates") is not None for row in frames) else None,
        "candidates": candidates if all(row.get("candidates") is not None for row in frames) else None,
        "original_boxes": proposals,
        "retries": retries if all("retries" in row for row in frames) else None,
        "reasons": [
            (REVIEW_REASONS.get(code, descriptions[code]), len(affected[code]), count, len(remaining[code]))
            for code, count in sorted(reasons.items(), key=lambda item: (-len(affected[item[0]]), -item[1], item[0]))
        ],
    }


def matching_rows(outcome):
    """Present reported values without turning undefined precision into zero."""
    by_class = outcome.get("by_class", {})
    return [(display_label(label), values.get("tp"), values.get("fp"), values.get("fn"),
             percent(values.get("precision")), percent(values.get("recall")))
            for label, values in by_class.items()]


class GenericResultsView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.sections = {}
        self.nav = tk.Frame(self, bg=BG)
        self.nav.pack(fill="x", pady=(0, 6))
        style = ttk.Style(self)
        style.configure("GenericResults.Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, rowheight=25, font=(FONT, 10))
        style.configure("GenericResults.Treeview.Heading", background=EDGE, foreground=TEXT,
                        font=(FONT, 10, "bold"), padding=(4, 4))
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content = tk.Frame(self.canvas, bg=BG)
        self.content_id = self.canvas.create_window(0, 0, anchor="nw", window=self.content)
        self.content.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.content_id, width=event.width))
        self.canvas.bind("<MouseWheel>", self._scroll)

    def _scroll(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _label(self, parent, text, *, muted=False, size=10, bold=False, bg=BG, wrap=0):
        item = tk.Label(parent, text=text, bg=bg, fg=MUTED if muted else TEXT, anchor="w",
                        justify="left", font=(FONT, size, "bold" if bold else "normal"), wraplength=wrap)
        if wrap:
            parent.bind("<Configure>", lambda event: item.configure(wraplength=max(160, event.width - 16)), add="+")
        item.bind("<MouseWheel>", self._scroll)
        return item

    def _note(self, parent, text):
        self._label(parent, text, muted=True, wrap=1050).pack(fill="x", pady=(2, 5))

    def _section(self, name, subtitle=None):
        section = tk.Frame(self.content, bg=BG)
        section.pack(fill="x", padx=(0, 8), pady=(4, 16))
        section.bind("<MouseWheel>", self._scroll)
        tk.Frame(section, bg=EDGE, height=1).pack(fill="x", pady=(0, 7))
        self._label(section, name, size=14, bold=True).pack(fill="x")
        if subtitle:
            self._note(section, subtitle)
        self.sections[name] = section
        ttk.Button(self.nav, text=name, command=lambda: self._jump(name)).pack(side="left", padx=(0, 5))
        return section

    def _jump(self, name):
        self.update_idletasks()
        self.canvas.yview_moveto(self.sections[name].winfo_y() / max(1, self.content.winfo_height()))

    def _cards(self, parent, values):
        cards = tk.Frame(parent, bg=BG)
        cards.pack(fill="x", pady=(3, 7))
        cards.bind("<MouseWheel>", self._scroll)
        for index, (label, value) in enumerate(values):
            cards.columnconfigure(index, weight=1, uniform="cards")
            card = tk.Frame(cards, bg=PANEL, padx=10, pady=7)
            card.grid(row=0, column=index, padx=(0, 4), sticky="nsew")
            card.bind("<MouseWheel>", self._scroll)
            self._label(card, label, bg=PANEL, muted=True).pack(fill="x")
            self._label(card, str("—" if value is None else value), bg=PANEL, size=15, bold=True).pack(fill="x")

    def _table(self, parent, headings, rows, *, widths=None, anchor_first="w", max_rows=9):
        if not rows:
            return
        host = tk.Frame(parent, bg=BG)
        host.pack(fill="x", pady=(2, 8))
        columns = tuple(f"column{index}" for index in range(len(headings)))
        table = ttk.Treeview(host, columns=columns, show="headings", selectmode="none",
                             style="GenericResults.Treeview", height=min(max_rows, len(rows)))
        table.tag_configure("stripe", background=STRIPE)
        table.grid(row=0, column=0, sticky="nsew")
        host.columnconfigure(0, weight=1)
        for index, (column, title) in enumerate(zip(columns, headings, strict=True)):
            width = widths[index] if widths else 100
            table.heading(column, text=title)
            table.column(column, width=width, minwidth=min(72, width), anchor=anchor_first if index == 0 else "center",
                         stretch=index == 0 or all(len(str(title)) < 16 for title in headings))
        if len(rows) > max_rows:
            vertical = ttk.Scrollbar(host, command=table.yview)
            table.configure(yscrollcommand=vertical.set)
            vertical.grid(row=0, column=1, sticky="ns")
        else:
            table.bind("<MouseWheel>", self._scroll)
        horizontal = ttk.Scrollbar(host, orient="horizontal", command=table.xview)
        table.configure(xscrollcommand=horizontal.set)
        horizontal.grid(row=1, column=0, sticky="ew")
        for index, values in enumerate(rows):
            table.insert("", "end", values=tuple("—" if value is None else value for value in values),
                         tags=("stripe",) if index % 2 else ())

    def _facts(self, parent, pairs):
        for title, value in pairs:
            self._label(parent, f"• {title}　{'—' if value is None else value}", wrap=1050).pack(fill="x", pady=(1, 1))

    def render(self, job, report=None, training=None, dataset=None, *, error=None):
        for parent in (self.nav, self.content):
            for child in parent.winfo_children():
                child.destroy()
        self.sections.clear()
        self.canvas.yview_moveto(0)
        if job is None:
            self._section("作業結果", "開始一批標註後，這裡會顯示採用、待審與訓練成果。")
            return
        report = report or {}
        self._decisions(job, report)
        self._training(report, error)
        self._evaluation(report.get("independent_evaluation"))
        self._provenance(job, report, training, dataset)

    def _decisions(self, job, report):
        section = self._section("標註與入訓統計", "以影像的目前狀態統計。單張與批量的人工確認共用標註。")
        summary = job.summary()
        self._cards(section, [(title, summary.get(key, 0)) for title, key in (
            ("作業影像", "frames"), ("AI 放行", "ai_ready"), ("人工確認", "approved"),
            ("待人工", "needs_review"), ("扣留", "excluded"))])
        self._facts(section, [("獨立參考圖數", summary.get("reference_frames", 0)),
                              ("移除重複輸入", summary.get("duplicate_inputs", 0))])
        inference = job.inference
        method = inference.get("method", {}) if inference else {}
        details = decision_details(inference, job.review_records() if inference else {})
        if details:
            duration = inference.get("timing", {}).get("inference_seconds")
            candidate_frames, candidates = details["candidate_frames"], details["candidates"]
            facts = [("找框與裁決", "MM Grounding DINO 提出候選，Qwen 用兩種順序覆核。"),
                     ("原始推論送審", f"{details['initial_review']} 張"),
                     ("批次推論耗時", f"{number(duration, 1)} 秒")]
            if qwen_model := method.get("qwen", {}).get("model"):
                facts.insert(1, ("裁決模型", qwen_model))
            if candidate_frames is not None and candidates is not None:
                score = candidate_frames / summary["frames"] if summary["frames"] else None
                facts.extend([
                    ("有候選影像", f"{candidate_frames}／{summary['frames']} 張 · {percent(score)}"),
                    ("候選裁決", f"{sum(candidates.values())} 個 · 採用 {candidates['accepted']} · "
                               f"拒絕 {candidates['rejected']} · 未決 {candidates['unresolved'] + candidates['pending']}"),
                ])
            if details["retries"] is not None:
                facts.append(("局部重查區域數", details["retries"]))
            self._facts(section, facts)
            self._note(section, "有候選影像比例以全部輸入影像為分母。候選、採用與送審是作業統計，準確率請看獨立驗收。")
            self._note(section, "逐圖整張信心取所有已採用框中最低的 MM 分數或 Qwen 採用支持度。若仍有疑似漏框，請依畫格狀態複核。")
            if details["reasons"]:
                self._table(section, ("送審原因", "原始圖數", "原始次數", "仍待審圖數"), details["reasons"],
                            widths=(340, 110, 110, 130))
                self._note(section, "同一張圖可能有多個原因，同一原因也可能在一張圖出現多次。")
        else:
            self._note(section, ("本作業使用已確認整張的人工標註。"
                                 if job.manifest.get("purpose") == "manual_training"
                                 else "目前尚無可統計的原始裁決明細。"))
        approved = summary.get("boxes_by_class", {})
        human = summary.get("human_boxes_by_class", {})
        trained = report.get("training", {}).get("boxes_by_class", {})
        original = details["original_boxes"] if details else None
        classes = method.get("classes", ())
        prompts = {row["label"]: row["prompt"] for row in classes}
        labels = dict.fromkeys([*prompts, *approved, *trained, *(original or {})])
        self._table(section, ("類別", "原始 AI 採用框", "人工確認框", "目前可匯出框", "實際入訓框", "MM 提示詞"),
                    [(label, original[label] if original is not None else None,
                      human.get(label, 0), approved.get(label, 0), trained.get(label), prompts.get(label)) for label in labels],
                    widths=(140, 125, 115, 125, 125, 200))

    def _training(self, report, error):
        section = self._section("訓練紀錄", "入訓影像與框數以封存的 YOLO 資料快照為準。")
        if error:
            self._note(section, f"無法開啟訓練收據。{error}")
        if not report:
            self._note(section, "目前尚無可讀取的訓練收據。")
            return
        train = report.get("training", {})
        completed, requested = report.get("epochs_completed"), report.get("epochs_requested")
        boxes = train.get("boxes_by_class")
        status = report.get("status")
        self._cards(section, [("狀態", TRAINING_STATES.get(status, status)),
                              ("輪數", f"{completed if completed is not None else '—'}／"
                                       f"{requested if requested is not None else '—'}"),
                              ("訓練影像", train.get("images")), ("訓練框", sum(boxes.values()) if boxes is not None else None)])
        self._facts(section, [
            ("模型", report.get("model")), ("運算裝置", report.get("device")),
            ("初始化", INITIALIZATIONS.get(report.get("initialization"), report.get("initialization"))),
            ("使用預訓練權重", yes_no(report.get("pretrained"))), ("繼續既有訓練", yes_no(report.get("resume"))),
            ("尺寸／批次／種子", f"{report.get('imgsz', '—')}／{report.get('batch', '—')}／{report.get('seed', '—')}"),
            ("訓練程序耗時", f"{number(report.get('elapsed_seconds'), 1)} 秒"),
        ])
        if missing := report.get("missing_training_classes"):
            self._note(section, f"缺少可入訓正例的類別　{'、'.join(missing)}")
        epochs = report.get("epoch_results", ())
        loss_keys = dict.fromkeys(key for row in epochs for key in row if key.startswith("train/"))
        if loss_keys:
            self._label(section, "各輪訓練損失", bold=True).pack(fill="x", pady=(5, 2))
            self._table(section, ("輪次", *(LOSS_NAMES.get(key, key) for key in loss_keys)), [
                (number(row.get("epoch"), 0), *(number(row.get(key), 4) for key in loss_keys)) for row in epochs
            ], widths=(85, *(140 for _ in loss_keys)), max_rows=5)
            self._note(section, "訓練損失用於追蹤最佳化過程，模型對獨立資料的表現另列於驗收。")

    def _evaluation(self, evaluation):
        section = self._section("獨立驗收", "在保留的參考影像上量測 YOLO 表現。")
        if not evaluation:
            self._note(section, "目前沒有獨立人工驗收結果。")
            return
        self._cards(section, [("驗收影像", evaluation.get("images")), ("mAP50", percent(evaluation.get("map50"))),
                              ("mAP50–95", percent(evaluation.get("map50_95"))),
                              ("固定信心門檻", number(evaluation.get("confidence"), 2))])
        self._note(section, "AP 使用輸出分數排序。下方命中、誤報、漏檢、精確率及召回率使用固定信心門檻。沒有人工正例的類別不納入 mAP。")
        if by_class := evaluation.get("by_class"):
            self._table(section, ("類別", "人工參考框", "AP50", "AP50–95"),
                        [(label, values.get("truth"), percent(values.get("ap50")), percent(values.get("ap50_95")))
                         for label, values in by_class.items()], widths=(155, 125, 125, 125))
        if "invalid_predictions" in evaluation:
            self._facts(section, [("排除無效預測框", evaluation["invalid_predictions"]),
                                  ("其中達固定信心門檻", evaluation.get("invalid_predictions_at_operating_threshold"))])
        for threshold, outcome in evaluation.get("thresholds", {}).items():
            self._label(section, f"IoU {threshold}", bold=True).pack(fill="x", pady=(5, 2))
            rows = matching_rows(outcome)
            self._table(section, ("類別", "命中 TP", "誤報 FP", "漏檢 FN", "精確率", "召回率"), rows,
                        widths=(175, 90, 90, 90, 110, 110))
            if rows:
                counts = [sum(row.get(key, 0) for row in outcome["by_class"].values()) for key in ("tp", "fp", "fn")]
                self._note(section, f"合計命中 {counts[0]}、誤報 {counts[1]}、漏檢 {counts[2]}。"
                                    f"每張平均誤報 {number(outcome.get('false_positives_per_image'))}。")
                if not (counts[0] or counts[1]):
                    self._note(section, "沒有預測達到固定信心門檻，精確率無定義。")
            labels = outcome.get("labels", ())
            matrix = outcome.get("matrix", ())
            if labels and len(matrix) == len(labels) and all(len(row) == len(labels) for row in matrix):
                self._label(section, "混淆矩陣", bold=True).pack(fill="x", pady=(5, 2))
                self._note(section, "左側列是模型預測，上方欄是人工參考。背景列表示漏檢，背景欄表示誤報。")
                self._table(section, ("預測 ↓ ／人工參考 →", *map(display_label, labels)),
                            [(display_label(label), *row) for label, row in zip(labels, matrix, strict=True)],
                            widths=(170, *(95 for _ in labels)))
        self._note(section, "「—」表示無法計算或未記錄，例如沒有預測時的精確率，或沒有人工正例時的召回率。")

    def _provenance(self, job, report, training, dataset):
        section = self._section("資料與證據", "路徑與雜湊來自已開啟的作業及訓練收據。")
        self._facts(section, [("作業資料夾", job.directory), ("訓練資料快照", dataset),
                              ("資料清單 SHA-256", report.get("dataset_manifest_sha256")),
                              ("分組策略", {"explicit_project_split": "按來源分開訓練與獨立驗收"}
                               .get(report.get("split_policy"), report.get("split_policy"))),
                              ("隔離配置", report.get("isolation_profile")),
                              ("隔離證據 SHA-256", report.get("isolation_sha256"))])
        checkpoint = report.get("fixed_checkpoint_before_test") or report.get("checkpoints", {}).get("last")
        if checkpoint and checkpoint.get("path"):
            path = Path(training) / checkpoint["path"] if training else checkpoint["path"]
            self._facts(section, [("本輪固定權重", path), ("權重 SHA-256", checkpoint.get("sha256"))])
        if audit := report.get("audit"):
            reads = Counter(audit.get("local_checkpoint_reads", ()))
            self._facts(section, [
                ("紀錄的本機權重讀取", "、".join(f"{name} × {count}" for name, count in reads.items()) or "0 筆"),
                ("攔截的外部權重讀取", len(audit["blocked_external_checkpoints"]) if "blocked_external_checkpoints" in audit else None),
                ("紀錄的網路事件", len(audit["network_events"]) if "network_events" in audit else None),
            ])
