"""Whole-frame shadow audit inside the existing annotation workbench."""

from copy import deepcopy
from hashlib import sha256
import tkinter as tk
from tkinter import ttk

from PIL import Image

from batch_acceptance import AUTO_CANDIDATE, HELD, HUMAN_CONFIRMED
from batch_feedback import summarize_feedback
from batch_workflow import REASONS, audit_frame, feedback_events, overview, prepare_audit, seal_reference, submit_audit
from generic_job import jobs_for
from project import Project


STRATUM_NAMES = {AUTO_CANDIDATE: "自動候選", HELD: "扣留", HUMAN_CONFIRMED: "人工確認"}
STATUS_NAMES = {
    "passed": "通過", "exceeds_limit": "錯誤率超標", "insufficient_evidence": "證據不足",
    "pending_reviews": "待完成抽檢", "pending_policy": "待核准放行門檻",
}
FAILURE_NAMES = {"candidate_miss": "候選漏找", "ranking": "選框順位", "qwen_decision": "Qwen 裁決",
                 "qc_miss": "QC 漏攔", "unknown": "證據不足"}


class BatchAuditView(tk.Frame):
    def __init__(self, parent, *, canvas_class, label_factory, background):
        super().__init__(parent, bg=background)
        self.job, self.entry, self.selected, self.details = None, None, None, None
        self.sample_ids, self.drafts = [], {}
        self.summary = tk.StringVar(master=self)
        self.message = tk.StringVar(master=self)
        self.summary_label = label_factory(self, variable=self.summary)
        self.summary_label.pack(fill="x", pady=(2, 4))

        toolbar = tk.Frame(self, bg=background)
        toolbar.pack(fill="x")
        self.prepare_button = ttk.Button(toolbar, text="封存抽檢清冊", command=self.prepare)
        self.prepare_button.pack(side="left", padx=(0, 5))
        self.counts = {}
        for key, title in STRATUM_NAMES.items():
            label_factory(toolbar, title, size=9).pack(side="left", padx=(0, 3))
            variable = self.counts[key] = tk.StringVar(master=self, value="6" if key == AUTO_CANDIDATE else "3")
            ttk.Entry(toolbar, width=3, textvariable=variable).pack(side="left", padx=(0, 5))

        self.sample_picker = ttk.Combobox(self, state="disabled")
        self.sample_picker.pack(fill="x", pady=(6, 2))
        self.sample_picker.bind("<<ComboboxSelected>>", self.pick_sample)
        self.message_label = label_factory(self, variable=self.message, size=9)
        self.message_label.pack(fill="x")
        self.canvas = canvas_class(self, lambda _: None)
        self.canvas.pack(fill="both", expand=True, pady=5)

        controls = tk.Frame(self, bg=background)
        controls.pack(fill="x")
        self.actor = tk.StringVar(master=self)
        ttk.Label(controls, text="覆核者").pack(side="left")
        ttk.Entry(controls, textvariable=self.actor, width=12).pack(side="left", padx=4)
        self.class_picker = ttk.Combobox(controls, state="disabled", width=12)
        self.class_picker.pack(side="left")
        self.class_picker.bind("<<ComboboxSelected>>", lambda _: setattr(self.canvas, "label", self.class_picker.get()))
        self.delete_button = ttk.Button(controls, text="刪盲標框", command=self.delete_box)
        self.delete_button.pack(side="left", padx=4)
        self.seal_button = ttk.Button(controls, text="封存盲標並比對", command=self.seal)
        self.seal_button.pack(side="right")

        self.verdict = tk.Frame(self, bg=background)
        self.errors = {}
        for key, text in REASONS.items():
            variable = self.errors[key] = tk.BooleanVar(master=self)
            ttk.Checkbutton(self.verdict, text=text, variable=variable).pack(side="left")
        self.no_error = tk.BooleanVar(master=self)
        ttk.Checkbutton(self.verdict, text="整張無誤", variable=self.no_error).pack(side="left")
        ttk.Button(self.verdict, text="提交抽檢結果", command=self.submit).pack(side="right")
        self.feedback = tk.StringVar(master=self)
        self.feedback_label = label_factory(self, variable=self.feedback, size=9)
        self.feedback_label.pack(fill="x", pady=(3, 0))
        self.bind("<Configure>", self._fit_text)

    def _fit_text(self, event):
        for label in (self.summary_label, self.message_label, self.feedback_label):
            label.configure(wraplength=max(220, event.width - 6))

    def load(self, job, *, frame_id=None):
        previous = (self.job.directory if self.job else None, self.plan_hash)
        self.job, self.entry = job, None
        self.sample_ids = []
        self.summary.set("影子驗收 · 選擇已完成裁決的作業")
        self.feedback.set("")
        self.prepare_button.state(["disabled"])
        values = []
        if job is not None:
            try:
                result = overview(job)
                self.entry = result["entry"]
                self.prepare_button.state(["disabled" if self.entry else "!disabled"])
                status = (STATUS_NAMES[result["result"]["status"]] if self.entry else
                          "內容已變更，請重新封存" if result["prior_rounds"] else "尚未封存")
                counts = [f"{title} {result['populations'][key]}" for key, title in STRATUM_NAMES.items()]
                self.summary.set(f"影子驗收 · {status} · " + "／".join(counts)
                                 + f" · 原流程 AI 採用 {result['original_auto']} · 重查 {result['retries']} 次")
                if self.entry:
                    samples = self.entry["plan"]["sample_ids"]
                    names = result["rows"]
                    membership = {ident: key for key in STRATUM_NAMES for ident in samples[key]}
                    self.sample_ids = sorted(membership, key=lambda ident: sha256(f"{self.plan_hash}:{ident}".encode()).digest())
                    for ident in self.sample_ids:
                        title = STRATUM_NAMES[membership[ident]] if ident in self.entry["references"] else "待盲標"
                        label = "已檢 " if ident in self.entry["reviews"] else ""
                        values.append(f"{label}{title} · {names[ident]['name']}")
                    rows = result["result"]["strata"]
                    progress = [f"{name} {rows[key]['reviewed']}/{rows[key]['sample_size']}，"
                                f"差異 {rows[key]['errors']}" for key, name in STRATUM_NAMES.items()]
                    auto = rows[AUTO_CANDIDATE]
                    bound = auto["upper_bound"]
                    if bound is not None:
                        progress.append(f"自動候選上界 {bound:.1%}")
                    records = [event for saved_job in jobs_for(Project.open(job.project_dir))
                               for event in feedback_events(saved_job)]
                    evidence = summarize_feedback(records)
                    diagnosed = evidence["rows"]
                    if diagnosed:
                        progress.extend(f"{FAILURE_NAMES.get(row['kind'], row['kind'])} {row['label']} "
                                        f"{row['frame_count']} 張／{row['source_count']} 來源"
                                        for row in diagnosed[:2])
                    if evidence["evidence_status"] == "pending_policy":
                        progress.append("改版樣本門檻待核准")
                    self.feedback.set(" · ".join(progress))
            except (OSError, ValueError, KeyError) as error:
                self.summary.set(str(error))
        if previous != (job.directory if job else None, self.plan_hash):
            self.drafts.clear()
            self.selected = None
        self.sample_picker.configure(values=values, state="readonly" if values else "disabled")
        if values:
            next_id = next((ident for ident in self.sample_ids if ident not in self.entry["reviews"]), self.sample_ids[0])
            ident = frame_id if frame_id in self.sample_ids else next_id
            self.sample_picker.current(self.sample_ids.index(ident))
        else:
            self.sample_picker.set("")
        self.pick_sample()

    @property
    def plan_hash(self):
        return self.entry["plan"]["sha256"] if self.entry else None

    def prepare(self):
        try:
            counts = {key: int(variable.get()) for key, variable in self.counts.items()}
            prepare_audit(self.job, sample_sizes=counts, seed=20261120)
        except (OSError, ValueError) as error:
            self.message.set(str(error))
            return
        self.load(self.job)

    def pick_sample(self, _event=None):
        if self.selected is not None and self.canvas.editable:
            self.drafts[self.selected] = deepcopy(self.canvas.boxes)
        index = self.sample_picker.current()
        self.selected = self.sample_ids[index] if 0 <= index < len(self.sample_ids) else None
        self.details = None
        self.verdict.pack_forget()
        self.class_picker.configure(state="disabled", values=())
        self.class_picker.set("")
        self.canvas.show(None, [])
        self.seal_button.state(["disabled"])
        self.delete_button.state(["disabled"])
        if self.selected is None:
            self.message.set("先封存抽檢清冊，再獨立標註樣本整張影像。")
            return
        try:
            details = audit_frame(self.job, self.selected, plan_sha256=self.plan_hash)
            with Image.open(details["image"]) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            self.message.set(str(error))
            return
        self.details = details
        reference = details["reference"]
        blind = reference is None
        if blind:
            self.class_picker.configure(values=details["labels"], state="readonly")
            self.class_picker.current(0)
            self.canvas.label = self.class_picker.get()
            self.seal_button.state(["!disabled"])
            self.delete_button.state(["!disabled"])
            self.canvas.show(image, deepcopy(self.drafts.get(self.selected, [])), editable=True)
            self.message.set("請獨立標完所有類別。空圖確認後可直接封存。")
        else:
            current = [{**box, "source": "manual"} for box in reference["boxes"]]
            self.canvas.show(image, current, overlay=details["comparison_boxes"])
            if details["review"] is None:
                for variable in self.errors.values():
                    variable.set(False)
                self.no_error.set(False)
                self.verdict.pack(fill="x", before=self.feedback_label)
                self.message.set(f"{reference['actor']} 已封存盲標。淺藍為盲標，橘線為抽檢前框。")
            else:
                reasons = "、".join(REASONS[key] for key in details["review"]["issues"])
                self.message.set(f"已提交 · {reasons or '無錯誤'}")

    def delete_box(self):
        if self.canvas.editable and self.canvas.selected is not None:
            del self.canvas.boxes[self.canvas.selected]
            self.canvas.selected = None
            self.canvas.render()

    def seal(self):
        if self.selected is None or not self.canvas.editable:
            return
        try:
            seal_reference(self.job, self.selected, self.canvas.boxes, actor=self.actor.get(), plan_sha256=self.plan_hash)
        except (OSError, ValueError) as error:
            self.message.set(str(error))
            return
        self.pick_sample()

    def submit(self):
        if self.selected is None or self.details is None or self.details["review"] is not None:
            return
        issues = [key for key, variable in self.errors.items() if variable.get()]
        if self.no_error.get() == bool(issues):
            self.message.set("請明確勾選差異類型或「整張無誤」，兩者擇一。")
            return
        try:
            submit_audit(self.job, self.selected, issues=issues, actor=self.actor.get(), plan_sha256=self.plan_hash)
        except (OSError, ValueError) as error:
            self.message.set(str(error))
            return
        self.load(self.job)
