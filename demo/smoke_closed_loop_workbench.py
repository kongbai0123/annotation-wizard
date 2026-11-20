"""Exercise the real Tk workbench on temporary images without opening a visible window."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import tkinter as tk

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import MODEL_DIR, WizardApp
from batch_workflow import STORE, overview
from generic_job import INFERENCE_SCHEMA, create_job
from generic_workbench import SHADOW_FILTER
from project import Project


BOX = {"label": "valve", "xyxy": [16, 18, 82, 92]}
MISSED = {"label": "valve", "xyxy": [112, 20, 172, 89]}


def fingerprint(paths):
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def fixture(directory):
    project = Project.create(directory / "project")
    paths = []
    for index in range(3):
        image = Image.new("RGB", (200, 120), (20, 32 + index, 62))
        draw = ImageDraw.Draw(image)
        draw.rectangle(BOX["xyxy"], outline="#93b5ef", width=3)
        if index == 0:
            draw.rectangle(MISSED["xyxy"], outline="#dcad7c", width=3)
        path = directory / f"fixture-{index}.png"
        image.save(path)
        paths.append(path)
    project.add_images(paths)
    job = create_job(project, "valve | valve | visible extent of each valve")
    rows = []
    for index, frame in enumerate(job.manifest["frames"]):
        rows.append({
            "id": frame["id"], "image_sha256": frame["image_sha256"],
            "boxes": [BOX], "exceptions": [],
            "candidates": [{**BOX, "id": "mm-1", "score": 0.9, "verdict": "accepted"}],
            "shadow_qc": {
                "disposition": "review" if index == 1 else "auto_candidate",
                "retry_count": 1 if index == 1 else 0,
                "flags": [{"code": "retry_requires_review", "reason": "已重查，須整張複核。"}] if index == 1 else [],
                "checks": [{"candidate_id": "mm-1", "sam_status": "measured", "sam_box_iou": .9,
                            "sam_box": [17, 19, 81, 91]}] if index == 0 else [],
            },
        })
    source = directory / "inference-fixture.json"
    source.write_text(json.dumps({"schema": INFERENCE_SCHEMA, "job_id": job.manifest["job_id"],
                                  "manifest_sha256": job.state["manifest_sha256"], "frames": rows}), encoding="utf-8")
    job.publish_inference(source)
    job.review(rows[2]["id"], [BOX])
    return project, job, rows


def run(directory):
    checks = []

    def require(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)
        print(f"PASS {description}", flush=True)

    project, job, rows = fixture(directory)
    frozen_paths = [project.directory / "project.json", job.manifest_path, job.inference_path,
                    job.directory / "state.json"]
    frozen_before = fingerprint(frozen_paths)
    root = tk.Tk()
    root.withdraw()
    callback_errors = []
    root.report_callback_exception = lambda *_error: callback_errors.append(_error)
    app = WizardApp(root, MODEL_DIR)
    try:
        app.open_path(project.directory)
        require(app.project.directory == project.directory, "open temporary project in actual WizardApp")
        require(app.show_generic_ai(), "open batch annotation in existing Tk root")
        pane = app.ai_pane
        pane.jobs_picker.current(0)
        pane._open_job()
        root.update()
        pane._pick_frame(rows[0]["id"])
        pane._set_preview(True)
        pane.preview_tabs.select(pane.details_tab)
        pane.candidate_picker.current(0)
        pane._candidate_selected()
        require("90.0%" in pane.candidate_notes and len(pane.canvas.overlay) == 2,
                "view SAM IoU and overlay beside the selected MM candidate")
        pane._set_preview(False)
        pane.review_filter.set(SHADOW_FILTER)
        pane._choose_filter()
        require(list(pane.frame_ids) == [rows[1]["id"]], "open shadow exception filter and show the retried frame")
        require("重查 1 次" in pane.decision_text.cget("text"), "render per-frame retry reason")
        pane.audit_button.invoke()
        root.update()
        require(pane.audit_open and pane.batch_audit.winfo_toplevel() is root, "show audit in same top-level window")
        audit = pane.batch_audit
        audit.prepare_button.invoke()
        root.update()
        require(audit.plan_hash is not None and len(audit.sample_ids) == 3, "freeze all three sampling strata in GUI")
        require(all(label.startswith("待盲標") for label in audit.sample_picker["values"]),
                "conceal per-frame sampling stratum until the blind annotation is sealed")
        audit.sample_picker.current(audit.sample_ids.index(rows[0]["id"]))
        audit.pick_sample()
        require(audit.canvas.picture is not None and audit.canvas.editable and not audit.canvas.overlay,
                "load original pixels with candidate overlay hidden")
        audit.actor.set("synthetic smoke fixture")
        audit.canvas.boxes = [BOX, MISSED]
        audit.seal_button.invoke()
        root.update()
        require(not audit.canvas.editable and len(audit.canvas.boxes) == 2 and len(audit.canvas.overlay) == 1,
                "seal full-frame annotation before showing the comparison")
        audit.submit()
        require(not overview(job)["entry"]["reviews"], "reject unmarked adjudication instead of counting it as no error")
        audit.errors["omission"].set(True)
        next_frame = audit.selected
        audit.submit()
        root.update()
        result = overview(job)["result"]
        require(result["strata"]["auto_candidate"]["errors"] == 1 and result["status"] == "pending_policy",
                "save an audit defect and keep production thresholds unapproved")
        require(audit.selected != next_frame, "advance to the next unreviewed sample")
        pane.audit_button.invoke()
        require(not pane.audit_open and app.show_generic_training(), "return to metrics and switch to training tab")
        require(app.active_workspace == "train" and pane.app.root is root, "training tab shares the same window")
        require(overview(job)["result"] == result and (job.directory / STORE).is_file(), "read persisted shadow evidence")
        require(fingerprint(frozen_paths) == frozen_before, "keep project, task, and sealed inference byte-identical")
        require(not callback_errors, "no Tk callback raised an exception")
    finally:
        app.close()
    return {"checks": checks, "sha256_unchanged": frozen_before, "fixture": "synthetic", "visible_windows": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    with TemporaryDirectory(prefix="annotation-wizard-closed-loop-") as temporary:
        report = run(Path(temporary))
    report.update(status="passed", seconds=round(perf_counter() - start, 3))
    args.receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {len(report['checks'])} checks in {report['seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
