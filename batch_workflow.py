"""Persist the workbench's shadow acceptance and blind audit evidence."""

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json

from batch_acceptance import AUTO_CANDIDATE, HELD, HUMAN_CONFIRMED, STRATA, assess_batch, freeze_batch
from generic_job import _boxes, _labels
from generic_qc import usable_box
from project import _digest, _project_lock, _write_json
from review_diagnosis import diagnose_reviewed_frame


STORE = "batch-quality-shadow.json"
SCHEMA = "annotation_wizard.batch_quality_shadow.v1"
REASONS = {
    "omission": "漏框", "false_positive": "誤框", "class_error": "類別錯誤", "extent_error": "框界錯誤",
}
PROTECTED_MARKERS = (
    "door-ab-blind.json", "acceptance-inventory.json",
    "workbench-copy-provenance.json", "review-learning-demo.json",
)


def _identity(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_development(job):
    if any((job.project_dir / marker).is_file() for marker in PROTECTED_MARKERS):
        raise ValueError("獨立盲標或驗收專案，以及示範專案，不能收集開發覆核證據。")


def _inventory(job):
    _require_development(job)
    job.refresh()
    if job.state["status"] != "review_ready":
        raise ValueError("請先完成本批推論或建立人工標註作業。")
    rows = {row["id"]: row for row in job.frame_rows()}
    inventory = []
    for ident, row in rows.items():
        qc = row.get("shadow_qc") or {}
        if row["status"] == "approved":
            stratum = HUMAN_CONFIRMED
        elif row["status"] == "ai_ready" and qc.get("disposition") == AUTO_CANDIDATE:
            stratum = AUTO_CANDIDATE
        else:
            stratum = HELD
        revision = _identity({
            "manifest": job.state["manifest_sha256"], "inference": job.state["inference_sha256"],
            "image": row["image_sha256"], "project": row["project_annotation_revision"],
            "status": row["status"], "boxes": row["boxes"], "qc": qc,
        })
        inventory.append({"frame_id": ident, "stratum": stratum, "revision": revision})
    return sorted(inventory, key=lambda row: row["frame_id"]), rows


def _state(job):
    path = job.directory / STORE
    if not path.exists():
        return {"schema": SCHEMA, "job_id": job.directory.name, "rounds": {}, "corrections": {}}
    if path.is_symlink() or not path.is_file():
        raise ValueError("批次覆核紀錄的位置無效。")
    result = json.loads(path.read_bytes())
    if (not isinstance(result, dict) or result.get("schema") != SCHEMA
            or result.get("job_id") != job.directory.name or not isinstance(result.get("rounds"), dict)):
        raise ValueError("批次覆核紀錄不屬於目前作業。")
    for revision, entry in result["rounds"].items():
        if not isinstance(entry, dict) or entry.get("sha256") != _identity(
            {key: value for key, value in entry.items() if key != "sha256"}
        ) or revision != _identity(entry["plan"]["frames"]):
            raise ValueError("批次覆核紀錄的雜湊不符。")
    corrections = result.setdefault("corrections", {})
    if not isinstance(corrections, dict) or any(_identity(event) != token for token, event in corrections.items()):
        raise ValueError("人工修正證據的雜湊不符。")
    return result


def _save(job, state, inventory, entry):
    entry["sha256"] = _identity({key: value for key, value in entry.items() if key != "sha256"})
    state["rounds"][_identity(inventory)] = entry
    _write_json(job.directory / STORE, state)


def _active(state, inventory):
    return state["rounds"].get(_identity(inventory))


def _audit_result(entry, inventory):
    reviews = {ident: row["error"] for ident, row in entry["reviews"].items()
               if entry["references"][ident]["role"] == "operator"}
    result = assess_batch(entry["plan"], reviews, current_frames=inventory)
    result["demonstration_reviews"] = len(entry["reviews"]) - len(reviews)
    return result


def _image(job, row):
    path = (job.project_dir / row["image"]).resolve(strict=True)
    if not path.is_relative_to(job.project_dir / "images") or _digest(path) != row["image_sha256"]:
        raise ValueError("原圖的位置或內容已變更，請核對封存畫格。")
    return path


def overview(job):
    inventory, rows = _inventory(job)
    state = _state(job)
    entry = _active(state, inventory)
    counts = Counter(item["stratum"] for item in inventory)
    return {
        "populations": {key: counts[key] for key in STRATA},
        "rows": rows, "entry": entry, "result": _audit_result(entry, inventory) if entry else None,
        "prior_rounds": len(state["rounds"]) - bool(entry),
        "original_auto": sum(row["status"] == "ai_ready" for row in rows.values()),
        "retries": sum((row.get("shadow_qc") or {}).get("retry_count", 0) for row in rows.values()),
        "held_by_source": dict(Counter(rows[item["frame_id"]]["source_group"]
                                      for item in inventory if item["stratum"] == HELD)),
    }


def prepare_audit(job, *, sample_sizes, seed, policy=None):
    """Freeze one sampling plan per unchanged frame inventory."""
    with _project_lock(job.directory):
        inventory, rows = _inventory(job)
        for row in rows.values():
            _image(job, row)
        plan = freeze_batch(inventory, sample_sizes=sample_sizes, seed=seed, policy=policy)
        state = _state(job)
        current = _active(state, inventory)
        if current:
            if current["plan"] != plan:
                raise ValueError("畫格版本未變更，此批沿用既有抽樣清冊。")
            return plan
        _save(job, state, inventory, {"plan": plan, "created_at": _now(), "references": {}, "reviews": {}})
        return plan


def _sample(entry, frame_id, plan_sha256):
    if (entry is None or entry["plan"]["sha256"] != plan_sha256
            or not any(frame_id in members for members in entry["plan"]["sample_ids"].values())):
        raise ValueError("請重新開啟抽檢。畫格內容或清冊可能已變更。")


def audit_frame(job, frame_id, *, plan_sha256):
    inventory, rows = _inventory(job)
    entry = _active(_state(job), inventory)
    _sample(entry, frame_id, plan_sha256)
    _audit_result(entry, inventory)
    row = rows[frame_id]
    reference = entry["references"].get(frame_id)
    return {
        "image": _image(job, row), "name": row["name"],
        "labels": _labels(job.manifest),
        "reference": reference,
        "comparison_boxes": row["boxes"] if reference is not None else [],
        "review": entry["reviews"].get(frame_id),
    }


def seal_reference(job, frame_id, boxes, *, actor, plan_sha256, role="operator"):
    """Save the complete blind annotation before exposing the pre-audit boxes."""
    if not isinstance(actor, str) or not actor.strip() or len(actor.strip()) > 80:
        raise ValueError("請輸入 1–80 字的覆核者名稱。")
    if role not in {"operator", "demonstration"}:
        raise ValueError("覆核者用途無效。")
    with _project_lock(job.directory):
        inventory, rows = _inventory(job)
        state = _state(job)
        entry = _active(state, inventory)
        _sample(entry, frame_id, plan_sha256)
        _audit_result(entry, inventory)
        if frame_id in entry["references"]:
            raise ValueError("該畫格的人工盲標已封存。")
        row = rows[frame_id]
        _image(job, row)
        reference = _boxes(boxes, row, _labels(job.manifest))
        entry["references"][frame_id] = {
            "actor": actor.strip(), "role": role, "boxes": reference, "sealed_at": _now(),
        }
        _save(job, state, inventory, entry)


def _source_digest(group):
    role, _, value = group.partition(":")
    if role in {"video", "image"} and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return value
    return _identity(group)


def _missing(before, reference):
    pool = [candidate for candidate in before.get("candidates", [])
            if usable_box(candidate.get("xyxy"), before["width"], before["height"])]
    return diagnose_reviewed_frame(
        reference_boxes=reference, candidates=pool, accepted_boxes=before["boxes"],
        reviewed_candidate_ids=[candidate["id"] for candidate in pool if candidate.get("votes")],
        pool_complete=("candidates" in before),
    )["targets"]


def _event(event_id, row, actor, role, diagnoses, issues=()):
    return {
        "review_id": event_id, "actor": actor, "role": role, "data_role": "development",
        "whole_frame_human_review": True,
        "image_sha256": row["image_sha256"], "source_sha256": _source_digest(row["source_group"]),
        "issues": list(issues), "diagnoses": [{"kind": kind, "label": label} for kind, label in sorted(diagnoses)],
    }


def _feedback(event_id, row, reference, issues, stratum):
    diagnoses = set()
    if "omission" in issues:
        diagnoses.update((item["cause"], item["label"]) for item in _missing(row, reference["boxes"])
                         if item["status"] == "missed")
    labels = {box["label"] for box in (*reference["boxes"], *row["boxes"])}
    if not issues and stratum == HELD:
        diagnoses.update(("qc_false_hold", box["label"]) for box in row["boxes"])
    if len(labels) == 1:
        label = next(iter(labels))
        for issue, kind in (("false_positive", "false_positive"), ("extent_error", "box_geometry"),
                            ("class_error", "category_conflict")):
            if issue in issues:
                diagnoses.add((kind, label))
    if issues and stratum == AUTO_CANDIDATE:
        diagnoses.update(("qc_miss", label) for _, label in diagnoses.copy())
    role = "audit" if reference["role"] == "operator" else "demonstration"
    return _event(event_id, row, reference["actor"], role, diagnoses, issues)


def submit_audit(job, frame_id, *, issues, plan_sha256, actor):
    """Record the human adjudication, bound to the prior and blinded annotations."""
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("請輸入覆核者名稱。")
    if (not isinstance(issues, (list, tuple))
            or any(not isinstance(issue, str) or issue not in REASONS for issue in issues)):
        raise ValueError("抽檢結論包含未知的錯誤類型。")
    issues = tuple(dict.fromkeys(issues))
    with _project_lock(job.directory):
        inventory, rows = _inventory(job)
        state = _state(job)
        entry = _active(state, inventory)
        _sample(entry, frame_id, plan_sha256)
        _audit_result(entry, inventory)
        reference = entry["references"].get(frame_id)
        if reference is None or actor.strip() != reference["actor"] or frame_id in entry["reviews"]:
            raise ValueError("須由原覆核者先提交整張盲標，且每個樣本只能計分一次。")
        row = rows[frame_id]
        _image(job, row)
        stratum = next(item["stratum"] for item in inventory if item["frame_id"] == frame_id)
        entry["reviews"][frame_id] = {
            "error": bool(issues), "issues": list(issues), "reviewed_at": _now(),
            "feedback": _feedback(f"{plan_sha256}:{frame_id}", row, reference, issues, stratum),
        }
        _save(job, state, inventory, entry)
        return _audit_result(entry, inventory)


def feedback_events(job):
    """Read audited evidence only, with source and operator roles preserved."""
    _require_development(job)
    state = _state(job)
    return [*state["corrections"].values(),
            *(review["feedback"] for entry in state["rounds"].values() for review in entry["reviews"].values())]


def record_correction(job, frame_id, *, actor):
    """Persist an operator's submitted whole-frame correction against sealed predictions."""
    if not isinstance(actor, str) or not actor.strip() or len(actor.strip()) > 80:
        raise ValueError("請輸入 1–80 字的覆核者名稱。")
    with _project_lock(job.directory):
        _, rows = _inventory(job)
        row = rows.get(frame_id)
        report = job.inference
        if not isinstance(row, Mapping) or row["status"] != "approved" or not report:
            return None
        model = next(item for item in report["frames"] if item["id"] == frame_id)
        if (model.get("shadow_qc") or {}).get("disposition") not in {"review", "held"} and not model["exceptions"]:
            return None
        _image(job, row)
        state = _state(job)
        key = _identity([job.state["inference_sha256"], frame_id, row["project_annotation_revision"]])
        previous = next((event for event in state["corrections"].values() if event.get("review_id") == key), None)
        if previous:
            return previous
        targets = _missing({**row, **model}, row["boxes"])
        diagnoses = {(item["cause"], item["label"]) for item in targets if item["status"] == "missed"}
        event = _event(key, row, actor.strip(), "operator", diagnoses, ("omission",) if diagnoses else ())
        state["corrections"][_identity(event)] = event
        _write_json(job.directory / STORE, state)
        return event
