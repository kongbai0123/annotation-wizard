"""Shadow-only evidence and release checks for annotation review rules between batches."""

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite


EVIDENCE_KINDS = {
    "candidate_miss", "ranking", "qwen_decision", "qc_miss", "qc_false_hold",
    "box_geometry", "category_conflict", "merge", "duplicate", "false_positive",
}
DEVELOPMENT_ROLES = {"operator", "audit", "development_review", "development_audit"}


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _bound(record):
    return {**record, "sha256": _digest(record)}


def _verified(record, schema):
    if not isinstance(record, dict) or record.get("schema") != schema:
        raise ValueError("Unexpected batch feedback record schema.")
    payload = {key: value for key, value in record.items() if key != "sha256"}
    if record.get("sha256") != _digest(payload):
        raise ValueError(f"Changed or invalid {schema} record.")
    return record


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _sha(value, name):
    _text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _minimum(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return value


def _rate(value, name):
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite rate between zero and one.")
    return value


def _scope(row):
    kind, label = row.get("kind"), row.get("label")
    if kind not in EVIDENCE_KINDS:
        raise ValueError("Unknown or unsupported feedback failure type.")
    return kind, _text(label, "class label")


def _development_events(events):
    """Keep verified human development records and count each image only once per class and failure."""
    selected = {}
    image_sources = {}
    excluded = 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Feedback must be a record.")
        if (event.get("role") not in DEVELOPMENT_ROLES
                or event.get("data_role", "development") not in {"development", "training", "train"}
                or event.get("whole_frame_human_review", True) is not True):
            excluded += 1
            continue
        ident = _text(event.get("review_id", event.get("id")), "feedback event ID")
        source = _sha(event.get("source_sha256"), "source identity")
        image = _sha(event.get("image_sha256"), "image identity")
        if image_sources.setdefault(image, source) != source:
            raise ValueError("The same image was attributed to multiple development sources.")
        scopes = set()
        for row in event.get("diagnoses", []):
            if not isinstance(row, dict):
                raise ValueError("Feedback diagnoses must be records.")
            if row.get("kind") in EVIDENCE_KINDS:
                scopes.add(_scope(row))
        normalized = {"id": ident, "role": event["role"], "source_sha256": source,
                      "image_sha256": image, "scopes": [list(scope) for scope in sorted(scopes)]}
        if selected.setdefault(ident, normalized) != normalized:
            raise ValueError("A feedback event ID was reused for conflicting evidence.")
    return [selected[key] for key in sorted(selected)], excluded


def summarize_feedback(events, evidence_policy=None):
    """Summarize existing review_learning cases without promoting or persisting a rule."""
    minimum_frames, minimum_sources = None, None
    if evidence_policy is not None:
        minimum_frames = _minimum(evidence_policy.get("min_frames"), "min_frames", 1)
        minimum_sources = _minimum(evidence_policy.get("min_sources"), "min_sources", 1)
        if minimum_sources > minimum_frames:
            raise ValueError("min_sources greater than min_frames is inconsistent.")
        evidence_policy = {"min_frames": minimum_frames, "min_sources": minimum_sources}
    selected, excluded = _development_events(events)
    frames_by_scope = defaultdict(set)
    sources_by_scope = defaultdict(set)
    for event in selected:
        for values in event["scopes"]:
            scope = tuple(values)
            frames_by_scope[scope].add(event["image_sha256"])
            sources_by_scope[scope].add(event["source_sha256"])
    rows = []
    for (kind, label), frames in sorted(frames_by_scope.items()):
        sources = sources_by_scope[kind, label]
        ready = evidence_policy is not None and len(frames) >= minimum_frames and len(sources) >= minimum_sources
        rows.append({"kind": kind, "label": label, "frame_count": len(frames), "source_count": len(sources),
                     "ready": ready})
    return {"evidence_status": ("pending_policy" if evidence_policy is None else
                                "ready" if any(row["ready"] for row in rows) else "insufficient_evidence"),
            "policy": evidence_policy, "rows": rows, "excluded_events": excluded, "accepted_events": len(selected),
            "evidence_sha256": _digest(selected),
            "development_sources": sorted({event["source_sha256"] for event in selected}),
            "development_images": sorted({event["image_sha256"] for event in selected})}


def _version(number, rules):
    if not isinstance(rules, (dict, list)):
        raise ValueError("Rule snapshots must be JSON objects or arrays.")
    return _bound({"schema": "annotation_wizard.rule_snapshot.v1", "number": number, "rules": deepcopy(rules)})


def new_registry(rules):
    """Seed an in-memory version registry with the caller's current rules."""
    initial = _version(1, rules)
    return _bound({"schema": "annotation_wizard.rule_registry.v1", "active_sha256": initial["sha256"],
                   "versions": [initial], "history": []})


def _active(registry):
    _verified(registry, "annotation_wizard.rule_registry.v1")
    for version in registry["versions"]:
        _verified(version, "annotation_wizard.rule_snapshot.v1")
    return next(version for version in registry["versions"] if version["sha256"] == registry["active_sha256"])


def batch_snapshot(registry, batch_id):
    """Freeze a copy of the rules for use by one batch."""
    return _bound({"schema": "annotation_wizard.rule_batch_snapshot.v1",
                   "batch_id": _text(batch_id, "batch ID"), "version": deepcopy(_active(registry))})


def prepare_release(registry, rules, target, events, evidence_policy=None):
    """Bind an intended rule version to the active version and human evidence."""
    active = _active(registry)
    kind, label = _scope(target)
    if _digest(active["rules"]) == _digest(rules):
        raise ValueError("An unchanged ruleset is not a release candidate.")
    evidence = summarize_feedback(events, evidence_policy)
    ready = any((row["kind"], row["label"]) == (kind, label) and row["ready"] for row in evidence["rows"])
    return _bound({
        "schema": "annotation_wizard.rule_release_proposal.v1",
        "baseline_sha256": active["sha256"], "target": {"kind": kind, "label": label},
        "candidate": _version(registry["versions"][-1]["number"] + 1, rules), "evidence": evidence,
        "status": "ready" if ready else evidence["evidence_status"] if evidence_policy is None else "insufficient_evidence",
    })


def _trial_inputs(trial, snapshot_sha256):
    if trial.get("snapshot_sha256") != snapshot_sha256:
        raise ValueError("Shadow trial used the wrong rule snapshot.")
    inputs = trial["inputs"]
    frames = inputs["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("Shadow comparison needs a fixed, nonempty evaluation set.")
    if not isinstance(inputs.get("budget"), dict) or not inputs["budget"]:
        raise ValueError("Shadow comparison needs a fixed review budget.")
    _sha(inputs.get("candidate_pool_sha256"), "candidate pool identity")
    sources, images = set(), set()
    for frame in frames:
        if frame.get("role") != "independent_evaluation":
            raise ValueError("All shadow evaluation frames must be marked for independent evaluation.")
        sources.add(_sha(frame.get("source_sha256"), "evaluation source identity"))
        identity = _sha(frame.get("image_sha256"), "evaluation image identity")
        if identity in images:
            raise ValueError("The independent evaluation image list contains duplicates.")
        images.add(identity)
        _sha(frame.get("human_reference_sha256"), "human reference identity")
    return inputs, sources, images


def _trial_metrics(trial, total_frames):
    metrics = trial["metrics"]
    values = {key: _minimum(metrics.get(key), key, 0)
              for key in ("auto_candidates", "false_releases", "manual_review", "held")}
    if (values["auto_candidates"] + values["manual_review"] + values["held"] != total_frames
            or values["false_releases"] > values["auto_candidates"]):
        raise ValueError("Shadow results do not account for every evaluation frame.")
    elapsed = metrics.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be a finite nonnegative number.")
    candidates, errors = values["auto_candidates"], values["false_releases"]
    return {**values, "elapsed_seconds": elapsed, "eligible_correct": candidates - errors,
            "false_release_rate": errors / candidates if candidates else None}


def compare_release(proposal, baseline, candidate, quality_policy=None):
    """Compare matched shadow runs. The caller must supply human evaluation and policy."""
    _verified(proposal, "annotation_wizard.rule_release_proposal.v1")
    version = _verified(proposal["candidate"], "annotation_wizard.rule_snapshot.v1")
    before, _, _ = _trial_inputs(baseline, proposal["baseline_sha256"])
    after, sources, images = _trial_inputs(candidate, version["sha256"])
    fingerprint = _digest(after)
    if _digest(before) != fingerprint:
        raise ValueError("Shadow trials must share evaluation frames, references, candidate pool and budget.")
    evidence = proposal["evidence"]
    if sources.intersection(evidence["development_sources"]) or images.intersection(evidence["development_images"]):
        raise ValueError("Development evidence and independent evaluation overlap.")
    base_metrics, new_metrics = (_trial_metrics(trial, len(images)) for trial in (baseline, candidate))
    issues = []
    if proposal["status"] != "ready":
        issues.append(proposal["status"])
    if quality_policy is None:
        issues.append("pending_quality_policy")
    else:
        max_rate = _rate(quality_policy.get("max_false_release_rate"), "max_false_release_rate")
        min_candidates = _minimum(quality_policy.get("min_auto_candidates"), "min_auto_candidates", 1)
        max_extra_review = _minimum(quality_policy.get("max_additional_reviews"), "max_additional_reviews", 0)
        if new_metrics["auto_candidates"] < min_candidates:
            issues.append("insufficient_candidate_coverage")
        if new_metrics["false_release_rate"] is None or new_metrics["false_release_rate"] > max_rate:
            issues.append("false_release_limit_not_met")
        if (base_metrics["false_release_rate"] is not None
                and new_metrics["false_release_rate"] is not None
                and new_metrics["false_release_rate"] > base_metrics["false_release_rate"]):
            issues.append("false_release_rate_regressed")
        if new_metrics["manual_review"] - base_metrics["manual_review"] > max_extra_review:
            issues.append("review_budget_regressed")
        if new_metrics["eligible_correct"] < base_metrics["eligible_correct"]:
            issues.append("correct_auto_candidates_regressed")
        improved = (new_metrics["false_releases"] < base_metrics["false_releases"]
                    or new_metrics["eligible_correct"] > base_metrics["eligible_correct"]
                    or (new_metrics["manual_review"] < base_metrics["manual_review"]
                        and new_metrics["held"] <= base_metrics["held"]))
        if not improved:
            issues.append("no_improvement")
        quality_policy = {"max_false_release_rate": max_rate, "min_auto_candidates": min_candidates,
                          "max_additional_reviews": max_extra_review}
    return _bound({"schema": "annotation_wizard.rule_shadow_comparison.v1", "mode": "shadow",
                   "proposal_sha256": proposal["sha256"], "input_sha256": fingerprint,
                   "evaluation_frames": len(images), "evaluation_sources": len(sources),
                   "baseline": base_metrics, "candidate": new_metrics, "quality_policy": quality_policy,
                   "status": "eligible" if not issues else "blocked", "issues": issues})


def _approval(approval, binding):
    if not isinstance(approval, dict) or approval.get("binding_sha256") != binding or approval.get("human") is not True:
        raise ValueError("Explicit human approval bound to this result is required.")
    return _text(approval.get("actor"), "approver")


def _registry_update(registry, target, entry, new_version=None):
    versions = deepcopy(registry["versions"])
    if new_version is not None:
        versions.append(deepcopy(new_version))
    return _bound({"schema": registry["schema"], "active_sha256": target, "versions": versions,
                   "history": [*deepcopy(registry["history"]), entry]})


def promote_release(registry, proposal, comparison, approval):
    """Return a new registry. An existing batch snapshot still points to its original rules."""
    current = _active(registry)
    _verified(proposal, "annotation_wizard.rule_release_proposal.v1")
    _verified(comparison, "annotation_wizard.rule_shadow_comparison.v1")
    candidate = _verified(proposal["candidate"], "annotation_wizard.rule_snapshot.v1")
    if (proposal["baseline_sha256"] != current["sha256"]
            or candidate["number"] != registry["versions"][-1]["number"] + 1):
        raise ValueError("Baseline changed; run a new shadow comparison before approval.")
    if (proposal["status"] != "ready" or comparison["proposal_sha256"] != proposal["sha256"]
            or comparison["status"] != "eligible"):
        raise ValueError("The proposal lacks matching evidence and a passing shadow comparison.")
    actor = _approval(approval, comparison["sha256"])
    entry = {"action": "promote", "from": current["sha256"], "to": candidate["sha256"],
             "actor": actor, "comparison_sha256": comparison["sha256"]}
    return _registry_update(registry, candidate["sha256"], entry, candidate)


def rollback_binding(registry, version_sha256):
    """Bind a rollback approval to the exact current registry and an earlier snapshot."""
    current = _active(registry)
    if (current["sha256"] == version_sha256
            or not any(version["sha256"] == version_sha256 for version in registry["versions"])):
        raise ValueError("Rollback requires a different, known version.")
    return _digest({"action": "rollback", "registry_sha256": registry["sha256"], "to": version_sha256})


def rollback_release(registry, version_sha256, approval):
    binding = rollback_binding(registry, version_sha256)
    actor = _approval(approval, binding)
    entry = {"action": "rollback", "from": registry["active_sha256"], "to": version_sha256,
             "actor": actor, "binding_sha256": binding}
    return _registry_update(registry, version_sha256, entry)
