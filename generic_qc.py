"""Apply shadow per-frame QC using separate semantic and geometric evidence."""

from math import isclose, isfinite
from pathlib import Path
from time import perf_counter

from grounding import overlap


QC_POLICY_VERSION = "annotation_wizard.frame_qc.v1"
MAX_RETRY_ATTEMPTS = 2
RIVAL_IOU = 0.65
RIVAL_SCORE_RATIO = 0.80
SAM_MINIMUM_IOU = 0.60
MAX_SAM_BOXES = 2
UNMATCHED_MM_SCORE = 0.50

REASONS = {
    "retry_requires_review": "這張畫格至少重查一次，須整張人工確認。",
    "legacy_review_pending": "畫格仍有既有流程未解決的項目。",
    "no_boxes": "沒有可暫列放行的框，須確認是否漏框或無目標。",
    "invalid_geometry": "框的座標不在影像範圍內，或其寬高無效。",
    "qwen_class_uncertain": "兩種順序的類別證據不足或不一致。",
    "mm_class_rival": "MM 在同一位置提出分數相近的其他類別，須核對。",
    "structure_class_conflict": "獨立可見結構證據與指定類別衝突。",
    "mm_extent_rival": "MM 對同一類別提出接近分數但範圍不同的框。",
    "possible_merged_instances": "原框包含兩個分開的同類 MM 框，可能併框。",
    "sam_extent_conflict": "SAM 遮罩的外接框與候選框差異過大。",
    "sam_extent_uncertain": "SAM 未取得可量測的遮罩，須人工確認。",
    "unmatched_mm_proposal": "未採用的高分 MM 候選缺少相符的已採用框。",
}


def usable_box(box, width, height):
    return (
        isinstance(box, (tuple, list))
        and len(box) == 4
        and all(type(value) in (int, float) and isfinite(value) for value in box)
        and 0 <= box[0] < box[2] <= width
        and 0 <= box[1] < box[3] <= height
    )


def qwen_class_evidence(candidate, settings):
    """Pool the class-compatible votes A/B without treating extent as a QC verdict."""
    votes = candidate.get("votes", [])
    if not isinstance(votes, list) or len(votes) != 2:
        return "not_evaluated"
    winners = []
    for vote in votes:
        if not isinstance(vote, dict) or not isinstance(vote.get("probabilities"), dict):
            return "not_evaluated"
        distribution = vote["probabilities"]
        weights = [distribution.get(key) for key in "ABCD"]
        mass = vote.get("allowed_token_mass")
        if any(type(weight) not in (int, float) or not isfinite(weight) or not 0 <= weight <= 1
               for weight in [*weights, mass]):
            return "not_evaluated"
        if not isclose(sum(weights), 1, abs_tol=1e-5):
            return "not_evaluated"
        if mass < settings["qwen_minimum_mass"]:
            return "uncertain"
        positive, negative, unsure = weights[0] + weights[1], weights[2], weights[3]
        ranked = sorted(zip((positive, negative, unsure), ("target", "other", "uncertain")), reverse=True)
        margin = ranked[0][0] - ranked[1][0]
        if margin == 0 or margin < settings["qwen_minimum_margin"]:
            return "uncertain"
        winners.append(ranked[0][1])
    return winners[0] if winners[0] == winners[1] else "uncertain"


def coverage_of(inner, outer):
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    inside = (max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
              * max(0, min(inner[3], outer[3]) - max(inner[1], outer[1])))
    return inside / area


def mm_rivals(candidate, proposals, min_score):
    box = candidate["xyxy"]
    similar, extents, contained = [], [], []
    for rival in proposals:
        if candidate["id"] == rival["id"] or rival["score"] < min_score:
            continue
        other = rival["xyxy"]
        iou = overlap(box, other)
        if rival["label"] != candidate["label"]:
            if iou >= RIVAL_IOU:
                similar.append({"candidate_id": rival["id"], "label": rival["label"],
                                "score": rival["score"], "iou": iou})
            continue
        if 0.50 <= iou < 0.90:
            extents.append(rival["id"])
        if iou <= 0.70 and coverage_of(other, box) >= 0.90:
            contained.append(rival)
    separated = next(([first["id"], second["id"]]
                      for index, first in enumerate(contained)
                      for second in contained[index + 1:]
                      if overlap(first["xyxy"], second["xyxy"]) < 0.25), [])
    return similar, extents, separated


def sam_measurement(candidate, observation, width, height):
    sam_box = observation.get("sam_box")
    measurement = {"sam_status": "not_evaluated"}
    if usable_box(sam_box, width, height) and usable_box(candidate["xyxy"], width, height):
        measurement.update(sam_status="measured", sam_box=sam_box,
                           sam_box_iou=overlap(candidate["xyxy"], sam_box))
        score = observation.get("sam_score")
        if type(score) in (int, float) and isfinite(score):
            measurement["sam_score"] = score
    elif observation.get("sam_status") == "failed":
        measurement.update(sam_status="failed", sam_error=observation.get("sam_error", "mask_unavailable"))
    elapsed = observation.get("sam_seconds")
    if type(elapsed) in (int, float) and isfinite(elapsed) and elapsed >= 0:
        measurement["sam_seconds"] = elapsed
    return measurement


class SAMShadowEvidence:
    """Generate bounded extent evidence from the pinned, local SAM 2.1 model."""

    def __init__(self, model_dir, *, engine_factory=None):
        from prepare_model import FILES, REVISION, verified

        directory = Path(model_dir).resolve(strict=True)
        for name in FILES:
            path = directory / name
            if not path.resolve().is_relative_to(directory) or not verified(path):
                raise ValueError(f"SAM 2.1 本機模型雜湊不符：{name}")
        if engine_factory is None:
            from segmentation import SegmentationEngine

            engine_factory = SegmentationEngine
        self.engine = engine_factory(directory)
        self.provenance = {
            "model": "sam2.1-hiera-tiny",
            "revision": REVISION,
            "files": {name: digest for name, (_, digest) in FILES.items()},
            "max_boxes_per_frame": MAX_SAM_BOXES,
            "mode": "box_prompt_extent_evidence",
        }

    def __call__(self, image, frame):
        by_candidate = {row["id"]: row for row in frame["candidates"]}
        proposals = []
        for check in frame["shadow_qc"]["checks"]:
            row = by_candidate[check["candidate_id"]]
            selected = check["selected_by_existing_flow"]
            if not (selected or (check["qwen_extent_advisory"] and check["qwen_class"] == "target")):
                continue
            if not usable_box(row["xyxy"], image.width, image.height):
                continue
            extent_risk = bool(check.get("mm_extent_rivals") or check.get("mm_contained_pair"))
            priority = (not extent_risk, not check["qwen_extent_advisory"], not selected,
                        row["score"], row["id"])
            proposals.append((priority, row))

        evidence = {}
        for _, row in sorted(proposals, key=lambda item: item[0])[:MAX_SAM_BOXES]:
            started = perf_counter()
            try:
                result = self.engine.generate("sam2", image, box=tuple(row["xyxy"]),
                                              positive=[], negative=[], image_key=frame["image_sha256"])
                ys, xs = result.mask.nonzero()
                if not xs.size or result.mask.shape != (image.height, image.width):
                    raise RuntimeError("SAM mask geometry is invalid")
                observation = {"sam_box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                               "sam_score": result.score}
            except (RuntimeError, ValueError) as error:
                observation = {"sam_status": "failed", "sam_error": type(error).__name__}
            observation["sam_seconds"] = round(perf_counter() - started, 6)
            evidence[row["id"]] = observation
        return evidence

    def close(self):
        self.engine = None


def describe_frame_qc(frame, settings, *, width, height, labels, retry_state=None, evidence=None):
    """Return additive shadow routing evidence. The caller keeps live verdicts intact."""
    checks, flags = [], []
    evidence = evidence if isinstance(evidence, dict) else {}
    retry_state = retry_state or {}
    retries = frame["retries"]
    retry_count = len(retries)
    selected = {box["candidate_id"] for box in frame["boxes"]}
    proposals = [row for row in frame["candidates"] if usable_box(row["xyxy"], width, height)
                 and type(row.get("score")) in (int, float) and isfinite(row["score"])
                 and settings["minimum_score"] <= row["score"] <= 1]
    proposal_ids = {item["id"] for item in proposals}

    def flag(code, candidate=None):
        detail = {"code": code, "reason": REASONS[code]}
        if candidate:
            detail.update(candidate_id=candidate["id"], label=candidate["label"], xyxy=candidate.get("xyxy"))
        flags.append(detail)

    if retry_count:
        flag("retry_requires_review")
    if frame["exceptions"]:
        flag("legacy_review_pending")
    if not selected:
        flag("no_boxes")
    for candidate in frame["candidates"]:
        ident = candidate["id"]
        semantic = qwen_class_evidence(candidate, settings)
        is_selected = ident in selected
        check = {
            "candidate_id": ident,
            "label": candidate["label"],
            "selected_by_existing_flow": is_selected,
            "qwen_class": semantic,
            "qwen_extent_advisory": candidate.get("reason_code") == "bad_extent",
        }
        observation = evidence.get(ident)
        observation = observation if isinstance(observation, dict) else {}
        check.update(sam_measurement(candidate, observation, width, height))
        checks.append(check)
        if not is_selected:
            if (ident in proposal_ids
                    and candidate["score"] >= max(UNMATCHED_MM_SCORE, settings["minimum_score"])
                    and not any(box["label"] == candidate["label"] and
                                overlap(candidate["xyxy"], box["xyxy"]) >= RIVAL_IOU for box in frame["boxes"])):
                flag("unmatched_mm_proposal", candidate)
            continue
        if semantic != "target":
            flag("qwen_class_uncertain", candidate)
        if ident not in proposal_ids:
            flag("invalid_geometry", candidate)
            continue

        similar, extents, separated = mm_rivals(
            candidate, proposals, max(settings["minimum_score"], candidate["score"] * RIVAL_SCORE_RATIO)
        )
        check.update(mm_class_rivals=similar, mm_extent_rivals=extents, mm_contained_pair=separated)
        for name, value in (("mm_class_rival", similar), ("mm_extent_rival", extents),
                            ("possible_merged_instances", separated)):
            if value:
                flag(name, candidate)

        structure_label = observation.get("structure_label")
        check["structure_label"] = structure_label if isinstance(structure_label, str) and structure_label else None
        check["structure_evidence"] = (
            "not_evaluated" if check["structure_label"] is None else
            "agreement" if structure_label == candidate["label"] else "conflict"
        )
        if check["structure_evidence"] == "conflict":
            flag("structure_class_conflict", candidate)

        if check["sam_status"] == "measured" and check["sam_box_iou"] < SAM_MINIMUM_IOU:
            flag("sam_extent_conflict", candidate)
        elif check["sam_status"] == "failed":
            flag("sam_extent_uncertain", candidate)

    return {
        "policy_version": QC_POLICY_VERSION,
        "shadow_only": True,
        "disposition": "held" if not selected and not proposals else "review" if flags else "auto_candidate",
        "retry_count": retry_count,
        "retry_stop_reason": retry_state.get("stop_reason", "not_needed"),
        "retry_requests": retry_state.get("request_count", 0),
        "retry_seconds": round(sum(retry.get("seconds", 0.0) for retry in retries), 3),
        "flags": flags,
        "checks": checks,
        "evidence_coverage": {
            "mm_rival_scope": "configured_classes" if len(labels) > 1 else "single_class",
            "structure_boxes_checked": sum(check.get("structure_evidence") in {"agreement", "conflict"}
                                           for check in checks),
            "sam_boxes_checked": sum(check["sam_status"] == "measured" for check in checks),
            "sam_boxes_failed": sum(check["sam_status"] == "failed" for check in checks),
            "sam_seconds": round(sum(check.get("sam_seconds", 0.0) for check in checks), 6),
        },
    }
