"""Explain reviewed omissions from recorded candidate evidence, without running models.

Machine review participation, human display exposure and final output are separate
facts. A final ``rejected`` verdict alone does not establish a Qwen decision.
"""

from collections import Counter
from copy import deepcopy
from math import isfinite

from generic_eval import box_iou as iou


SCHEMA = "annotation_wizard.review_diagnosis.v1"
CAUSES = ("candidate_miss", "ranking", "qwen_decision", "unknown")


def _match(references, predictions, threshold):
    """Find a maximum-cardinality pairing, preferring larger IoU per reference."""
    overlaps = [[iou(reference, prediction) for prediction in predictions] for reference in references]
    choices = [sorted((index for index, value in enumerate(row) if value >= threshold),
                      key=lambda index: (-row[index], index)) for row in overlaps]
    owners = {}

    def augment(reference, visited):
        for prediction in choices[reference]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in owners or augment(owners[prediction], visited):
                owners[prediction] = reference
                return True
        return False

    for reference in sorted(range(len(choices)), key=lambda index: (len(choices[index]), index)):
        augment(reference, set())
    return sorted((reference, prediction, overlaps[reference][prediction])
                  for prediction, reference in owners.items())


def _boxes(rows, kind):
    result, identifiers = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Each {kind} must be a dictionary.")
        values = row.get("xyxy")
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(f"Each {kind} needs four original-pixel xyxy coordinates.")
        if any(type(value) not in (int, float) or not isfinite(value) for value in values):
            raise ValueError(f"Each {kind} must have finite numeric coordinates.")
        left, top, right, bottom = values
        if not (0 <= left < right and 0 <= top < bottom):
            raise ValueError(f"Each {kind} needs a positive box within the nonnegative image plane.")
        identifier = row.get("id", row.get("candidate_id", f"{kind}-{index}"))
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"Each {kind} needs a unique nonempty identifier when provided.")
        identifiers.add(identifier)
        label = row.get("qwen_label") or row.get("label")
        if kind in {"reference", "accepted"} and (not isinstance(label, str) or not label):
            raise ValueError(f"Each {kind} needs a nonempty label.")
        result.append({**deepcopy(row), "id": identifier, "xyxy": list(values), "model_label": label})
    return result


def _ids(values, name):
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a collection of candidate identifiers.")
    result = set(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain nonempty candidate identifiers.")
    return result


def _candidate_evidence(candidate, overlap, reviewed, exposed, budget_excluded):
    identifier = candidate["id"]
    qwen_verdict = candidate.get("qwen_verdict")
    direct_qwen = qwen_verdict in {"accepted", "rejected", "unresolved"} or bool(candidate.get("qwen_votes"))
    machine_reviewed = identifier in reviewed if reviewed is not None else None
    if direct_qwen:
        machine_reviewed = True
    return {
        "candidate_id": identifier,
        "iou": round(overlap, 6),
        "mm_label": candidate.get("label", candidate.get("prompt_class")),
        "model_label": candidate["model_label"],
        "reviewer_label": candidate.get("qwen_label"),
        "verdict": candidate.get("verdict"),
        "qwen_verdict": qwen_verdict,
        "decision_code": candidate.get("decision_code"),
        "machine_reviewed": machine_reviewed,
        "human_exposed": identifier in exposed if exposed is not None else None,
        "display_budget_excluded": identifier in budget_excluded if budget_excluded is not None else False,
    }


def _cause(candidate_matches, target_label, pool_complete):
    if not candidate_matches:
        if pool_complete:
            return "candidate_miss", ["No recorded MM box reaches the matching IoU, across all prompt classes."]
        return "unknown", ["No compatible MM box is recorded, but the candidate pool is incomplete."]

    # A box reviewed by the machine is stronger evidence than another unused box.
    decisions = [row for row in candidate_matches if row["machine_reviewed"] is True and (
        row["qwen_verdict"] in {"rejected", "unresolved"}
        or row["verdict"] in {"rejected", "unresolved", "held", "hold"}
        or (row["reviewer_label"] is not None and row["reviewer_label"] != target_label)
        or (row["verdict"] == "accepted" and row["model_label"] != target_label)
    )]
    if decisions:
        return "qwen_decision", [
            "A compatible box participated in machine review and was rejected, held or assigned another class."
        ]
    if any(row["machine_reviewed"] is False for row in candidate_matches):
        return "ranking", ["A compatible MM box is explicitly absent from the complete machine-review shortlist."]
    if any(row["display_budget_excluded"] for row in candidate_matches):
        return "ranking", ["A compatible MM box has explicit evidence of exclusion from the human display budget."]
    return "unknown", [
        "A compatible MM box exists, but the records do not establish why the final target is missing.",
        "Human display exposure or skipping alone does not identify a model failure."
    ]


def diagnose_reviewed_frame(
    *, reference_boxes, candidates, accepted_boxes=None, reviewed_candidate_ids=None,
    exposed_candidate_ids=None, display_budget_excluded_candidate_ids=None,
    pool_complete=True, minimum_iou=0.5,
):
    """Return one diagnosis per final reviewed target from passed dictionaries.

    ``candidates`` is the recorded MM pool before shortlisting. ``pool_complete``
    states whether that pool is complete for this frame and run. Its default
    preserves a candidate-miss diagnosis when the caller has all generated boxes.
    Candidates from any prompt class can supply geometric coverage.

    ``reviewed_candidate_ids`` is a *complete* set actually evaluated by the
    machine reviewer, or None when participation is unknown. It is distinct from
    ``exposed_candidate_ids``, the human-visible overlay set. To identify a human
    display-budget failure, pass explicit ``display_budget_excluded_candidate_ids``.

    Accepted output is derived from candidate ``verdict == 'accepted'`` unless
    supplied separately. Accepted matches are same-class and one-to-one. These
    helpers trust caller-established whole-frame review and source provenance.
    """
    if type(minimum_iou) not in (int, float) or not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0, 1].")
    if type(pool_complete) is not bool:
        raise ValueError("pool_complete must be a boolean.")
    references = _boxes(reference_boxes, "reference")
    pool = _boxes(candidates, "candidate")
    outputs = _boxes(
        [row for row in candidates if row.get("verdict") == "accepted"] if accepted_boxes is None else accepted_boxes,
        "accepted",
    )
    reviewed = _ids(reviewed_candidate_ids, "reviewed_candidate_ids")
    exposed = _ids(exposed_candidate_ids, "exposed_candidate_ids")
    budget_excluded = _ids(display_budget_excluded_candidate_ids, "display_budget_excluded_candidate_ids")
    if exposed is not None and budget_excluded is not None and exposed & budget_excluded:
        raise ValueError("A candidate cannot be both human-exposed and excluded from the display budget.")

    matches = {}
    labels = dict.fromkeys(row["model_label"] for row in references)
    for label in labels:
        reference_indices = [index for index, row in enumerate(references) if row["model_label"] == label]
        output_indices = [index for index, row in enumerate(outputs) if row["model_label"] == label]
        pairs = _match([references[index]["xyxy"] for index in reference_indices],
                       [outputs[index]["xyxy"] for index in output_indices], minimum_iou)
        for ref_index, output_index, overlap in pairs:
            matches[reference_indices[ref_index]] = (outputs[output_indices[output_index]]["id"], overlap)

    targets, counts = [], Counter()
    for index, reference in enumerate(references):
        candidate_matches = []
        for candidate in pool:
            overlap = iou(reference["xyxy"], candidate["xyxy"])
            if overlap >= minimum_iou:
                candidate_matches.append(_candidate_evidence(candidate, overlap, reviewed, exposed, budget_excluded))
        candidate_matches.sort(key=lambda row: (-row["iou"], row["candidate_id"]))
        matched = matches.get(index)
        if matched:
            cause, evidence = None, ["A final accepted box matches this reviewed target with the same class."]
        else:
            cause, evidence = _cause(candidate_matches, reference["model_label"], pool_complete)
            counts[cause] += 1
        targets.append({
            "reference_index": index,
            "label": reference["model_label"],
            "xyxy": reference["xyxy"],
            "status": "matched" if matched else "missed",
            "cause": cause,
            "evidence": evidence,
            "matching_accepted_id": matched[0] if matched else None,
            "matching_iou": round(matched[1], 6) if matched else None,
            "candidate_matches": candidate_matches,
        })
    return {
        "schema": SCHEMA,
        "minimum_iou": minimum_iou,
        "pool_complete": pool_complete,
        "summary": {"reviewed_targets": len(references), "matched_targets": len(matches),
                    "missed_targets": len(references) - len(matches), **{key: counts[key] for key in CAUSES}},
        "targets": targets,
    }
