"""Score sealed detection predictions against independent bounding-box references."""

from math import isfinite


AP_THRESHOLDS = tuple(round(0.50 + index * 0.05, 2) for index in range(10))


def box_iou(first, second):
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def validate_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(v) in (int, float) and isfinite(v) for v in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3])


def prepared_frames(frames, labels):
    if (not isinstance(labels, (list, tuple)) or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels) or "background" in labels):
        raise ValueError("Evaluation requires unique, nonempty labels without the reserved background class.")
    known = set(labels)
    seen, prepared = set(), []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("Every evaluation frame must be an object.")
        identifier = frame.get("id", str(index))
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Evaluation frame identifiers must be unique nonempty strings.")
        seen.add(identifier)
        record = {"id": identifier}
        for key in ("reference", "predictions"):
            rows = frame.get(key)
            if not isinstance(rows, list):
                raise ValueError(f"Frame {identifier} requires a {key} list.")
            for row in rows:
                if (not isinstance(row, dict) or row.get("label") not in known
                        or not validate_box(row.get("xyxy"))):
                    raise ValueError(f"Frame {identifier} contains an unknown class or invalid box.")
                if key == "predictions" and (type(row.get("score")) not in (int, float)
                        or not isfinite(row["score"]) or not 0 <= row["score"] <= 1):
                    raise ValueError(f"Frame {identifier} contains an invalid prediction score.")
            record[key] = rows
        prepared.append(record)
    return prepared


def ranked_matches(frames, label, threshold, minimum_score=0):
    """Match each prediction to the best unused same-class reference in its frame."""
    references = [[row["xyxy"] for row in frame["reference"] if row["label"] == label] for frame in frames]
    predictions = [(row["score"], frame_index, prediction_index, row["xyxy"])
                   for frame_index, frame in enumerate(frames)
                   for prediction_index, row in enumerate(frame["predictions"])
                   if row["label"] == label and row["score"] >= minimum_score]
    predictions.sort(key=lambda item: (-item[0], frames[item[1]]["id"], item[2]))
    used = [set() for _ in frames]
    outcomes = []
    for score, frame_index, _, box in predictions:
        candidates = [(box_iou(box, truth), truth_index) for truth_index, truth in enumerate(references[frame_index])
                      if truth_index not in used[frame_index]]
        best_iou, truth_index = max(candidates, key=lambda item: (item[0], -item[1]), default=(0.0, -1))
        hit = best_iou >= threshold
        if hit:
            used[frame_index].add(truth_index)
        outcomes.append((score, hit))
    return sum(map(len, references)), outcomes


def ap101(truth_count, outcomes):
    """Mean interpolated precision at recall 0.00, 0.01, ..., 1.00."""
    if truth_count == 0:
        return None
    if not outcomes:
        return 0.0
    curve = []
    positives = 0
    for rank, (_, hit) in enumerate(outcomes, 1):
        positives += int(hit)
        curve.append((positives / truth_count, positives / rank))
    # Precision at a recall level is the greatest precision achieved at that or
    # a higher recall. Unreached recall levels contribute zero.
    return sum(max((precision for recall, precision in curve if recall >= step / 100), default=0.0)
               for step in range(101)) / 101


def confusion(frames, labels, threshold, confidence):
    """Associate by descending geometric IoU, regardless of class, once per box."""
    positions = {label: index for index, label in enumerate(labels)}
    background = len(labels)
    matrix = [[0] * (background + 1) for _ in range(background + 1)]
    for frame in frames:
        references = frame["reference"]
        predictions = [row for row in frame["predictions"] if row["score"] >= confidence]
        edges = [(box_iou(prediction["xyxy"], reference["xyxy"]), pi, ri)
                 for pi, prediction in enumerate(predictions) for ri, reference in enumerate(references)]
        edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
        used_predictions, used_references = set(), set()
        for iou, pi, ri in edges:
            if iou < threshold:
                break
            if pi in used_predictions or ri in used_references:
                continue
            used_predictions.add(pi)
            used_references.add(ri)
            matrix[positions[predictions[pi]["label"]]][positions[references[ri]["label"]]] += 1
        for pi, row in enumerate(predictions):
            if pi not in used_predictions:
                matrix[positions[row["label"]]][background] += 1
        for ri, row in enumerate(references):
            if ri not in used_references:
                matrix[background][positions[row["label"]]] += 1
    return matrix


def threshold_counts(frames, labels, threshold, confidence):
    classes = {}
    for label in labels:
        count, outcomes = ranked_matches(frames, label, threshold, confidence)
        tp = sum(hit for _, hit in outcomes)
        fp, fn = len(outcomes) - tp, count - tp
        classes[label] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / count if count else None}
    total = {key: sum(row[key] for row in classes.values()) for key in ("tp", "fp", "fn")}
    return {"by_class": classes, **total,
            "precision": total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else None,
            "recall": total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else None,
            "false_positives_per_image": total["fp"] / len(frames) if frames else None,
            "matrix": confusion(frames, labels, threshold, confidence), "labels": [*labels, "background"]}


def score_detection(frames, labels, *, confidence=0.25):
    """Calculate AP on all prediction scores and P/R at the given confidence.

    Classes without reference instances have undefined AP and are excluded from
    macro mAP. Their predictions still count as false positives at fixed score
    thresholds. Confusion matrix rows are predictions and columns are references.
    """
    if type(confidence) not in (int, float) or not isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be a finite value between zero and one.")
    prepared = prepared_frames(frames, labels)
    by_class = {}
    for label in labels:
        averages = {}
        for threshold in AP_THRESHOLDS:
            truth_count, outcomes = ranked_matches(prepared, label, threshold)
            averages[f"{threshold:.2f}"] = ap101(truth_count, outcomes)
        valid = [ap for ap in averages.values() if ap is not None]
        by_class[label] = {"truth": sum(row["label"] == label for frame in prepared for row in frame["reference"]),
                           "predictions": sum(row["label"] == label for frame in prepared for row in frame["predictions"]),
                           "ap50": averages["0.50"], "ap50_95": sum(valid) / len(valid) if valid else None,
                           "ap_by_iou": averages}
    ap50 = [row["ap50"] for row in by_class.values() if row["ap50"] is not None]
    averaged = [row["ap50_95"] for row in by_class.values() if row["ap50_95"] is not None]
    return {"images": len(prepared), "labels": list(labels), "confidence": confidence,
            "map50": sum(ap50) / len(ap50) if ap50 else None,
            "map50_95": sum(averaged) / len(averaged) if averaged else None,
            "by_class": by_class,
            "thresholds": {f"{threshold:.2f}": threshold_counts(prepared, labels, threshold, confidence)
                           for threshold in (0.50, 0.75)},
            "protocol": {"ap": "101-point interpolated precision, all supplied scores, IoU 0.50:0.05:0.95",
                         "precision_recall": "same-class score-ordered one-to-one matching",
                         "confusion_matrix": "class-agnostic descending-IoU association, predicted rows and reference columns",
                         "classes_without_references": "AP undefined and excluded from mAP; false positives retained"}}
