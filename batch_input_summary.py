"""Build the current batch input view without modifying saved annotations."""

from collections import Counter
from collections.abc import Mapping
from math import isfinite

from generic_job import _original_xyxy, _saved_by_operator


def _numeric_coordinates(values):
    return (isinstance(values, (list, tuple)) and len(values) == 4
            and all(type(value) in (int, float) and isfinite(value) for value in values))


def _valid_box(box):
    if not isinstance(box, dict) or not isinstance(box.get("label"), str) or not box["label"].strip():
        return False
    coordinates = box.get("xyxy")
    if not _numeric_coordinates(coordinates):
        return False
    x1, y1, x2, y2 = coordinates
    return 0 <= x1 < x2 and 0 <= y1 < y2


def _stored_boxes(asset):
    for annotation in asset.get("annotations", ()):
        if (not isinstance(annotation, dict) or annotation.get("type") != "bbox"
                or not _numeric_coordinates(annotation.get("bbox"))):
            continue
        label, xyxy = annotation.get("label"), _original_xyxy(annotation)
        box = {"label": label, "xyxy": xyxy, "source": (
            "manual" if _saved_by_operator(annotation) else annotation.get("source") or "unknown")}
        if _valid_box(box) and xyxy[2] <= asset["width"] and xyxy[3] <= asset["height"]:
            yield box


def build_input_rows(assets, job_rows, job_labels):
    """Return effective rows in current input order with box totals across the visible classes."""
    by_id = job_rows if isinstance(job_rows, Mapping) else {row["id"]: row for row in job_rows}
    labels = set(job_labels)
    rows = []
    for asset in assets:
        identifier = asset["id"]
        if (effective := by_id.get(identifier)) is not None:
            boxes = list(effective.get("boxes", ()))
            if effective.get("status") == "independent_test":
                boxes.clear()
            else:
                boxes.extend(box for box in _stored_boxes(asset) if box["label"] not in labels)
            rows.append({**effective, "summary_boxes": boxes})
            continue
        boxes = list(_stored_boxes(asset))
        manual = sum(box["source"] == "manual" for box in boxes)
        rows.append({**asset, "status": "awaiting_inference", "boxes": boxes, "confidence": None,
                     "manual_box_count": manual, "ai_box_count": len(boxes) - manual, "summary_boxes": boxes})
    return rows


def input_class_counts(rows):
    """Count boxes in the input scope, including pending decisions and excluding independent tests."""
    return Counter(box["label"] for row in rows if row["status"] != "independent_test"
                   for box in row.get("summary_boxes", ()) if _valid_box(box))
