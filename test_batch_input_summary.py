from collections import Counter
from unittest import TestCase

from batch_input_summary import build_input_rows, input_class_counts


def _asset(identifier, *annotations):
    return {"id": identifier, "name": identifier + ".png", "width": 100, "height": 50,
            "annotations": annotations}


def _annotation(label, source="manual", bbox=(4, 5, 15, 20), **extra):
    return {"type": "bbox", "label": label, "bbox": bbox, "source": source, **extra}


class BatchInputSummaryTests(TestCase):
    def test_preferred_job_boxes_replace_in_scope_stored_boxes(self):
        asset = _asset("first", _annotation("bolt"), _annotation("wood"), _annotation("bolt"))
        model_box = {"label": "bolt", "xyxy": [10, 12, 30, 36], "source": "mm_qwen"}
        row = {"id": "first", "status": "pending", "boxes": [model_box], "confidence": 0.2,
               "manual_box_count": 0, "ai_box_count": 1}

        rows = build_input_rows([asset], [row], ["bolt"])

        self.assertEqual(rows[0]["boxes"], [model_box])
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(rows[0]["confidence"], 0.2)
        self.assertNotIn("summary_boxes", row)
        self.assertEqual(input_class_counts(rows), Counter(bolt=1, wood=1))

    def test_saved_boxes_keep_origin_without_claiming_ai_inference(self):
        asset = _asset(
            "second",
            _annotation("bolt"),
            _annotation("bolt", "video_auto"),
            _annotation("nut", "grounding_dino", suggestion={"judge": "qwen"}),
            _annotation("nut", "grounding_dino", suggestion={"judge": "qwen"}, human_touched=True),
            _annotation("nut", "grounding_dino"),
            _annotation("invalid", bbox=(4, 5, -10, 20)),
            _annotation("outside", bbox=(90, 5, 15, 20)),
        )

        row, = build_input_rows([asset], {}, [])

        self.assertEqual(row["status"], "awaiting_inference")
        self.assertIsNone(row["confidence"])
        self.assertEqual(row["boxes"][0]["xyxy"], [4, 5, 19, 25])
        self.assertEqual(row["manual_box_count"], 3)
        self.assertEqual(row["ai_box_count"], 2)
        self.assertEqual(input_class_counts([row]), Counter(bolt=2, nut=3))

    def test_scope_order_and_independent_rows(self):
        first, second = _asset("first", _annotation("bolt")), _asset("second", _annotation("wood"))
        job_rows = {"first": {"id": "first", "status": "independent_test",
                              "boxes": [{"label": "nut", "xyxy": [0, 0, 10, 10]}]},
                    "other": {"id": "other", "status": "ai_ready",
                              "boxes": [{"label": "nut", "xyxy": [0, 0, 10, 10]}]}}

        rows = build_input_rows([second, first], job_rows, ["nut"])

        self.assertEqual([row["id"] for row in rows], ["second", "first"])
        self.assertEqual(input_class_counts(rows), Counter(wood=1))
