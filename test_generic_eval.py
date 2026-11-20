import unittest

from generic_eval import score_detection


def truth(label="tool", box=(0, 0, 10, 10)):
    return {"label": label, "xyxy": list(box)}


def prediction(label="tool", box=(0, 0, 10, 10), score=0.9):
    return {**truth(label, box), "score": score}


class GenericEvaluationTests(unittest.TestCase):
    def test_perfect_boxes_have_perfect_ap_and_confusion(self):
        frames = [{"id": "first", "reference": [truth()], "predictions": [prediction()]},
                  {"id": "second", "reference": [truth("bolt")], "predictions": [prediction("bolt")]}]
        result = score_detection(frames, ["tool", "bolt"])
        self.assertEqual(result["map50"], 1)
        self.assertEqual(result["map50_95"], 1)
        self.assertEqual(result["thresholds"]["0.75"]["matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    def test_duplicate_boxes_and_negatives_increase_false_positives(self):
        frames = [{"id": "positive", "reference": [truth()],
                   "predictions": [prediction(score=.8), prediction(score=.7)]},
                  {"id": "negative", "reference": [], "predictions": [prediction(score=.9)]}]
        result = score_detection(frames, ["tool"])
        stats = result["thresholds"]["0.50"]
        self.assertEqual((stats["tp"], stats["fp"], stats["fn"]), (1, 2, 0))
        self.assertEqual(stats["false_positives_per_image"], 1)
        self.assertAlmostEqual(result["map50"], .5)
        self.assertEqual(stats["matrix"], [[1, 2], [0, 0]])

    def test_wrong_class_is_off_diagonal_and_counts_fp_fn(self):
        result = score_detection([{"id": "x", "reference": [truth()],
                                   "predictions": [prediction("bolt")]}], ["tool", "bolt"])
        stats = result["thresholds"]["0.50"]
        self.assertEqual(stats["matrix"], [[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(stats["by_class"]["tool"]["fn"], 1)
        self.assertEqual(stats["by_class"]["bolt"]["fp"], 1)
        self.assertIsNone(result["by_class"]["bolt"]["ap50"])
        self.assertEqual(result["map50"], 0)

    def test_ap_includes_low_scores_but_threshold_counts_filter_them(self):
        result = score_detection([{"id": "x", "reference": [truth()],
                                   "predictions": [prediction(score=.1)]}], ["tool"])
        self.assertEqual(result["map50"], 1)
        self.assertEqual(result["thresholds"]["0.50"]["by_class"]["tool"]["fn"], 1)
        self.assertEqual(result["thresholds"]["0.50"]["matrix"], [[0, 0], [1, 0]])

    def test_box_quality_changes_ap_and_stricter_threshold_counts(self):
        result = score_detection([{"id": "x", "reference": [truth()],
                                   "predictions": [prediction(box=(0, 0, 7, 10))]}], ["tool"])
        self.assertEqual(result["map50"], 1)
        self.assertAlmostEqual(result["map50_95"], .5)
        self.assertEqual(result["thresholds"]["0.50"]["tp"], 1)
        self.assertEqual(result["thresholds"]["0.75"]["tp"], 0)

    def test_one_to_one_does_not_reuse_reference_and_empty_sets_remain_undefined(self):
        result = score_detection([{"id": "x", "reference": [truth(), truth(box=(1, 0, 11, 10))],
                                   "predictions": [prediction()]}], ["tool"])
        self.assertEqual(result["thresholds"]["0.50"]["tp"], 1)
        self.assertEqual(result["thresholds"]["0.50"]["fn"], 1)
        self.assertAlmostEqual(result["map50"], 51 / 101)
        empty = score_detection([], ["tool"])
        self.assertIsNone(empty["map50"])
        self.assertIsNone(empty["thresholds"]["0.50"]["false_positives_per_image"])

    def test_invalid_reference_and_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            score_detection([{"id": "x", "reference": [truth(box=(0, 0, 0, 1))], "predictions": []}], ["tool"])
        with self.assertRaises(ValueError):
            score_detection([{"id": "x", "reference": [], "predictions": []}] * 2, ["tool"])


if __name__ == "__main__":
    unittest.main()
