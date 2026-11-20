import unittest

from generic_results_view import decision_details, matching_rows, percent


class GenericResultPresentationTests(unittest.TestCase):
    def test_queue_reason_separates_occurrences_and_distinct_images(self):
        exception = {"code": "uncertain_review"}
        output = {"frames": [
            {"id": "reviewed", "boxes": [], "exceptions": [exception, exception],
             "candidates": [{"verdict": "unresolved"}], "retries": [{}]},
            {"id": "pending", "boxes": [], "exceptions": [exception], "candidates": [], "retries": []},
            {"id": "ready", "boxes": [{"label": "hinge"}],
             "candidates": [{"verdict": "accepted"}, {"verdict": "rejected"}], "retries": []},
        ]}
        facts = decision_details(output, {"reviewed"})
        self.assertEqual(facts["initial_review"], 2)
        self.assertEqual(facts["reasons"], [("Qwen 兩次判讀證據不足", 2, 3, 1)])
        self.assertEqual(facts["candidate_frames"], 2)
        self.assertEqual(facts["candidates"]["accepted"], 1)
        self.assertEqual(facts["original_boxes"]["hinge"], 1)
        self.assertEqual(facts["retries"], 1)

    def test_missing_candidate_record_does_not_become_zero_coverage(self):
        facts = decision_details({"frames": [{"id": "legacy", "boxes": [], "exceptions": []}]}, set())
        self.assertIsNone(facts["candidate_frames"])
        self.assertIsNone(facts["candidates"])
        self.assertEqual(facts["reasons"], [("空畫格需確認", 1, 1, 1)])

    def test_undefined_precision_stays_distinct_from_zero_recall(self):
        outcome = {"by_class": {"hinge": {"tp": 0, "fp": 0, "fn": 2, "precision": None, "recall": 0.0}}}
        self.assertEqual(matching_rows(outcome), [("hinge", 0, 0, 2, "—", "0.0%")])
        self.assertEqual(percent(.062), "6.2%")


if __name__ == "__main__":
    unittest.main()
