from copy import deepcopy
import math
import unittest

from review_diagnosis import diagnose_reviewed_frame


TARGET = {"label": "shipping_container", "xyxy": [10, 10, 50, 50]}


def candidate(identifier="mm-1", **values):
    return {"id": identifier, "label": "shipping_container", "xyxy": [10, 10, 50, 50], **values}


class ReviewDiagnosisTests(unittest.TestCase):
    def diagnose(self, candidates, **options):
        return diagnose_reviewed_frame(reference_boxes=[TARGET], candidates=candidates, **options)

    def test_missing_geometry_requires_complete_pool(self):
        self.assertEqual(self.diagnose([])["targets"][0]["cause"], "candidate_miss")
        self.assertEqual(self.diagnose([], pool_complete=False)["targets"][0]["cause"], "unknown")

    def test_final_rejection_without_machine_participation_is_unknown(self):
        row = self.diagnose([candidate(verdict="rejected")])["targets"][0]
        self.assertEqual(row["cause"], "unknown")
        self.assertIsNone(row["candidate_matches"][0]["machine_reviewed"])

    def test_human_unseen_or_seen_does_not_imply_ranking(self):
        for exposed in (None, [], ["mm-1"]):
            with self.subTest(exposed=exposed):
                self.assertEqual(self.diagnose([candidate()], exposed_candidate_ids=exposed)["targets"][0]["cause"],
                                 "unknown")

    def test_definitive_machine_shortlist_exclusion_is_ranking(self):
        result = self.diagnose([candidate()], reviewed_candidate_ids=[])
        self.assertEqual(result["targets"][0]["cause"], "ranking")

    def test_explicit_human_display_budget_exclusion_is_ranking(self):
        result = self.diagnose([candidate()], display_budget_excluded_candidate_ids=["mm-1"])
        self.assertEqual(result["targets"][0]["cause"], "ranking")

    def test_machine_decision_takes_priority_over_other_unseen_candidate(self):
        result = self.diagnose([candidate(verdict="rejected"), candidate("mm-2")], reviewed_candidate_ids=["mm-1"])
        self.assertEqual(result["targets"][0]["cause"], "qwen_decision")

    def test_explicit_qwen_hold_records_machine_participation(self):
        row = self.diagnose([candidate(qwen_verdict="unresolved")])["targets"][0]
        self.assertEqual(row["cause"], "qwen_decision")
        self.assertTrue(row["candidate_matches"][0]["machine_reviewed"])

    def test_other_prompt_class_preserves_geometric_coverage(self):
        result = self.diagnose([candidate(label="telescopic_reach_stacker")])
        self.assertEqual(result["targets"][0]["cause"], "unknown")
        self.assertEqual(len(result["targets"][0]["candidate_matches"]), 1)
        self.assertEqual(self.diagnose([candidate(label="telescopic_reach_stacker")],
                                      reviewed_candidate_ids=["mm-1"])["targets"][0]["cause"], "unknown")

    def test_accepted_class_and_geometry_must_both_match(self):
        result = self.diagnose([candidate(verdict="accepted", qwen_label="shipping_container")])
        self.assertEqual(result["summary"]["matched_targets"], 1)
        self.assertIsNone(result["targets"][0]["cause"])
        wrong = self.diagnose([candidate(verdict="accepted", qwen_label="telescopic_reach_stacker")],
                              reviewed_candidate_ids=["mm-1"])
        self.assertEqual(wrong["summary"]["missed_targets"], 1)
        self.assertEqual(wrong["targets"][0]["cause"], "qwen_decision")

    def test_separate_accepted_output_and_one_to_one_matching(self):
        result = diagnose_reviewed_frame(reference_boxes=[TARGET, TARGET], candidates=[],
                                        accepted_boxes=[candidate("final-1")])
        self.assertEqual(result["summary"]["matched_targets"], 1)
        self.assertEqual(result["summary"]["missed_targets"], 1)

    def test_inputs_are_not_mutated(self):
        original = [candidate(verdict="rejected", qwen_votes=[{"winner": "N"}])]
        snapshot = deepcopy(original)
        self.diagnose(original)
        self.assertEqual(original, snapshot)

    def test_invalid_coordinates_ids_and_conflicting_exposure_are_rejected(self):
        for box in ([10, 10, 10, 50], [-1, 10, 50, 50], [10, 10, math.nan, 50]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.diagnose([candidate(xyxy=box)])
        with self.assertRaises(ValueError):
            self.diagnose([candidate(), candidate()])
        with self.assertRaises(ValueError):
            self.diagnose([candidate()], exposed_candidate_ids=["mm-1"],
                          display_budget_excluded_candidate_ids=["mm-1"])


if __name__ == "__main__":
    unittest.main()
