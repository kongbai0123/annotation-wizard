"""Check evidence composition and original-decision provenance without loading models."""

from copy import deepcopy
import unittest

from generic_confidence import candidate_confidence, frame_confidence
from generic_inference import finalize_frame


def candidate(ident, *, mm=.60, verdict="accepted", first=(.91, .72), second=(.83, .90)):
    return {"id": ident, "label": "banjo", "score": mm, "verdict": verdict,
            "reason_code": verdict, "xyxy": [0, 0, 24, 20],
            "votes": [{"probabilities": {"A": support}, "winner": "A", "allowed_token_mass": mass}
                      for support, mass in (first, second)]}


class GenericConfidenceTests(unittest.TestCase):
    def test_composite_uses_both_orders_and_every_accepted_box(self):
        first = candidate("first", mm=.72, first=(.90, .80), second=(.85, .85))
        second = candidate("second", mm=.62, first=(.80, .80), second=(.92, .60))
        rejected = candidate("rejected", verdict="rejected", mm=.12)
        frame = {"candidates": [first, second, rejected], "status": "ready", "exceptions": []}
        before = deepcopy(frame)

        score = frame_confidence(frame)

        self.assertEqual(score["status"], "ready")
        self.assertEqual((score["accepted_count"], score["candidate_count"]), (2, 3))
        self.assertEqual(score["mm_score"], .62)
        self.assertAlmostEqual(score["qwen_score"], .552)
        self.assertAlmostEqual(score["score"], .552)
        self.assertEqual(frame, before)

    def test_pending_and_manual_edits_preserve_original_model_evidence(self):
        frame = {"candidates": [candidate("accepted"), candidate("rejected", verdict="rejected")],
                 "status": "review", "boxes": [{"candidate_id": "accepted"}],
                 "exceptions": [{"code": "unmatched_scene_target"}, {"code": "unmatched_scene_target"}]}
        pending = frame_confidence(frame)
        submitted = frame_confidence({**frame, "status": "approved", "boxes": [{"label": "harp"}]})
        self.assertEqual(pending["status"], "review")
        self.assertEqual(pending["reasons"], ["unmatched_scene_target"])
        self.assertEqual(pending, submitted)

    def test_missing_or_invalid_evidence_is_unavailable_rather_than_zero(self):
        missing = candidate("missing", mm=float("nan"))
        missing["votes"] = missing["votes"][:1]
        no_score = frame_confidence({"candidates": [missing]})
        self.assertIsNone(no_score["score"])
        self.assertIsNone(no_score["mm_score"])
        self.assertIsNone(no_score["qwen_score"])
        self.assertIn("missing_qwen_evidence", no_score["reasons"])
        unopened = frame_confidence({"status": "awaiting_inference", "candidates": []})
        self.assertIsNone(unopened["score"])
        self.assertEqual(unopened["status"], "unavailable")
        self.assertEqual(unopened["reasons"], ["inference_not_run"])

    def test_completed_empty_is_a_zero_without_fake_component_scores(self):
        empty = frame_confidence({"candidates": [candidate("rejected", verdict="rejected")],
                                  "status": "review"})
        self.assertEqual(empty["status"], "no_accepted")
        self.assertEqual(empty["score"], 0.)
        self.assertIsNone(empty["mm_score"])
        self.assertIsNone(empty["qwen_score"])
        self.assertEqual(empty["reasons"], ["no_accepted_boxes"])

    def test_candidate_support_reports_acceptance_even_when_rejected(self):
        rejected = candidate("rejected", verdict="rejected", first=(.1, .8), second=(.2, .5))
        rejected["votes"][0]["winner"] = "C"
        score = candidate_confidence(rejected)
        self.assertEqual(score["mm_score"], .6)
        self.assertAlmostEqual(score["qwen_score"], .08)

    def test_finalized_frame_persists_the_same_score_as_legacy_derivation(self):
        selected = candidate("kept")
        row = {"id": "first", "image_sha256": "0" * 64, "candidates": [selected], "retries": [],
               "scene_scan": {"status": "parsed", "targets": [{"label": "banjo", "xyxy": selected["xyxy"]}]}}
        result = finalize_frame(row)
        score = result.pop("confidence")
        self.assertEqual(score, frame_confidence(result))
        self.assertEqual(score["score"], .6)
        self.assertEqual(result["status"], "ready")


if __name__ == "__main__":
    unittest.main()
