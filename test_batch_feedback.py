"""Exercise evidence, shadow comparison and next-batch release without models."""

from copy import deepcopy
from hashlib import sha256
import unittest

from batch_feedback import (batch_snapshot, compare_release, new_registry, prepare_release, promote_release,
                            rollback_binding, rollback_release, summarize_feedback)


TARGET = {"kind": "candidate_miss", "label": "shipping_container"}
EVIDENCE_POLICY = {"min_frames": 2, "min_sources": 2}
QUALITY_POLICY = {"max_false_release_rate": 0.20, "min_auto_candidates": 4, "max_additional_reviews": 0}


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def event(identifier, source, image, *, role="operator", diagnosis=None):
    return {"review_id": identifier, "role": role, "source_sha256": digest(source), "image_sha256": digest(image),
            "diagnoses": [TARGET.copy() if diagnosis is None else diagnosis]}


def evidence():
    return [event("one", "source-a", "image-a"), event("two", "source-b", "image-b", role="audit")]


def setup_comparison():
    registry = new_registry({"retry": 1})
    proposal = prepare_release(registry, {"retry": 2}, TARGET, evidence(), EVIDENCE_POLICY)
    inputs = {"frames": [
        {"role": "independent_evaluation", "source_sha256": digest(f"eval-source-{index % 2}"),
         "image_sha256": digest(f"eval-image-{index}"), "human_reference_sha256": digest(f"truth-{index}")}
        for index in range(6)],
        "budget": {"rechecks": 2}, "candidate_pool_sha256": digest("fixed candidate pool")}
    before = {"snapshot_sha256": registry["active_sha256"], "inputs": deepcopy(inputs),
              "metrics": {"auto_candidates": 4, "false_releases": 2, "manual_review": 2, "held": 0,
                          "elapsed_seconds": 20}}
    after = {"snapshot_sha256": proposal["candidate"]["sha256"], "inputs": deepcopy(inputs),
             "metrics": {"auto_candidates": 5, "false_releases": 1, "manual_review": 1, "held": 0,
                         "elapsed_seconds": 22}}
    return registry, proposal, before, after


def approval(binding, actor="驗收人"):
    return {"human": True, "actor": actor, "binding_sha256": binding}


class BatchFeedbackTests(unittest.TestCase):
    def test_deduplicate_frames_scope_and_distinct_sources(self):
        rows = evidence()
        rows.append(event("three", "source-a", "image-a", role="audit"))
        rows[-1]["diagnoses"].append(TARGET.copy())
        rows += [event("four", "source-c", "image-c", role="demonstration"),
                 event("five", "source-c", "image-d", role="independent_evaluation"),
                 event("six", "source-c", "image-e", diagnosis={"kind": "ranking", "label": TARGET["label"]}),
                 event("seven", "source-d", "image-f", diagnosis={"kind": "candidate_miss", "label": "other"})]
        rows.append({**event("eight", "source-e", "image-g"), "whole_frame_human_review": False})
        rows.append({**event("nine", "source-f", "image-h"), "data_role": "acceptance"})
        result = summarize_feedback(rows + [rows[0]], EVIDENCE_POLICY)
        self.assertEqual(result["evidence_status"], "ready")
        self.assertEqual((result["accepted_events"], result["excluded_events"]), (5, 4))
        self.assertEqual(result["rows"], [
            {"kind": "candidate_miss", "label": "other", "frame_count": 1, "source_count": 1, "ready": False},
            {**TARGET, "frame_count": 2, "source_count": 2, "ready": True},
            {"kind": "ranking", "label": TARGET["label"], "frame_count": 1, "source_count": 1, "ready": False},
        ])

    def test_same_source_or_no_policy_never_qualifies(self):
        rows = [event("one", "source-a", "image-a"), event("two", "source-a", "image-b")]
        qualified = summarize_feedback(rows, EVIDENCE_POLICY)
        self.assertEqual(qualified["evidence_status"], "insufficient_evidence")
        self.assertEqual(qualified["rows"][0]["source_count"], 1)
        pending = summarize_feedback(evidence())
        self.assertEqual(pending["evidence_status"], "pending_policy")
        self.assertFalse(pending["rows"][0]["ready"])

    def test_conflicting_id_and_duplicate_image_provenance_raise(self):
        with self.assertRaisesRegex(ValueError, "event ID"):
            summarize_feedback([event("one", "source-a", "image-a"), event("one", "source-b", "image-b")])
        with self.assertRaisesRegex(ValueError, "multiple development sources"):
            summarize_feedback([event("one", "source-a", "image-a"), event("two", "source-b", "image-a")])

    def test_target_must_have_enough_development_evidence(self):
        registry, _, before, after = setup_comparison()
        proposal = prepare_release(registry, {"retry": 2}, {"kind": "ranking", "label": TARGET["label"]},
                                   evidence(), EVIDENCE_POLICY)
        result = compare_release(proposal, before, after, QUALITY_POLICY)
        self.assertEqual(result["issues"], ["insufficient_evidence"])
        with self.assertRaisesRegex(ValueError, "lacks matching evidence"):
            promote_release(registry, proposal, result, approval(result["sha256"]))

    def test_same_source_or_image_as_development_rejects_eval(self):
        _, proposal, before, after = setup_comparison()
        for field, value in (("source_sha256", digest("source-a")), ("image_sha256", digest("image-a"))):
            with self.subTest(field=field):
                updated = [deepcopy(trial) for trial in (before, after)]
                for trial in updated:
                    trial["inputs"]["frames"][0][field] = value
                with self.assertRaisesRegex(ValueError, "overlap"):
                    compare_release(proposal, *updated, QUALITY_POLICY)

    def test_both_trials_need_same_budget_candidates_and_human_truth(self):
        _, proposal, before, after = setup_comparison()
        for key, value in (("budget", {"rechecks": 3}), ("candidate_pool_sha256", digest("different candidates"))):
            with self.subTest(key=key):
                altered = deepcopy(after)
                altered["inputs"][key] = value
                with self.assertRaisesRegex(ValueError, "must share"):
                    compare_release(proposal, before, altered, QUALITY_POLICY)
        altered = deepcopy(after)
        altered["inputs"]["frames"][0]["human_reference_sha256"] = digest("different reference")
        with self.assertRaisesRegex(ValueError, "must share"):
            compare_release(proposal, before, altered, QUALITY_POLICY)

    def test_quality_failures_and_absent_policy_block(self):
        _, proposal, before, after = setup_comparison()
        pending = compare_release(proposal, before, after)
        self.assertEqual((pending["status"], pending["issues"]), ("blocked", ["pending_quality_policy"]))
        after["metrics"].update(auto_candidates=2, false_releases=2, manual_review=4)
        comparison = compare_release(proposal, before, after, QUALITY_POLICY)
        self.assertEqual(comparison["status"], "blocked")
        self.assertCountEqual(comparison["issues"], [
            "insufficient_candidate_coverage", "false_release_limit_not_met", "false_release_rate_regressed",
            "review_budget_regressed", "correct_auto_candidates_regressed", "no_improvement",
        ])

    def test_promotion_is_explicit_and_loads_only_in_new_batch_then_rolls_back(self):
        registry, proposal, before, after = setup_comparison()
        old_batch = batch_snapshot(registry, "batch-001")
        result = compare_release(proposal, before, after, QUALITY_POLICY)
        self.assertEqual(result["status"], "eligible")
        self.assertEqual((result["evaluation_frames"], result["evaluation_sources"]), (6, 2))
        self.assertEqual(result["candidate"]["eligible_correct"], 4)
        self.assertEqual(result["candidate"]["false_release_rate"], 0.20)
        with self.assertRaisesRegex(ValueError, "Explicit human approval"):
            promote_release(registry, proposal, result, approval(digest("unrelated")))
        updated = promote_release(registry, proposal, result, approval(result["sha256"]))
        self.assertEqual(old_batch["version"]["rules"], {"retry": 1})
        self.assertEqual(batch_snapshot(registry, "batch-test")["version"], old_batch["version"])
        self.assertEqual(batch_snapshot(updated, "batch-002")["version"]["rules"], {"retry": 2})
        version_one = old_batch["version"]["sha256"]
        restored = rollback_release(updated, version_one, approval(rollback_binding(updated, version_one)))
        self.assertEqual(batch_snapshot(restored, "batch-003")["version"], old_batch["version"])
        self.assertEqual([row["action"] for row in restored["history"]], ["promote", "rollback"])
        with self.assertRaisesRegex(ValueError, "Baseline changed"):
            promote_release(restored, proposal, result, approval(result["sha256"]))

    def test_mutated_rule_or_metrics_cannot_be_promoted(self):
        registry, proposal, before, after = setup_comparison()
        result = compare_release(proposal, before, after, QUALITY_POLICY)
        altered = deepcopy(result)
        altered["candidate"]["false_release_rate"] = 0
        with self.assertRaisesRegex(ValueError, "Changed or invalid"):
            promote_release(registry, proposal, altered, approval(result["sha256"]))
        proposal["candidate"]["rules"]["retry"] = 3
        with self.assertRaisesRegex(ValueError, "Changed or invalid"):
            promote_release(registry, proposal, result, approval(result["sha256"]))


if __name__ == "__main__":
    unittest.main()
