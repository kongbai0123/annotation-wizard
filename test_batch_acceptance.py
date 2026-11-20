import hashlib
import json
from copy import deepcopy
from fractions import Fraction
from math import comb
import unittest

from batch_acceptance import (
    AUTO_CANDIDATE,
    HELD,
    HUMAN_CONFIRMED,
    assess_batch,
    freeze_batch,
    hypergeometric_error_bounds,
)


POLICY = {"revision": "reviewed-policy-v1", "confidence": 0.95, "max_error_rate": 0.10}


def frames(auto=36, held=0, human=0):
    return [
        {"frame_id": f"{stratum}-{index:03d}", "stratum": stratum, "revision": f"image-and-state-{index}"}
        for stratum, count in ((AUTO_CANDIDATE, auto), (HELD, held), (HUMAN_CONFIRMED, human))
        for index in range(count)
    ]


def freeze(rows=None, auto=29, held=0, human=0, policy=POLICY):
    return freeze_batch(
        frames() if rows is None else rows,
        sample_sizes={AUTO_CANDIDATE: auto, HELD: held, HUMAN_CONFIRMED: human},
        seed=21,
        policy=policy,
    )


def reviews_for(manifest, *, errors=()):
    return {
        frame_id: frame_id in errors
        for sample in manifest["sample_ids"].values()
        for frame_id in sample
    }


class HypergeometricBoundsTests(unittest.TestCase):
    def test_bounds_match_direct_tail_sums_for_small_populations(self):
        for population in range(1, 11):
            for draws in range(population + 1):
                for observed in range(draws + 1):
                    for confidence in (0.50, 0.80, 0.95, 0.99):
                        denominator = comb(population, draws)
                        alpha = 1 - Fraction(str(confidence))
                        lower_candidates, upper_candidates = [], []
                        for total in range(population + 1):
                            terms = [
                                Fraction(comb(total, actual) * comb(population - total, draws - actual), denominator)
                                for actual in range(draws + 1)
                            ]
                            if sum(terms[observed:]) >= alpha:
                                lower_candidates.append(total)
                            if sum(terms[:observed + 1]) >= alpha:
                                upper_candidates.append(total)
                        self.assertEqual(
                            hypergeometric_error_bounds(population, draws, observed, confidence),
                            (min(lower_candidates), max(upper_candidates)),
                            (population, draws, observed, confidence),
                        )

    def test_small_lot_and_census_have_different_upper_bounds(self):
        self.assertEqual(hypergeometric_error_bounds(36, 6, 0, 0.95), (0, 13))
        self.assertEqual(hypergeometric_error_bounds(36, 36, 2, 0.95), (2, 2))
        self.assertEqual(hypergeometric_error_bounds(36, 0, 0, 0.95), (0, 36))

    def test_invalid_parameters_fail(self):
        for case in ((3, 4, 0, 0.95), (3, 1, 2, 0.95), (3, 3, 0, 1),
                     (3, -1, 0, 0.95), (True, 1, 1, 0.95), (3, 1, 1, float("nan"))):
            with self.subTest(case=case), self.assertRaises(ValueError):
                hypergeometric_error_bounds(*case)


class BatchAcceptanceTests(unittest.TestCase):
    def test_json_roundtrip_and_deterministic_stratified_samples(self):
        rows = frames(auto=12, held=4, human=6)
        manifest = freeze(rows, auto=5, held=1, human=2)
        self.assertEqual(manifest, freeze(reversed(rows), auto=5, held=1, human=2))
        manifest = json.loads(json.dumps(manifest))
        reports = assess_batch(manifest, reviews_for(manifest), current_frames=rows)["strata"]
        self.assertEqual(reports[AUTO_CANDIDATE]["inclusion_probability"], 5 / 12)
        self.assertEqual(reports[HELD]["inclusion_probability"], 1 / 4)
        self.assertEqual(reports[HUMAN_CONFIRMED]["inclusion_probability"], 2 / 6)
        for stratum, ids in manifest["sample_ids"].items():
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(frame_id.startswith(stratum) for frame_id in ids))

    def test_gate_reports_pass_fail_and_insufficient_evidence_separately(self):
        manifest = freeze(auto=36)
        passed = assess_batch(manifest, reviews_for(manifest))
        failed = assess_batch(manifest, reviews_for(manifest, errors=manifest["sample_ids"][AUTO_CANDIDATE][:4]))
        small = freeze(auto=6)
        uncertain = assess_batch(small, reviews_for(small))
        self.assertEqual((passed["status"], passed["eligible"]), ("passed", True))
        self.assertEqual((failed["status"], failed["eligible"]), ("exceeds_limit", False))
        self.assertEqual((uncertain["status"], uncertain["eligible"]), ("insufficient_evidence", False))
        self.assertGreater(uncertain["strata"][AUTO_CANDIDATE]["upper_bound"], POLICY["max_error_rate"])

    def test_limit_is_inclusive_and_uses_exact_count(self):
        rows = frames(auto=10)
        manifest = freeze(rows, auto=10)
        self.assertEqual(assess_batch(manifest, reviews_for(manifest, errors=["auto_candidate-000"]))["status"], "passed")

    def test_three_strata_and_diagnostics_are_reported_separately(self):
        manifest = freeze(frames(auto=8, held=3, human=3), auto=8, held=3, human=3)
        diagnostic_errors = manifest["sample_ids"][HELD] + manifest["sample_ids"][HUMAN_CONFIRMED]
        result = assess_batch(manifest, reviews_for(manifest, errors=diagnostic_errors))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["strata"][HELD]["errors"], 3)
        self.assertEqual(result["strata"][HUMAN_CONFIRMED]["errors"], 3)
        self.assertEqual(result["strata"][AUTO_CANDIDATE]["errors"], 0)

    def test_no_policy_and_pending_reviews_are_fail_closed(self):
        unapproved = freeze(auto=36, policy=None)
        result = assess_batch(unapproved, reviews_for(unapproved))
        self.assertEqual((result["status"], result["eligible"]), ("pending_policy", False))
        self.assertEqual(result["strata"][AUTO_CANDIDATE]["reviewed"], 36)
        self.assertIsNone(result["strata"][AUTO_CANDIDATE]["upper_bound"])
        plan = freeze(frames(auto=1, held=1), auto=1, held=1)
        result = assess_batch(plan, {"auto_candidate-000": False})
        self.assertEqual((result["status"], result["eligible"]), ("pending_reviews", False))

    def test_empty_auto_population_is_not_a_successful_auto_audit(self):
        plan = freeze(frames(auto=0, human=1), human=1)
        result = assess_batch(plan, reviews_for(plan))
        self.assertEqual(result["strata"][AUTO_CANDIDATE]["status"], "not_applicable")
        self.assertIsNone(result["strata"][AUTO_CANDIDATE]["upper_bound"])
        self.assertEqual(result["status"], "passed")
        only_held = freeze(frames(auto=0, held=1), held=1)
        self.assertEqual(assess_batch(only_held, reviews_for(only_held))["status"], "insufficient_evidence")
        no_review = freeze(auto=0)
        self.assertEqual(assess_batch(no_review, {})["status"], "insufficient_evidence")

    def test_sample_mutation_and_truthy_values_are_rejected(self):
        plan = freeze(auto=6)
        unsampled = next(row["frame_id"] for row in plan["frames"] if row["frame_id"] not in plan["sample_ids"][AUTO_CANDIDATE])
        with self.assertRaisesRegex(ValueError, "selected"):
            assess_batch(plan, {unsampled: False})
        with self.assertRaisesRegex(ValueError, "completed"):
            assess_batch(plan, {plan["sample_ids"][AUTO_CANDIDATE][0]: 0})
        forged = deepcopy(plan)
        forged["sample_ids"][AUTO_CANDIDATE][0] = unsampled
        del forged["sha256"]
        digest = hashlib.sha256(
            json.dumps(forged, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()
        forged["sha256"] = digest
        with self.assertRaisesRegex(ValueError, "sample membership"):
            assess_batch(forged, {})

    def test_manifest_binds_frame_content_membership_and_disposition(self):
        rows = frames(auto=2, human=1)
        plan = freeze(rows, auto=2)
        changed = deepcopy(rows)
        changed[1]["revision"] = "corrected"
        with self.assertRaisesRegex(ValueError, "revision changed"):
            assess_batch(plan, {}, current_frames=changed)
        changed = deepcopy(rows)
        changed[1]["stratum"] = HELD
        with self.assertRaisesRegex(ValueError, "stratum"):
            assess_batch(plan, {}, current_frames=changed)
        with self.assertRaisesRegex(ValueError, "membership"):
            assess_batch(plan, {}, current_frames=rows[:-1])

    def test_duplicate_frames_and_incomplete_policy_are_rejected(self):
        row = frames(auto=1)[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze([row, row])
        with self.assertRaisesRegex(ValueError, "policy"):
            freeze(policy={"revision": "draft", "confidence": 0.95})
        plan = freeze(auto=1)
        plan["policy"]["max_error_rate"] = 0.90
        with self.assertRaisesRegex(ValueError, "manifest"):
            assess_batch(plan, {})


if __name__ == "__main__":
    unittest.main()
