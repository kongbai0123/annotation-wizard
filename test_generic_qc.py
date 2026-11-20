"""Model-free checks for shadow QC evidence, retry bounds and legacy compatibility."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from generic_inference import DEFAULTS, execute_retries, file_hash, run_inference, select_vote
from generic_qc import SAMShadowEvidence, describe_frame_qc, qwen_class_evidence


LABELS = ["container_handler", "forklift"]
SETTINGS = {**DEFAULTS, "shadow_qc": True}
POSITIVE = [([0.82, 0.10, 0.04, 0.04], .9), ([0.86, 0.06, 0.04, 0.04], .9)]
EXTENT_HINT = [([0.10, 0.82, 0.04, 0.04], .9), ([0.06, 0.86, 0.04, 0.04], .9)]
NEGATIVE = [([0.04, 0.04, 0.82, 0.10], .9), ([0.04, 0.04, 0.86, 0.06], .9)]


def candidate(ident, box, label=LABELS[0], score=.8, votes=POSITIVE):
    return {"id": ident, "label": label, "xyxy": box, "score": score, **select_vote(votes, SETTINGS)}


def frame_with(accepted, *, others=(), retries=(), exceptions=()):
    return {"candidates": [*accepted, *others],
            "boxes": [{"candidate_id": item["id"], "label": item["label"], "xyxy": item["xyxy"]}
                      for item in accepted],
            "retries": list(retries), "exceptions": list(exceptions)}


def evaluate(frame, evidence=None):
    return describe_frame_qc(frame, SETTINGS, width=100, height=100, labels=LABELS, evidence=evidence)


class ShadowFrameQCTests(unittest.TestCase):
    def test_clean_frame_is_shadow_candidate_without_claiming_mask_evidence(self):
        diag = evaluate(frame_with([candidate("valid", [12, 12, 42, 49])]))
        self.assertEqual(diag["disposition"], "auto_candidate")
        self.assertTrue(diag["shadow_only"])
        self.assertEqual(diag["checks"][0]["qwen_class"], "target")
        self.assertEqual(diag["checks"][0]["sam_status"], "not_evaluated")
        self.assertEqual(diag["checks"][0]["structure_evidence"], "not_evaluated")
        self.assertEqual(diag["evidence_coverage"]["mm_rival_scope"], "configured_classes")
        self.assertEqual(diag["flags"], [])

    def test_qwen_extent_opinion_does_not_rate_class_negative(self):
        box = candidate("range", [12, 12, 42, 49], votes=EXTENT_HINT)
        self.assertEqual(box["reason_code"], "bad_extent")
        self.assertEqual(qwen_class_evidence(box, SETTINGS), "target")
        diag = evaluate(frame_with([box]))
        self.assertTrue(diag["checks"][0]["qwen_extent_advisory"])
        self.assertEqual(diag["disposition"], "auto_candidate")
        self.assertNotIn("sam_extent_conflict", [flag["code"] for flag in diag["flags"]])

    def test_rejected_overlapping_mm_rival_is_still_class_evidence(self):
        chosen = candidate("selected", [12, 12, 42, 49])
        rival = candidate("other-class", [13, 12, 42, 49], label=LABELS[1], votes=NEGATIVE)
        diag = evaluate(frame_with([chosen], others=[rival]))
        self.assertEqual(rival["verdict"], "rejected")
        self.assertEqual(diag["disposition"], "review")
        self.assertEqual(diag["checks"][0]["mm_class_rivals"][0]["label"], LABELS[1])
        self.assertIn("mm_class_rival", [flag["code"] for flag in diag["flags"]])

    def test_geometry_and_independent_evidence_flag_extent_and_class(self):
        combined = candidate("combined", [10, 10, 90, 90])
        left = candidate("left", [12, 12, 38, 84])
        right = candidate("right", [61, 12, 86, 84])
        diag = evaluate(frame_with([combined], others=[left, right]), {"combined": {
            "sam_box": [12, 12, 38, 84], "structure_label": LABELS[1],
        }})
        codes = {flag["code"] for flag in diag["flags"]}
        self.assertTrue({"possible_merged_instances", "sam_extent_conflict", "structure_class_conflict"} <= codes)
        self.assertEqual(diag["checks"][0]["mm_contained_pair"], ["left", "right"])
        self.assertEqual(diag["evidence_coverage"]["sam_boxes_checked"], 1)
        self.assertEqual(diag["disposition"], "review")

    def test_disagreement_unmatched_or_missing_inference_stays_in_review(self):
        chosen = candidate("selected", [12, 12, 42, 49], votes=[POSITIVE[0], NEGATIVE[0]])
        missed = candidate("missed", [48, 48, 88, 89], votes=NEGATIVE)
        diag = evaluate(frame_with([chosen], others=[missed]))
        self.assertEqual(diag["disposition"], "review")
        self.assertEqual({flag["code"] for flag in diag["flags"]},
                         {"qwen_class_uncertain", "unmatched_mm_proposal"})
        self.assertEqual(evaluate(frame_with([]))["disposition"], "held")

    def test_usable_retried_box_still_requires_whole_frame_confirmation(self):
        chosen = candidate("selected", [12, 12, 42, 49])
        diag = evaluate(frame_with([chosen], retries=[{"seconds": .2}, {"seconds": .3}]))
        self.assertEqual(diag["retry_count"], 2)
        self.assertEqual(diag["retry_seconds"], .5)
        self.assertEqual(diag["disposition"], "review")
        self.assertEqual([flag["code"] for flag in diag["flags"]], ["retry_requires_review"])


class SAMShadowEvidenceTests(unittest.TestCase):
    def test_optional_provider_checks_pins_caps_calls_and_prioritizes_extent_risk(self):
        ranked = [candidate("large", [0, 0, 48, 48], score=.9),
                  candidate("low", [60, 8, 80, 33], score=.7),
                  candidate("high", [60, 60, 90, 90], score=.9)]
        fragments = [candidate("part_a", [2, 2, 20, 45]), candidate("part_b", [25, 2, 44, 45])]
        frame = frame_with(ranked, others=fragments)
        frame["image_sha256"] = "0" * 64
        frame["shadow_qc"] = evaluate(frame)
        calls = []

        class Engine:
            def __init__(self, directory):
                pass

            def generate(self, request, image, *, box, **kwargs):
                calls.append((request, box, kwargs["image_key"]))
                if len(calls) == 2:
                    raise RuntimeError("example mask not found")
                mask = np.zeros((100, 100), dtype=bool)
                mask[2:15, 2:20] = True
                return SimpleNamespace(mask=mask, score=.88)

        with TemporaryDirectory() as temporary, patch("prepare_model.verified", return_value=True) as verify:
            provider = SAMShadowEvidence(Path(temporary), engine_factory=Engine)
            self.assertEqual(verify.call_count, 4)
            self.assertEqual(len(provider.provenance["files"]), 4)
            self.assertEqual(provider.provenance["max_boxes_per_frame"], 2)
            evidence = provider(Image.new("RGB", (100, 100)), frame)
            self.assertEqual(calls, [("sam2", (0, 0, 48, 48), frame["image_sha256"]),
                                     ("sam2", (60, 8, 80, 33), frame["image_sha256"])])
            self.assertEqual(evidence["large"]["sam_box"], [2, 2, 20, 15])
            self.assertEqual(evidence["low"]["sam_status"], "failed")
            self.assertNotIn("high", evidence)
            diag = evaluate(frame, evidence)
            self.assertEqual(diag["evidence_coverage"]["sam_boxes_checked"], 1)
            self.assertEqual(diag["evidence_coverage"]["sam_boxes_failed"], 1)
            self.assertIn("sam_extent_uncertain", [flag["code"] for flag in diag["flags"]])
            self.assertEqual(diag["checks"][2]["sam_status"], "not_evaluated")
            provider.close()
            self.assertIsNone(provider.engine)

    def test_optional_provider_rejects_unverified_files_before_engine_start(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "雜湊不符"):
                SAMShadowEvidence(Path(temporary), engine_factory=self.fail)


class LocalRetryTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100))
        self.labels = {LABELS[0]: {"label": LABELS[0], "prompt": "handler"}}

    def requests(self, *regions):
        return {"candidates": [], "retries": [{"label": LABELS[0], "bounds": box} for box in regions]}

    def test_actual_calls_are_capped_and_legacy_behavior_is_preserved(self):
        calls = []

        class Detector:
            def propose(self, crop, prompt, *, candidate_limit):
                calls.append(crop.size)
                return [([1, 1, 20, 20], .9)]

        regions = ([0, 0, 30, 30], [30, 0, 60, 30], [60, 0, 90, 30])
        shadow = self.requests(*regions)
        execute_retries(Detector(), self.image, shadow, self.labels, SETTINGS)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(shadow["retries"]), 2)
        self.assertEqual(shadow["retry_state"], {"request_count": 3, "stop_reason": "budget_exhausted"})
        legacy = self.requests(*regions)
        execute_retries(Detector(), self.image, legacy, self.labels, DEFAULTS)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(legacy["retries"]), 3)
        self.assertNotIn("retry_state", legacy)

    def test_crop_equal_to_full_frame_is_skipped_and_duplicate_stops_rechecks(self):
        calls = []

        class Detector:
            def propose(self, crop, prompt, *, candidate_limit):
                calls.append(crop.size)
                return [([2, 2, 15, 15], .9)]

        row = self.requests([0, 0, 100, 100], [0, 0, 30, 30], [50, 50, 80, 80])
        row["candidates"] = [candidate("known", [2, 2, 15, 15])]
        execute_retries(Detector(), self.image, row, self.labels, SETTINGS)
        self.assertEqual(calls, [(30, 30)])
        self.assertEqual(row["retries"][0]["novel_eligible_candidates"], 0)
        self.assertEqual(row["retry_state"]["stop_reason"], "no_novel_eligible_proposals")

    def test_local_crop_keeps_newly_eligible_evidence_at_the_original_position(self):
        class Detector:
            def propose(self, crop, prompt, *, candidate_limit):
                return [([2, 2, 15, 15], .7)]

        row = self.requests([0, 0, 30, 30])
        low = candidate("low", [2, 2, 15, 15], score=.09)
        low.update(verdict="rejected", reason_code="low_score_proposal")
        row["candidates"] = [low]
        execute_retries(Detector(), self.image, row, self.labels, SETTINGS)
        self.assertEqual(row["retries"][0]["novel_eligible_candidates"], 1)
        self.assertEqual([item["id"] for item in row["candidates"]], ["low", "retry-0002"])

    def test_worker_publishes_shadow_result_and_preserves_existing_verdict(self):
        class Detector:
            def __init__(self, directory):
                pass

            def propose(self, image, prompt, *, candidate_limit):
                return [] if image.size == (100, 100) else [([12, 12, 42, 42], .9)]

        class Judge:
            def __init__(self, directory, classes, settings):
                pass

            def scan(self, image):
                return {"status": "parsed", "targets": [{"label": LABELS[0], "xyxy": [20, 20, 50, 50]}]}

            def review(self, image, candidates):
                for item in candidates:
                    item.update(select_vote(POSITIVE, SETTINGS))

        with TemporaryDirectory() as temp:
            directory = Path(temp)
            image = directory / "image.png"
            self.image.save(image)
            (directory / "model.safetensors").write_bytes(b"unused test weight")
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"job_id": "shadow-smoke", "config": {"classes": [{
                "id": LABELS[0], "label": LABELS[0], "prompt": "handler", "description": "one handler"
            }], "settings": {"shadow_qc": True}}, "frames": [{
                "id": "single", "image": image.name, "width": 100, "height": 100,
                "image_sha256": file_hash(image),
            }]}), encoding="utf-8")
            live_path = directory / "output.json"
            report = run_inference(manifest, directory, directory, live_path,
                                   engine_factory=Detector, judge_factory=Judge)
            prediction = report["frames"][0]
            self.assertEqual(prediction["status"], "ready")
            self.assertEqual(prediction["boxes"][0]["xyxy"], [20, 20, 50, 50])
            self.assertEqual(prediction["shadow_qc"]["disposition"], "review")
            self.assertEqual(report["summary"]["shadow_qc"]["retry_count"], 1)
            self.assertEqual(report["summary"]["shadow_qc"]["dispositions"], {"review": 1})
            self.assertFalse(report["method"]["shadow_qc"]["export_policy_changed"])
            self.assertEqual(json.loads(live_path.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
