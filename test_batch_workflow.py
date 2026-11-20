"""Exercise the shadow audit workflow with a real temporary project and saved predictions."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from batch_acceptance import AUTO_CANDIDATE, HELD, HUMAN_CONFIRMED
from batch_workflow import STORE, audit_frame, feedback_events, overview, prepare_audit, record_correction, seal_reference, submit_audit
from generic_job import INFERENCE_SCHEMA, GenericJob, create_job
from project import Project


RATE_POLICY = {"revision": "synthetic-test-only", "max_error_rate": 0.25, "confidence": 0.95}
AUDIT_ALL = {AUTO_CANDIDATE: 3, HELD: 2, HUMAN_CONFIRMED: 1}
BOX = {"label": "valve", "xyxy": [2, 2, 9, 10]}
MISSED = {"label": "valve", "xyxy": [11, 2, 18, 10]}


class BatchWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = Project.create(self.root / "project")
        files = []
        for index in range(6):
            path = self.root / f"fixture-{index}.png"
            Image.new("RGB", (24, 16), (30, 40 + index, 70)).save(path)
            files.append(path)
        self.project.add_images(files)
        job = create_job(self.project, "valve | valve | visible body")
        rows = []
        for index, frame in enumerate(job.manifest["frames"]):
            row = {
                "id": frame["id"],
                "image_sha256": frame["image_sha256"],
                "boxes": [BOX],
                "exceptions": [{"code": "class_conflict"}] if index == 3 else [],
                "candidates": [{**BOX, "id": "mm-01", "verdict": "accepted", "votes": [{"winner": "A"}]}],
            }
            if index != 4:
                row["shadow_qc"] = {"disposition": AUTO_CANDIDATE if index < 3 else HELD,
                                    "retry_count": int(index == 3)}
            rows.append(row)
        prediction = self.root / "inference-fixture.json"
        prediction.write_text(json.dumps({"schema": INFERENCE_SCHEMA,
                                          "job_id": job.manifest["job_id"],
                                          "manifest_sha256": job.state["manifest_sha256"],
                                          "frames": rows}), encoding="utf-8")
        job.publish_inference(prediction)
        job.review(rows[5]["id"], [BOX])
        self.job = job
        self.ids = [row["id"] for row in rows]

    def _freeze(self, *, sizes=AUDIT_ALL, seed=42, policy=RATE_POLICY):
        return prepare_audit(self.job, sample_sizes=sizes, seed=seed, policy=policy)

    def _review(self, plan, frame, *, role="operator", issues=(), boxes=None, actor="operator-1"):
        seal_reference(self.job, frame, [BOX] if boxes is None else boxes, actor=actor,
                       role=role, plan_sha256=plan["sha256"])
        return submit_audit(self.job, frame, issues=issues, actor=actor, plan_sha256=plan["sha256"])

    def test_freeze_samples_every_stratum_with_equal_known_probabilities(self):
        sizes = {AUTO_CANDIDATE: 2, HELD: 1, HUMAN_CONFIRMED: 1}
        plan = self._freeze(sizes=sizes)
        status = overview(GenericJob.open(self.job.directory))
        self.assertEqual(status["populations"], {AUTO_CANDIDATE: 3, HELD: 2, HUMAN_CONFIRMED: 1})
        self.assertEqual(status["retries"], 1)
        self.assertEqual(status["entry"]["plan"], plan)
        self.assertEqual(status["result"]["status"], "pending_reviews")
        self.assertEqual({key: row["inclusion_probability"] for key, row in status["result"]["strata"].items()},
                         {AUTO_CANDIDATE: 2 / 3, HELD: 1 / 2, HUMAN_CONFIRMED: 1})
        self.assertEqual(status["prior_rounds"], 0)
        unsampled = next(frame for frame in self.ids[:3] if frame not in plan["sample_ids"][AUTO_CANDIDATE])
        with self.assertRaisesRegex(ValueError, "抽檢"):
            audit_frame(self.job, unsampled, plan_sha256=plan["sha256"])
        self.assertEqual(self._freeze(sizes=dict(reversed(list(sizes.items())))), plan)

    def test_rejects_a_second_sampling_plan_without_any_inventory_change(self):
        plan = self._freeze()
        store = self.job.directory / STORE
        original = store.read_bytes()
        for overrides in ({"seed": 43},
                          {"sizes": {**AUDIT_ALL, AUTO_CANDIDATE: 2}},
                          {"policy": {**RATE_POLICY, "max_error_rate": 0.50}}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "既有抽樣"):
                self._freeze(**overrides)
        self.assertEqual(store.read_bytes(), original)
        self.assertEqual(self._freeze(), plan)
        self.assertEqual(store.read_bytes(), original)

    def test_original_is_blind_until_reference_is_sealed_and_one_operator_adjudicates_once(self):
        plan = self._freeze()
        frame = self.ids[0]
        frozen_paths = (self.job.manifest_path, self.job.inference_path, self.job.directory / "state.json",
                        self.job.project_dir / "project.json")
        frozen = {path: path.read_bytes() for path in frozen_paths}
        prompt = audit_frame(self.job, frame, plan_sha256=plan["sha256"])
        self.assertTrue(prompt["image"].is_file())
        self.assertEqual(prompt["labels"], ["valve"])
        self.assertEqual(prompt["comparison_boxes"], [])
        self.assertIsNone(prompt["reference"])
        self.assertIsNone(prompt["review"])
        with self.assertRaisesRegex(ValueError, "整張盲標"):
            submit_audit(self.job, frame, issues=(), actor="operator-1", plan_sha256=plan["sha256"])

        seal_reference(self.job, frame, [BOX, MISSED], actor=" operator-1 ", plan_sha256=plan["sha256"])
        comparison = audit_frame(self.job, frame, plan_sha256=plan["sha256"])
        self.assertEqual(comparison["reference"]["boxes"], [BOX, MISSED])
        self.assertEqual(comparison["comparison_boxes"], [{**BOX, "source": "mm_qwen"}])
        with self.assertRaisesRegex(ValueError, "已封存"):
            seal_reference(self.job, frame, [BOX], actor="operator-1", plan_sha256=plan["sha256"])
        with self.assertRaisesRegex(ValueError, "原覆核者"):
            submit_audit(self.job, frame, issues=(), actor="someone-else", plan_sha256=plan["sha256"])
        result = submit_audit(self.job, frame, issues=["omission", "omission"], actor="operator-1",
                              plan_sha256=plan["sha256"])
        self.assertEqual(result["status"], "pending_reviews")
        self.assertEqual(result["strata"][AUTO_CANDIDATE]["errors"], 1)
        self.assertEqual(feedback_events(self.job)[0]["diagnoses"],
                         [{"kind": "candidate_miss", "label": "valve"}, {"kind": "qc_miss", "label": "valve"}])
        self.assertEqual(audit_frame(self.job, frame, plan_sha256=plan["sha256"])["review"]["issues"], ["omission"])
        with self.assertRaisesRegex(ValueError, "只能計分一次"):
            submit_audit(self.job, frame, issues=(), actor="operator-1", plan_sha256=plan["sha256"])
        self.assertEqual({path: path.read_bytes() for path in frozen_paths}, frozen)

    def test_demo_is_distinguishable_from_operator_and_keeps_source_identity(self):
        plan = self._freeze(policy=None)
        self._review(plan, self.ids[0], actor="human")
        self._review(plan, self.ids[1], role="demonstration", actor="demo-runner")
        by_role = {row["role"]: row for row in feedback_events(self.job)}
        self.assertEqual(set(by_role), {"audit", "demonstration"})
        rows = {row["id"]: row for row in self.job.frame_rows()}
        for role, frame in (("audit", self.ids[0]), ("demonstration", self.ids[1])):
            event = by_role[role]
            self.assertEqual(event["image_sha256"], rows[frame]["image_sha256"])
            source_key = rows[frame]["source_group"]
            self.assertEqual(event["source_sha256"], source_key.split(":", 1)[1])
            self.assertEqual(event["review_id"], f"{plan['sha256']}:{frame}")
        self.assertEqual(overview(self.job)["result"]["status"], "pending_policy")

    def test_demo_does_not_fill_operator_review_requirement(self):
        plan = self._freeze()
        for samples in plan["sample_ids"].values():
            for frame in samples:
                self._review(plan, frame, role="demonstration" if frame == self.ids[0] else "operator")
        result = overview(self.job)["result"]
        self.assertEqual(result["status"], "pending_reviews")
        self.assertEqual(result["demonstration_reviews"], 1)
        self.assertEqual(result["strata"][AUTO_CANDIDATE]["reviewed"], 2)

    def test_whole_frame_correction_records_development_evidence_once(self):
        frame = self.ids[3]
        self.assertIsNone(record_correction(self.job, frame, actor="annotator"))
        original_inference = self.job.inference_path.read_bytes()
        self.job.review(frame, [BOX, MISSED])
        event = record_correction(self.job, frame, actor="annotator")
        self.assertEqual((event["role"], event["actor"]), ("operator", "annotator"))
        self.assertEqual(event["diagnoses"], [{"kind": "candidate_miss", "label": "valve"}])
        self.assertEqual(record_correction(self.job, frame, actor="annotator"), event)
        self.assertEqual(feedback_events(self.job), [event])
        self.assertEqual(self.job.inference_path.read_bytes(), original_inference)

    def test_completed_blind_census_integrates_the_finite_batch_gate(self):
        plan = self._freeze()
        for stratum, sample in plan["sample_ids"].items():
            for frame in sample:
                issues = ("false_positive",) if stratum == AUTO_CANDIDATE and frame == self.ids[0] else ()
                self._review(plan, frame, issues=issues)
        result = overview(self.job)["result"]
        self.assertEqual((result["status"], result["eligible"]), ("exceeds_limit", False))
        auto = result["strata"][AUTO_CANDIDATE]
        self.assertEqual((auto["errors"], auto["reviewed"], auto["upper_bound"]), (1, 3, 1 / 3))

    def test_changing_frame_disposition_invalidates_the_old_plan_without_losing_the_record(self):
        plan = self._freeze()
        frame = self.ids[0]
        self._review(plan, frame)
        store = self.job.directory / STORE
        original = json.loads(store.read_bytes())
        self.job.review(frame, [MISSED])
        status = overview(self.job)
        self.assertEqual(status["populations"], {AUTO_CANDIDATE: 2, HELD: 2, HUMAN_CONFIRMED: 2})
        self.assertIsNone(status["entry"])
        self.assertEqual(status["prior_rounds"], 1)
        with self.assertRaisesRegex(ValueError, "抽檢"):
            audit_frame(self.job, frame, plan_sha256=plan["sha256"])
        updated = self._freeze()
        self.assertNotEqual(updated["sha256"], plan["sha256"])
        persisted = json.loads(store.read_bytes())
        for revision, old_round in original["rounds"].items():
            self.assertEqual(persisted["rounds"][revision], old_round)
        self.assertEqual(len(persisted["rounds"]), 2)

    def test_detects_source_image_drift_without_updating_the_frozen_plan(self):
        plan = self._freeze()
        frame = self.ids[0]
        selected_image = audit_frame(self.job, frame, plan_sha256=plan["sha256"])["image"]
        store = self.job.directory / STORE
        original = store.read_bytes()
        Image.new("RGB", (24, 16), "white").save(selected_image)
        with self.assertRaisesRegex(ValueError, "原圖"):
            audit_frame(self.job, frame, plan_sha256=plan["sha256"])
        with self.assertRaisesRegex(ValueError, "原圖"):
            self._freeze()
        self.assertEqual(store.read_bytes(), original)

    def test_independent_project_marker_prevents_development_audit(self):
        marker = self.job.project_dir / "acceptance-inventory.json"
        marker.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "驗收專案"):
            self._freeze()
        self.assertFalse((self.job.directory / STORE).exists())


if __name__ == "__main__":
    unittest.main()
