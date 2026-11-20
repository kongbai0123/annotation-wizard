"""Exercise interchangeable classes, frame provenance, and export without a GPU."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from generic_job import (
    GenericJob, INFERENCE_SCHEMA, REFERENCE_SCHEMA, create_job, create_manual_training_job, jobs_for, parse_classes,
)
from project import Project


def _write(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return path


def _box(label="襯套", x=1):
    return {"label": label, "xyxy": [x, 1, x + 5, 7]}


def _predictions(job, rows):
    frames = []
    for frame, (boxes, exceptions) in zip(job.manifest["frames"], rows, strict=True):
        frames.append({"id": frame["id"], "image_sha256": frame["image_sha256"], "boxes": boxes,
                       "exceptions": exceptions})
    return {"schema": INFERENCE_SCHEMA, "job_id": job.manifest["job_id"],
            "manifest_sha256": job.state["manifest_sha256"], "frames": frames}


class GenericJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.project = Project.create(self.root / "project")
        images = []
        for index in range(4):
            path = self.root / f"capture-{index}.png"
            Image.new("RGB", (20, 10), (10, 20 + index, 30)).save(path)
            images.append(path)
        self.project.add_images(images)

    def test_same_engine_accepts_unrelated_class_definitions(self):
        self.assertEqual(parse_classes("鏽點\n零件 | cap | visible body"), [
            {"id": 0, "label": "鏽點", "prompt": "鏽點", "description": "",
             "prompt_provided": False, "description_provided": False},
            {"id": 1, "label": "零件", "prompt": "cap", "description": "visible body",
             "prompt_provided": True, "description_provided": True},
        ])
        with self.assertRaisesRegex(ValueError, "重複"):
            parse_classes("Bolt\nbolt")

        for label, class_text, provided in [("襯套", "襯套", False), ("盤扣", "盤扣 | 盤扣", True)]:
            job = create_job(self.project, parse_classes(class_text))
            self.assertEqual(job.manifest["config"]["classes"][0]["prompt_provided"], provided)
            report = _predictions(job, [([_box(label)], [])] * 4)
            job.publish_inference(_write(self.root / "model.json", report))
            export = job.export_dataset()
            inventory = json.loads((export / "dataset.json").read_bytes())
            self.assertEqual(inventory["labels"], [label])
            self.assertEqual({frame["split"] for frame in inventory["frames"]}, {"train"})
            self.assertIsNone(inventory["independent_test"])
            self.assertEqual(job.export_dataset(), export)
        self.assertEqual(len(jobs_for(self.project)), 2)
        self.assertEqual(Project.open(self.project.directory).labels, [])

    def test_pending_or_excluded_frames_cannot_become_accidental_negatives(self):
        job = create_job(self.project, "襯套")
        report = _predictions(job, [([_box()], []), ([_box()], [{"code": "merge"}]), ([], []), ([], [])])
        self.assertTrue(all(frame["status"] == "awaiting_inference" for frame in job.frame_rows()))
        output = _write(self.root / "model.json", report)
        self.assertEqual(job.publish_inference(output)["needs_review"], 3)
        first_export = job.export_dataset()
        self.assertEqual(len(json.loads((first_export / "dataset.json").read_bytes())["frames"]), 1)
        merged, ambiguous, excluded = job.frames_for_review()
        self.assertEqual(ambiguous["exceptions"], [{"code": "empty_unverified", "reason": "整張畫格尚未有人確認為空畫面。"}])
        with self.assertRaisesRegex(ValueError, "範圍"):
            job.review(merged["id"], [_box(x=19)])
        job.review(merged["id"], [_box(x=2), _box(x=8)])
        job.review(ambiguous["id"], [])
        job.review(excluded["id"], exclude=True)
        reloaded = GenericJob.open(job.directory)
        self.assertEqual(reloaded.summary()["boxes_by_class"], {"襯套": 3})
        self.assertEqual(reloaded.summary()["excluded"], 1)
        self.assertEqual(reloaded.frame_rows()[1]["exceptions"], [{"code": "merge"}])
        self.assertIn("name", reloaded.frame_rows()[1])
        self.assertFalse(reloaded.frames_for_review())
        second_export = reloaded.export_dataset()
        self.assertNotEqual(first_export, second_export)
        self.assertEqual((second_export / "labels" / f"{ambiguous['id']}.txt").read_bytes(), b"")
        self.assertFalse((second_export / "labels" / f"{excluded['id']}.txt").exists())
        self.assertEqual(reloaded.inference, report)
        reloaded.reopen(merged["id"])
        self.assertEqual(reloaded.summary()["needs_review"], 1)

    def test_holdout_excludes_entire_source_from_inference_and_training(self):
        train, holdout, same_source, other = self.project.assets
        reference = _write(self.root / "blind.json", {
            "schema": REFERENCE_SCHEMA, "provenance": "independent blind operator",
            "source_groups": {same_source["source_sha256"]: "camera-B"},
            "frames": [{"source_sha256": holdout["source_sha256"], "source_group": "camera-B",
                        "boxes": [_box()]}],
        })
        job = create_job(self.project, "襯套", source_groups={holdout["id"]: "camera-B"},
                         holdout=reference)
        self.assertEqual({row["id"] for row in job.manifest["frames"]}, {train["id"], other["id"]})
        self.assertNotIn(holdout["image_sha256"], job.manifest_path.read_text(encoding="utf-8"))
        self.assertNotIn("independent blind operator", job.manifest_path.read_text(encoding="utf-8"))
        self.project.add_label("襯套")
        for item in (train, holdout):
            self.project.add_box(item["id"], "襯套", _box()["xyxy"])
            self.project.set_reviewed(item["id"], True)
        manual = create_manual_training_job(self.project, "襯套")
        self.assertEqual([row["id"] for row in manual.manifest["frames"]], [train["id"]])
        output = _write(self.root / "model.json", _predictions(job, [([_box()], [])] * 2))
        job.publish_inference(output)
        export = job.export_dataset()
        inventory = json.loads((export / "dataset.json").read_bytes())
        self.assertEqual(inventory["independent_test"]["source_sha256"], sha256(reference.read_bytes()).hexdigest())
        test = [row for row in inventory["frames"] if row["split"] == "test"]
        self.assertEqual([(row["image_sha256"], row["shot"]) for row in test],
                         [(holdout["image_sha256"], "camera-B")])
        self.assertEqual(job.summary()["reference_frames"], 1)
        inventory["frames"][-1]["split"] = "train"
        _write(export / "dataset.json", inventory)
        with self.assertRaisesRegex(ValueError, "快照"):
            job.export_dataset()

    def test_manifest_and_model_hashes_are_checked_on_reopen_or_export(self):
        job = create_job(self.project, "襯套")
        report = _predictions(job, [([_box()], [])] * 4)
        incorrect_report = {**report, "manifest_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "封存"):
            job.publish_inference(_write(self.root / "bad-model.json", incorrect_report))
        job.publish_inference(_write(self.root / "model.json", report))
        relative = job.manifest["frames"][0]["image"]
        (job.project_dir / relative).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            job.export_dataset()
        self.assertEqual(list((job.directory / "datasets").iterdir()), [])
        job.manifest_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "校驗"):
            GenericJob.open(job.directory)

    def test_single_frame_boxes_win_over_conflicting_ai_and_keep_unrelated_candidates(self):
        asset = self.project.assets[0]["id"]
        self.project.add_label("襯套")
        self.project.add_label("零件")
        job = create_job(self.project, "襯套\n零件")
        manual = _box(x=2)
        self.project.add_box(asset, manual["label"], manual["xyxy"])
        self.assertEqual(job.frame_rows()[0]["manual_box_count"], 1)
        incorrect_class = _box("零件", x=2)
        different_object = _box("零件", x=12)
        oversized = {"label": "襯套", "xyxy": [0, 0, 14, 10]}
        report = _predictions(job, [([incorrect_class, oversized, different_object], [])] + [([_box()], [])] * 3)
        job.publish_inference(_write(self.root / "model.json", report))
        sealed = job.inference_path.read_bytes()

        partial = job.frame_rows()[0]
        self.assertEqual(partial["status"], "pending")
        self.assertEqual(partial["boxes"], [{**manual, "source": "manual"}, {**different_object, "source": "mm_qwen"}])
        self.assertEqual(partial["exceptions"][0]["code"], "manual_review_pending")
        self.assertFalse((job.export_dataset() / "labels" / f"{asset}.txt").exists())

        self.project.set_reviewed(asset, True)
        complete = job.frame_rows()[0]
        self.assertEqual(complete["status"], "approved")
        self.assertEqual(complete["boxes"], [{**manual, "source": "manual"}])
        labels = (job.export_dataset() / "labels" / f"{asset}.txt").read_text().splitlines()
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].split()[0], "0")
        self.assertEqual(job.inference_path.read_bytes(), sealed)

    def test_batch_edits_write_back_to_single_editor_and_reject_a_stale_draft(self):
        asset = self.project.assets[0]["id"]
        self.project.add_label("場景")
        unrelated = self.project.add_box(asset, "場景", [7, 1, 16, 6])
        job = create_job(self.project, "襯套")
        report = _predictions(job, [([_box()], [])] * 4)
        job.publish_inference(_write(self.root / "model.json", report))
        stale = job.frame_rows()[0]["project_annotation_revision"]
        self.project.replace_box(asset, unrelated, [8, 1, 16, 6])
        with self.assertRaisesRegex(ValueError, "已有新版本"):
            job.review(asset, [_box(x=2)], expected_project_revision=stale)
        current = job.frame_rows()[0]["project_annotation_revision"]
        job.review(asset, [_box(x=2)], expected_project_revision=current)
        single = Project.open(self.project.directory)
        saved = {box["label"]: box for box in single.assets[0]["annotations"]}
        self.assertEqual(saved["場景"]["id"], unrelated)
        self.assertEqual(saved["襯套"]["bbox"], [2, 1, 5, 6])
        self.assertEqual(job.frame_rows()[0]["status"], "approved")
        single.delete_annotation(asset, saved["襯套"]["id"])
        after = job.frame_rows()[0]
        self.assertEqual(after["status"], "pending")
        self.assertEqual(after["boxes"], [])
        self.assertEqual(job.inference, report)

    def test_confirmed_empty_overrides_ai_and_manual_only_training_keeps_scope(self):
        first, second, third, _ = (item["id"] for item in self.project.assets)
        self.project.add_label("襯套")
        self.project.add_box(first, "襯套", _box()["xyxy"])
        self.project.set_reviewed(first, True)
        self.project.set_reviewed(second, True)
        self.project.add_box(third, "襯套", _box()["xyxy"])
        batch = create_job(self.project, "襯套")
        report = _predictions(batch, [([_box()], [])] * 4)
        batch.publish_inference(_write(self.root / "model.json", report))
        self.assertEqual((batch.frame_rows()[1]["status"], batch.frame_rows()[1]["boxes"]), ("approved", []))
        self.assertEqual(batch.frame_rows()[2]["status"], "pending")

        manual = create_manual_training_job(self.project, "襯套")
        self.assertIsNone(manual.inference)
        self.assertEqual({row["id"] for row in manual.frame_rows()}, {first, second})
        dataset = manual.export_dataset()
        self.assertTrue((dataset / "labels" / f"{first}.txt").read_bytes())
        self.assertEqual((dataset / "labels" / f"{second}.txt").read_bytes(), b"")
        self.project.add_label("新類別")
        with self.assertRaisesRegex(ValueError, "所有類別"):
            create_manual_training_job(self.project, "襯套\n新類別")


if __name__ == "__main__":
    unittest.main()
