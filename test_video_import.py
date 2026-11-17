"""Regressions for review-gated video annotations using synthetic staged PNGs."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from project import Project
from video_worker import frame_annotations


VIDEO_SHA256 = hashlib.sha256(b"synthetic source identity").hexdigest()
MODEL_SHA256 = hashlib.sha256(b"synthetic weight identity").hexdigest()


class VideoCandidateTests(unittest.TestCase):
    def test_untracked_detection_survives_alongside_tracked_box(self):
        boxes = SimpleNamespace(xyxy=np.array([[0, 0, 4, 4], [8, 6, 12, 10], [13, 7, 20, 16]]),
                                conf=np.array([0.1, 0.8, 0.6]), cls=np.array([0, 1, 0]))
        tracks = [[7, 5, 13, 11, 21, 0.8, 1, 1]]

        candidates = frame_annotations(boxes, {0: "open", 1: "close"}, tracks, 40, 30, 0.25)
        self.assertEqual([(box["label"], box["track_id"], box["bbox"]) for box in candidates],
                         [("close", 21, [8, 6, 4, 4]), ("open", None, [13, 7, 7, 9])])


class VideoImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = Project.create(self.root / "project")
        self.result = self.root / "worker" / "result.json"
        (self.result.parent / "frames").mkdir(parents=True)
        frames = []
        for index, time_ms in ((0, 0.0), (5, 200.0)):
            relative_path = f"frames/{index:09d}.png"
            image_file = self.result.parent / relative_path
            Image.new("RGB", (80, 50), (index, 10, 30)).save(image_file)
            frames.append({
                "frame_index": index,
                "time_ms": time_ms,
                "width": 80,
                "height": 50,
                "file": relative_path,
                "sha256": hashlib.sha256(image_file.read_bytes()).hexdigest(),
                "boxes": [],
            })
        frames[0]["boxes"].append({
            "bbox": [4, 8, 24, 32], "label": "握把", "class_id": 3,
            "track_id": 21, "confidence": 0.85,
        })
        self.manifest = {
            "schema_version": 1,
            "source": {"path": "synthetic-sample.mp4", "sha256": VIDEO_SHA256,
                       "width": 80, "height": 50, "fps": 25.0, "frame_count": 10},
            "model": {"path": "synthetic-model.pt", "sha256": MODEL_SHA256},
            "sample_fps": 5.0,
            "conf": 0.3,
            "frames": frames,
        }
        self.write_manifest()

    def write_manifest(self):
        self.result.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")

    def import_result(self, video_sha=VIDEO_SHA256, model_sha=MODEL_SHA256):
        return self.project.import_video_annotations(self.result, approved_video_sha256=video_sha,
                                                     approved_model_sha256=model_sha)

    def snapshot(self):
        files = self.project.directory.rglob("*")
        return {file.relative_to(self.project.directory).as_posix(): file.read_bytes()
                for file in files if file.is_file()}

    def test_mismatched_approval_rejects_video_or_model_without_writes(self):
        before = self.snapshot()
        for source, model in (("0" * 64, MODEL_SHA256), (VIDEO_SHA256, "0" * 64)):
            with self.subTest(source=source, model=model), self.assertRaises(ValueError):
                self.import_result(source, model)
            self.assertEqual(self.snapshot(), before)

    def test_staging_rolls_back_if_later_frame_is_tampered_or_escapes_directory(self):
        self.project.add_label("保留原類別")
        before = self.snapshot()
        frame = self.manifest["frames"][1]
        frame_file = self.result.parent / frame["file"]
        original = frame_file.read_bytes()
        Image.new("RGB", (80, 50), "white").save(frame_file)
        with self.assertRaises(ValueError):
            self.import_result()
        self.assertEqual(self.snapshot(), before)
        frame_file.write_bytes(original)

        frame["file"] = "../escape.png"
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.import_result()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.project.labels, ["保留原類別"])
        self.assertEqual(self.project.assets, [])

    def test_review_gates_export_for_detections_and_empty_frames(self):
        self.assertEqual(self.import_result(), {"imported_frames": 2, "added_boxes": 1, "skipped_frames": 0})
        project = Project.open(self.project.directory)
        detected, empty = project.assets
        self.assertFalse(detected["reviewed"])
        self.assertFalse(empty["reviewed"])
        self.assertEqual(empty["annotations"], [])
        self.assertEqual(detected["video"]["sha256"], empty["video"]["sha256"])
        self.assertEqual(detected["video"]["sha256"], VIDEO_SHA256)
        self.assertEqual(detected["video"]["model"]["sha256"], MODEL_SHA256)
        self.assertEqual(empty["video"]["frame_index"], 5)
        self.assertEqual(empty["video"]["time_ms"], 200.0)
        annotation = detected["annotations"][0]
        self.assertEqual(annotation["track_id"], 21)
        self.assertEqual(annotation["model_class_id"], 3)
        self.assertEqual(annotation["bbox"], [4, 8, 24, 32])
        self.assertEqual(annotation["score"], 0.85)
        for name, export in (("yolo", project.export_yolo), ("coco", project.export_coco)):
            with self.subTest(format=name), self.assertRaises(ValueError):
                export(self.root / f"{name}-pending")
            self.assertFalse((self.root / f"{name}-pending").exists())

        project.set_reviewed(detected["id"], True)
        classes = project.export_yolo(self.root / "yolo-reviewed")
        labels = classes.parent / "labels"
        self.assertEqual(classes.read_text(encoding="utf-8").splitlines(), ["握把"])
        self.assertEqual({file.stem for file in labels.iterdir()}, {detected["id"]})
        self.assertEqual([float(value) for value in next(labels.iterdir()).read_text().split()],
                         [0.0, 0.2, 0.48, 0.3, 0.64])

        project.set_reviewed(empty["id"], True)
        classes = project.export_yolo(self.root / "yolo-reviewed-empty")
        self.assertEqual((classes.parent / "labels" / f"{empty['id']}.txt").read_text(), "")
        coco = json.loads(project.export_coco(self.root / "coco-reviewed").read_text(encoding="utf-8"))
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(len(coco["annotations"]), 1)
        self.assertEqual(coco["annotations"][0]["bbox"], [4, 8, 24, 32])

    def test_repeat_import_preserves_user_review_and_existing_frames(self):
        self.import_result()
        self.project.set_reviewed(self.project.assets[0]["id"], True)
        before = self.snapshot()
        self.assertEqual(self.import_result(), {"imported_frames": 0, "added_boxes": 0, "skipped_frames": 2})
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.project.assets), 2)
        self.assertTrue(self.project.assets[0]["reviewed"])


if __name__ == "__main__":
    unittest.main()
