"""Regressions for native rectangles and compatible detection exports."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from project import Project


class BoxAnnotationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = Project.create(self.root / "project")
        self.paths = []
        for i in range(3):
            path = self.root / f"sample-{i}.png"
            Image.new("RGB", (137, 73), (10 + i, 20, 30)).save(path)
            self.paths.append(path)
        self.project.add_images(self.paths)
        self.asset_ids = [asset["id"] for asset in self.project.assets]
        self.project.add_label("汽車")
        self.project.add_label("人")

    def test_box_reopens_and_edit_keeps_id_and_invalidates_review(self):
        asset_id = self.asset_ids[0]
        annotation_id = self.project.add_box(asset_id, "汽車", (15, 10, 80, 60))
        self.project.set_reviewed(asset_id, True)
        reopened = Project.open(self.project.directory)
        original = reopened.assets[0]["annotations"][0]
        self.assertEqual(original["id"], annotation_id)
        self.assertEqual(original["bbox"], [15, 10, 65, 50])
        self.assertNotIn("counts", original)
        self.assertTrue(reopened.assets[0]["reviewed"])
        reopened.replace_box(asset_id, annotation_id, (15, 10, 80, 60), expected_bbox=original["bbox"])
        self.assertTrue(Project.open(self.project.directory).assets[0]["reviewed"])

        self.assertEqual(reopened.replace_box(asset_id, annotation_id, (14, 9, 137, 73),
                                             expected_bbox=original["bbox"]), annotation_id)
        persisted = Project.open(self.project.directory).assets[0]
        self.assertEqual(len(persisted["annotations"]), 1)
        self.assertEqual(persisted["annotations"][0]["bbox"], [14, 9, 123, 64])
        self.assertFalse(persisted["reviewed"])
        with self.assertRaises(ValueError):
            self.project.replace_box(asset_id, annotation_id, (5, 2, 30, 20),
                                     expected_bbox=original["bbox"])
        self.assertEqual(Project.open(self.project.directory).assets[0], persisted)

    def test_invalid_box_keeps_existing_annotations(self):
        asset_id = self.asset_ids[0]
        self.project.add_box(asset_id, "汽車", (2, 2, 19, 22))
        before = (self.project.directory / "project.json").read_bytes()
        for points in ((0, 3, 0, 16), (0, 2, float("nan"), 16), (-9, 2, -5, 15)):
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.project.add_box(asset_id, "汽車", points)
        self.assertEqual((self.project.directory / "project.json").read_bytes(), before)

    def test_mixed_legacy_mask_and_box_export_without_inventing_labels(self):
        labeled, negative, unreviewed = self.asset_ids
        self.project.add_box(labeled, "汽車", (12, 11, 50, 29))
        mask = np.zeros((73, 137), dtype=bool)
        mask[40:53, 90:117] = True
        mask[43, 95] = False
        self.project.add_annotation(labeled, "人", mask, "sam2")
        self.project.set_reviewed(negative, True)
        manifest = self.project.directory / "project.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        # Version 1 projects saved RLE by counts/size, without an explicit geometry kind.
        old = data["assets"][0]["annotations"][1]
        old_manifest_fields = {"id", "label", "source", "score", "size", "counts", "bbox", "area"}
        data["assets"][0]["annotations"][1] = {key: value for key, value in old.items()
                                                if key in old_manifest_fields}
        manifest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        reopened = Project.open(self.project.directory)
        np.testing.assert_array_equal(reopened.decode_mask(reopened.assets[0]["annotations"][1]), mask)

        classes_file = reopened.export_yolo(self.root / "yolo")
        self.assertEqual(classes_file.relative_to(self.root).as_posix(), "yolo/classes.txt")
        self.assertEqual(classes_file.read_text(encoding="utf-8").splitlines(), ["汽車", "人"])
        images, labels = classes_file.parent / "images", classes_file.parent / "labels"
        self.assertEqual({path.stem for path in images.iterdir()}, {labeled, negative})
        self.assertEqual({path.stem for path in labels.iterdir()}, {labeled, negative})
        self.assertFalse((images / f"{unreviewed}.png").exists())
        self.assertEqual((labels / f"{negative}.txt").read_text(encoding="utf-8").strip(), "")
        rows = [line.split() for line in (labels / f"{labeled}.txt").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        by_label = {row[0]: tuple(map(float, row[1:])) for row in rows}
        np.testing.assert_allclose(by_label["0"], (31 / 137, 20 / 73, 38 / 137, 18 / 73), atol=1e-6)
        np.testing.assert_allclose(by_label["1"], (103.5 / 137, 46.5 / 73, 27 / 137, 13 / 73), atol=1e-6)

        coco_file = reopened.export_coco(self.root / "coco")
        coco = json.loads(coco_file.read_text(encoding="utf-8"))
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(len(coco["annotations"]), 2)
        by_category = {annotation["category_id"]: annotation for annotation in coco["annotations"]}
        self.assertEqual(by_category[1]["bbox"], [12, 11, 38, 18])
        self.assertNotIn("segmentation", by_category[1])
        self.assertEqual(by_category[2]["segmentation"]["counts"], old["counts"])


if __name__ == "__main__":
    unittest.main()
