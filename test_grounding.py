from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from project import Project


class GroundingProjectTests(unittest.TestCase):
    def test_saved_candidate_records_manual_acceptance_and_rank(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "door.png"
            Image.new("RGB", (100, 100)).save(photo)
            project = Project.create(root / "project")
            project.add_images([photo])
            label = project.add_label("open")
            self.assertEqual(project.grounding_prompt, "car door")
            project.set_grounding_prompt("  vehicle  door. ")
            asset = project.assets[0]
            self.assertEqual(asset["annotations"], [])
            project.add_box(asset["id"], label, (10, 20, 40, 60), suggestion=("vehicle door", 5, 0.42))
            restored = Project.open(project.directory)
            self.assertEqual(restored.grounding_prompt, "vehicle door")
            record = restored.assets[0]["annotations"][0]
            self.assertEqual(record["source"], "grounding_dino")
            self.assertEqual(record["suggestion"], {"prompt": "vehicle door", "rank": 5})
            self.assertEqual(record["bbox"], [10, 20, 30, 40])
            self.assertEqual(record["score"], 0.42)
            self.assertFalse(restored.assets[0]["reviewed"])

    def test_invalid_candidate_rank_does_not_write(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "door.png"
            Image.new("RGB", (100, 100)).save(photo)
            project = Project.create(root / "project")
            project.add_images([photo])
            label = project.add_label("open")
            asset = project.assets[0]
            with self.assertRaises(ValueError):
                project.add_box(asset["id"], label, (10, 20, 40, 60), suggestion=("car door", 6, 0.42))
            self.assertEqual(Project.open(project.directory).assets[0]["annotations"], [])
