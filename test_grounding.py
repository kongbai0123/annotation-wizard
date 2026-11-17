from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from project import Project
from grounding import GroundingEngine


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
            original = (10, 20, 40, 60)
            identifier = project.add_box(asset["id"], label, original, suggestion=("vehicle door", 5, 0.42),
                                         original_box=original)
            restored = Project.open(project.directory)
            self.assertEqual(restored.grounding_prompt, "vehicle door")
            record = restored.assets[0]["annotations"][0]
            self.assertEqual(record["source"], "grounding_dino")
            self.assertEqual(record["suggestion"], {"prompt": "vehicle door", "rank": 5,
                                                    "original_xyxy": list(original), "adjusted": False})
            self.assertEqual(record["bbox"], [10, 20, 30, 40])
            self.assertEqual(record["score"], 0.42)
            self.assertFalse(restored.assets[0]["reviewed"])
            restored.replace_box(asset["id"], identifier, (11, 20, 40, 60))
            edited = Project.open(project.directory).assets[0]["annotations"][0]
            self.assertTrue(edited["suggestion"]["adjusted"])
            self.assertEqual(edited["suggestion"]["original_xyxy"], list(original))
            self.assertEqual(edited["score"], 0.42)

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

    def test_failed_model_health_check_is_not_retried(self):
        engine = GroundingEngine()
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch}), patch("grounding.load_model", return_value=(object(), object())) as load, \
                patch.object(engine, "_check_examples", side_effect=ValueError("bad fixture")):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "bad fixture"):
                    engine.ensure_loaded()
        load.assert_called_once()
        self.assertIsNone(engine.model)
        self.assertIsNotNone(engine.load_error)
