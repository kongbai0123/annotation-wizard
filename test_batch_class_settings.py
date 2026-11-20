"""Exercise project class defaults shared by single and batch annotation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from project import Project


class BatchClassSettingsTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.project = Project.create(self.path)

    def test_existing_projects_share_labels_and_store_detached_batch_defaults(self):
        self.project.add_label("扳手")
        self.assertEqual(Project.open(self.path).batch_class_settings, {})
        self.project.save_batch_class(" 扳手 ", prompt=" wrench ", description="整件工具")
        self.project.save_batch_class("螺絲起子")

        reopened = Project.open(self.path)
        self.assertEqual(reopened.labels, ["扳手", "螺絲起子"])
        self.assertEqual(reopened.batch_class_settings, {
            "扳手": {"prompt": "wrench", "description": "整件工具"},
            "螺絲起子": {"prompt": "", "description": ""},
        })
        detached = reopened.batch_class_settings
        detached["扳手"]["prompt"] = "unused"
        self.assertEqual(reopened.batch_class_settings["扳手"]["prompt"], "wrench")

        reopened.save_batch_class("扳手", prompt="box end wrench", description="框住可見部分")
        self.assertEqual(Project.open(self.path).batch_class_settings["扳手"],
                         {"prompt": "box end wrench", "description": "框住可見部分"})

    def test_conflicting_spellings_and_invalid_text_leave_project_unchanged(self):
        self.project.add_label("Wrench")
        manifest = self.path / "project.json"
        before = manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "重複"):
            self.project.save_batch_class("wrench")
        with self.assertRaisesRegex(ValueError, "單行"):
            self.project.save_batch_class("Wrench", description="line one\nline two")
        with self.assertRaisesRegex(ValueError, "單行"):
            self.project.save_batch_class("other", prompt=1)
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(self.project.labels, ["Wrench"])
