"""Regression checks for generic job inputs and model-free inference routing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from generic_inference import DEFAULTS, accepted_candidates, file_hash, finalize_frame, merge_class_drafts, parse_scan, run_inference, select_vote


class GenericInferenceTests(unittest.TestCase):
    def test_dynamic_class_prompt_and_result_are_not_cargo_specific(self):
        calls = []

        class Engine:
            def __init__(self, directory):
                pass

            def propose(self, image, prompt, *, candidate_limit):
                calls.append((prompt, candidate_limit))
                return [((10, 10, 40, 40), 0.6)]

        class Judge:
            def __init__(self, directory, classes, settings):
                self.label = classes[0]["label"]

            def scan(self, image):
                return {"status": "parsed", "targets": [{"label": self.label, "xyxy": [10, 10, 40, 40]}], "raw": "fake"}

            def review(self, image, candidates):
                for row in candidates:
                    row.update(verdict="accepted", reason_code="accepted", votes=[])

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images/tool.png"
            image.parent.mkdir()
            Image.new("RGB", (100, 80)).save(image)
            dino, qwen = root / "dino", root / "qwen"
            dino.mkdir()
            qwen.mkdir()
            (dino / "model.safetensors").write_bytes(b"model replaced by a test double")
            manifest = root / "manifest.json"
            job = {"job_id": "unseen-tool", "config": {"classes": [{"id": "tool", "label": "六角扳手",
                       "prompt": "hex key", "description": "The visible L-shaped hex key, one tool per box."}]},
                   "frames": [{"id": "one", "image": "images/tool.png", "image_sha256": file_hash(image),
                               "width": 100, "height": 80}]}
            manifest.write_text(json.dumps(job), encoding="utf-8")
            output = root / "output/result.json"
            result = run_inference(manifest, dino, qwen, output, engine_factory=Engine, judge_factory=Judge)
            self.assertEqual(calls, [("hex key", 5)])
            self.assertEqual(result["job_id"], job["job_id"])
            self.assertEqual(result["manifest_sha256"], file_hash(manifest))
            self.assertEqual(result["frames"][0]["boxes"][0]["label"], "六角扳手")
            self.assertEqual(result["frames"][0]["status"], "ready")
            self.assertFalse(result["frames"][0]["boxes"][0]["reviewed"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["summary"]["accepted_boxes"], 1)
            output.unlink()
            image.write_bytes(b"tampered image")
            with self.assertRaisesRegex(ValueError, "影像在確認後已變更"):
                run_inference(manifest, dino, qwen, output, engine_factory=Engine, judge_factory=Judge)
            self.assertFalse(output.exists())

    def test_low_agreement_and_class_conflicts_require_review(self):
        answer = select_vote([([0.8, 0.1, 0.05, 0.05], 0.8), ([0.1, 0.8, 0.05, 0.05], 0.8)], DEFAULTS)
        self.assertEqual(answer["verdict"], "unresolved")
        rows = [{"id": "A", "label": "tool", "xyxy": [10, 10, 50, 50], "score": .8,
                 "verdict": "accepted", "reason_code": "accepted"},
                {"id": "B", "label": "handle", "xyxy": [12, 12, 50, 50], "score": .9,
                 "verdict": "accepted", "reason_code": "accepted"},
                {"id": "C", "label": "tool", "xyxy": [60, 10, 90, 40], "score": .7,
                 "verdict": "accepted", "reason_code": "accepted"}]
        accepted = accepted_candidates(rows)
        self.assertEqual([r["id"] for r in accepted], ["C"])
        self.assertTrue(all(r["reason_code"] == "conflicting_classes" for r in rows[:2]))

    def test_empty_and_invalid_scene_scans_are_not_silent_negatives(self):
        image = Image.new("RGB", (100, 80))
        scan = parse_scan('{"uncertain":false,"targets":[{"label":"new class","box":[10,20,50,60]}]}', image, {"new class"})
        self.assertEqual(scan["targets"], [{"label": "new class", "xyxy": [1, 1.6, 5, 4.8]}])
        invalid = parse_scan('{"uncertain":false,"targets":[{"label":"old class","box":[10,20,50,60]}]}', image, {"new class"})
        self.assertEqual(invalid["status"], "invalid")
        row = {"id": "blank", "image_sha256": "0" * 64, "candidates": [], "retries": [], "scene_scan": scan}
        result = finalize_frame(row)
        self.assertEqual(result["status"], "review")
        self.assertEqual({x["code"] for x in result["exceptions"]}, {"unmatched_scene_target", "no_accepted_boxes"})

    def test_competing_extents_use_qwen_support_and_abstain_when_close(self):
        def candidate(ident, box, score, margin):
            vote = {"winner": "A", "margin": margin, "allowed_token_mass": .7}
            return {"id": ident, "label": "tool", "xyxy": box, "score": score,
                    "verdict": "accepted", "reason_code": "accepted", "votes": [dict(vote), dict(vote)]}

        broad = candidate("broad", [0, 0, 100, 100], .9, .3)
        tight = candidate("tight", [6, 6, 94, 94], .4, .6)
        selected = accepted_candidates([broad, tight])
        self.assertEqual([row["id"] for row in selected], ["tight"])
        self.assertEqual(broad["reason_code"], "better_supported_extent")
        self.assertEqual(broad["preferred_candidate_id"], "tight")

        broad = candidate("broad", [0, 0, 100, 100], .9, .5)
        tight = candidate("tight", [6, 6, 94, 94], .4, .54)
        row = {"id": "rivals", "image_sha256": "0" * 64, "candidates": [broad, tight], "retries": [],
               "scene_scan": {"status": "parsed", "targets": [], "raw": "fake"}}
        result = finalize_frame(row)
        self.assertEqual(result["boxes"], [])
        self.assertEqual(result["status"], "review")
        self.assertEqual(sum(e["code"] == "competing_extents" for e in result["exceptions"]), 2)

    def test_missing_target_receives_one_dynamic_local_retry(self):
        calls = []

        class Engine:
            def __init__(self, directory):
                pass

            def propose(self, image, prompt, *, candidate_limit):
                calls.append((image.size, prompt))
                if image.size == (100, 80):
                    return []
                return [((12, 12, image.width - 12, image.height - 12), .6)]

        class Judge:
            def __init__(self, directory, classes, settings):
                self.label = classes[0]["label"]

            def scan(self, image):
                return {"status": "parsed", "targets": [{"label": self.label, "xyxy": [20, 20, 50, 50]}], "raw": "fake"}

            def review(self, image, candidates):
                for row in candidates:
                    row.update(verdict="accepted", reason_code="accepted", votes=[])

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (100, 80)).save(image)
            dino, qwen = root / "dino", root / "qwen"
            dino.mkdir()
            qwen.mkdir()
            (dino / "model.safetensors").write_bytes(b"mock")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"job_id": "retry", "config": {"classes": [
                {"id": "tool", "label": "tool", "prompt": "hex key", "description": "one tool"}]}, "frames": [
                {"id": "one", "image": "image.png", "image_sha256": file_hash(image), "width": 100, "height": 80}]}))
            result = run_inference(manifest, dino, qwen, root / "output.json", engine_factory=Engine, judge_factory=Judge)
            self.assertEqual(calls, [((100, 80), "hex key"), ((54, 54), "hex key")])
            frame = result["frames"][0]
            self.assertEqual(frame["status"], "ready")
            self.assertEqual(frame["boxes"][0]["xyxy"], [20, 20, 50, 50])
            self.assertEqual(frame["boxes"][0]["candidate_id"], "retry-0001")
            self.assertEqual(frame["retries"][0]["label"], "tool")

    def test_class_draft_translates_missing_prompt_and_keeps_user_definitions(self):
        classes = [{"id": 0, "label": "六角扳手", "prompt": "六角扳手", "description": "",
                    "prompt_provided": False, "description_provided": False},
                   {"id": 1, "label": "玻璃瓶", "prompt": "glass bottle", "description": "Only the visible glass body.",
                    "prompt_provided": True, "description_provided": True}]
        draft = {"classes": [{"id": 0, "label": "六角扳手", "prompt": "hex key.", "description": "One visible L-shaped hex key."},
                             {"id": 1, "label": "玻璃瓶", "prompt": "wrong change", "description": "wrong extent"}]}
        effective, provenance = merge_class_drafts(classes, json.dumps(draft))
        self.assertEqual(effective[0]["prompt"], "hex key")
        self.assertEqual(effective[0]["label"], "六角扳手")
        self.assertEqual(effective[1], classes[1])
        self.assertEqual(provenance["classes"][0]["fields"]["prompt"], "qwen_draft")
        fallback, evidence = merge_class_drafts(classes, "unparseable response")
        self.assertEqual(fallback[0]["prompt"], "六角扳手")
        self.assertIn("visible extent", fallback[0]["description"])
        self.assertEqual(fallback[1], classes[1])
        self.assertEqual(evidence["status"], "literal_fallback")

    def test_unicode_class_draft_is_used_before_dino_and_frozen_in_result(self):
        prompts = []

        class Engine:
            def __init__(self, directory):
                pass

            def propose(self, image, prompt, *, candidate_limit):
                prompts.append(prompt)
                return [((10, 10, 40, 40), .7)]

        class Judge:
            def __init__(self, directory, classes, settings):
                self.label = classes[0]["label"]

            def draft_classes(self, classes):
                return json.dumps({"classes": [{"id": 0, "label": "六角扳手", "prompt": "hex key",
                                               "description": "One visible L-shaped hex key."}]})

            def scan(self, image):
                return {"status": "parsed", "targets": [{"label": self.label, "xyxy": [10, 10, 40, 40]}], "raw": "fake"}

            def review(self, image, candidates):
                for row in candidates:
                    row.update(verdict="accepted", reason_code="accepted", votes=[])

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (100, 80)).save(image)
            dino, qwen = root / "dino", root / "qwen"
            dino.mkdir()
            qwen.mkdir()
            (dino / "model.safetensors").write_bytes(b"mock")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"job_id": "unicode", "config": {"classes": [
                {"id": 0, "label": "六角扳手", "prompt": "六角扳手", "description": "",
                 "prompt_provided": False, "description_provided": False}]}, "frames": [
                {"id": "one", "image": "image.png", "image_sha256": file_hash(image), "width": 100, "height": 80}]}))
            result = run_inference(manifest, dino, qwen, root / "output.json", engine_factory=Engine, judge_factory=Judge)
            self.assertEqual(prompts, ["hex key"])
            self.assertEqual(result["method"]["classes"][0]["label"], "六角扳手")
            self.assertEqual(result["method"]["classes"][0]["prompt"], "hex key")
            self.assertEqual(result["method"]["class_preparation"]["classes"][0]["original_prompt"], "六角扳手")


if __name__ == "__main__":
    unittest.main()
