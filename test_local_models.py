"""Live regression checks which run when the pinned local models are installed."""

import unittest

import numpy as np

from fill_worker import VideoModel, fill_segment
from grounding import FILES, HEALTH_EXAMPLES, GroundingEngine, overlap
from prepare_model import FILES as SAM_FILES, MODEL_DIR as SAM_DIR, verified


class InstalledModelTests(unittest.TestCase):
    def test_installed_dino_retains_all_decoder_heads_and_passes_health_check(self):
        engine = GroundingEngine()
        required = [engine.directory / name for name in FILES]
        required.extend(engine.directory / "health" / row["file"] for row in HEALTH_EXAMPLES)
        if not all(path.is_file() for path in required):
            self.skipTest("MM Grounding DINO and its health examples are optional local files")
        import torch

        engine.ensure_loaded()
        for name in ("bbox_embed", "class_embed"):
            heads = getattr(engine.model, name)
            self.assertIs(heads, getattr(engine.model.model.decoder, name))
            self.assertEqual(len({id(head) for head in heads}), 6)
            first = heads[0].state_dict()
            for head in heads[1:]:
                for key, tensor in head.state_dict().items():
                    self.assertFalse(torch.equal(tensor, first[key]), f"{name}.{key} must differ from layer 0")

    def test_installed_sam_can_blend_a_short_non_square_clip(self):
        if not all((SAM_DIR / name).is_file() for name in SAM_FILES):
            self.skipTest("SAM 2.1 weights are optional local files")
        self.assertTrue(all(verified(SAM_DIR / name) for name in SAM_FILES))
        model = VideoModel(SAM_DIR)
        images = []
        for x in (90, 94, 98):
            frame = np.full((240, 320, 3), 30, dtype=np.uint8)
            frame[50:191, x:x + 131] = (235, 190, 25)
            frame[61:83, x + 15:x + 110] = (137, 89, 27)
            images.append(frame)
        ref = [94, 50, 225, 191]
        segment = {"left": {"boxes": {"door": {"label": "open", "xyxy": [90, 50, 221, 191]}}},
                   "right": {"boxes": {"door": {"label": "open", "xyxy": [98, 50, 229, 191]}}}}
        result = list(fill_segment(model, images, segment, [1]))
        self.assertEqual(len(result), 1)
        self.assertGreater(overlap(result[0][1][0]["xyxy"], ref), 0.5)
