"""Use local SAM 2.1 to propose or fill boxes between reviewed video anchors."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from prepare_model import FILES, verified
from video_worker import MAX_FRAME_BYTES, MAX_SAMPLED_FRAMES, emit, install_local_guard, local_path, verify_approved


def blend_box(first, last, ratio):
    if first is None or last is None:
        return first if last is None else last
    return [a + (b - a) * ratio for a, b in zip(first, last, strict=True)]


def mask_box(cv2, np, mask, width, height):
    visible = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(visible, connectivity=8)
    if count <= 1:
        return None
    x, y, w, h, area = map(int, max(stats[1:], key=lambda row: row[cv2.CC_STAT_AREA]))
    if area < visible.size * 0.0005:
        return None
    return [x * width / visible.shape[1], y * height / visible.shape[0],
            (x + w) * width / visible.shape[1], (y + h) * height / visible.shape[0]]


class VideoModel:
    def __init__(self, directory):
        import cv2
        import numpy as np
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.cv2, self.np, self.torch = cv2, np, torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda:0" else torch.float32
        self.processor = Sam2VideoProcessor.from_pretrained(directory, local_files_only=True)
        self.model = Sam2VideoModel.from_pretrained(directory, dtype=self.dtype, local_files_only=True,
                                                   use_safetensors=True).to(self.device).eval()

    def propagate(self, images, prompts):
        height, width = images[0].shape[:2]
        tracks = list(prompts)
        ids = list(range(1, len(tracks) + 1))
        result = {track: {} for track in tracks}
        with self.torch.inference_mode():
            session = self.processor.init_video_session(
                video=images, inference_device=self.device, processing_device="cpu",
                video_storage_device="cpu", dtype=self.dtype)
            self.processor.add_inputs_to_inference_session(
                session, frame_idx=0, obj_ids=ids, input_boxes=[[prompts[track]["xyxy"] for track in tracks]])
            for frame in self.model.propagate_in_video_iterator(session, start_frame_idx=0):
                masks = frame.pred_masks[:, 0].float().cpu().numpy()
                logits = frame.object_score_logits.float().reshape(-1).cpu().tolist()
                for obj, logit, mask in zip(frame.object_ids, logits, masks, strict=True):
                    if logit > 0:
                        box = mask_box(self.cv2, self.np, mask, width, height)
                        if box is not None:
                            result[tracks[int(obj) - 1]][frame.frame_idx] = box
        return result


def decode_segment(cv2, capture, start, end, dimensions):
    if (end - start + 1) * math.prod(dimensions) * 3 > 768 * 1024**2:
        raise ValueError("關鍵畫格相距過遠，無法同時載入。請在中間增加已確認畫格。")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, start) or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        raise RuntimeError("無法定位原片的關鍵畫格。")
    images = []
    for index in range(start, end + 1):
        ok, frame = capture.read()
        if not ok or frame is None or frame.shape[:2] != dimensions[::-1]:
            raise RuntimeError(f"無法解碼原片第 {index} 格，或影片尺寸已變更。")
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return images


def output_offsets(gap, fps, target_fps):
    return sorted({round(i * fps / target_fps) for i in range(1, math.ceil(gap * target_fps / fps))
                   if 0 < round(i * fps / target_fps) < gap})


def fill_segment(model, images, segment, offsets):
    left, right = segment["left"]["boxes"], segment["right"]["boxes"]
    first = model.propagate(images, left)
    last = model.propagate(images[::-1], right)
    gap = len(images) - 1
    for offset in offsets:
        ratio = offset / gap
        boxes = []
        for track in left:
            xyxy = blend_box(first[track].get(offset), last[track].get(gap - offset), ratio)
            if xyxy is None:
                break
            a, b = left[track]["label"], right[track]["label"]
            boxes.append({"track": track, "xyxy": xyxy, "label": a if ratio <= 0.5 else b,
                          "needs_label_review": a != b})
        if len(boxes) == len(left):
            yield offset, boxes


def run_fill(plan, model_dir, output):
    source = plan["video"]
    video = local_path(Path(source["path"]), must_exist=True)
    verify_approved(video, source["sha256"], "影片")
    model_dir = local_path(model_dir, must_exist=True)
    if not all(verified(model_dir / name) for name in FILES):
        raise ValueError("本機 SAM 2.1 權重或設定驗證失敗。")
    output.mkdir()
    (output / "frames").mkdir()
    os.environ.update(TRANSFORMERS_OFFLINE="1", USE_HUB_KERNELS="NO")
    install_local_guard(model_dir / "model.safetensors", output)
    import cv2

    model = VideoModel(model_dir)
    capture = cv2.VideoCapture(str(video))
    result = {"source_sha256": source["sha256"], "frames": [], "missed": 0, "suggestions": []}
    existing = set(plan["existing"])
    staged_bytes = 0
    try:
        if not capture.isOpened() or not math.isclose(capture.get(cv2.CAP_PROP_FPS), source["fps"]):
            raise ValueError("無法開啟核對過的影片，或影片幀率與專案不符。")
        for index, segment in enumerate(plan["segments"]):
            left, right = segment["left"], segment["right"]
            gap = right["index"] - left["index"]
            emit("progress", stage="sam_fill", frames_read=index, total_frames=len(plan["segments"]),
                 frames_sampled=len(result["frames"]))
            refresh = segment.get("refresh", {})
            offsets = {i for i in output_offsets(gap, source["fps"], plan["sample_fps"])
                       if left["index"] + i not in existing}
            offsets.update(int(i) - left["index"] for i in refresh)
            if not offsets and not plan["preview"]:
                continue
            images = decode_segment(cv2, capture, left["index"], right["index"], (plan["width"], plan["height"]))
            if plan["preview"]:
                paths = model.propagate(images, left["boxes"])
                result["suggestions"] = [{"track": track, "xyxy": paths[track][gap], "label": box["label"]}
                                         for track, box in left["boxes"].items() if gap in paths[track]]
                break
            saved = 0
            for offset, boxes in fill_segment(model, images, segment, sorted(offsets)):
                if len(result["frames"]) >= MAX_SAMPLED_FRAMES:
                    raise ValueError("補框輸出超過 20,000 張，請降低儲存畫格率。")
                frame = left["index"] + offset
                time_ms = left["time_ms"] + (right["time_ms"] - left["time_ms"]) * offset / gap
                if str(frame) in refresh:
                    result["frames"].append({"frame_index": frame, "time_ms": time_ms, "boxes": boxes,
                                             "segment": index})
                    saved += 1
                    continue
                name = f"frames/{frame:09d}.png"
                pixels = cv2.cvtColor(images[offset], cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 2])
                if not ok:
                    raise RuntimeError(f"無法儲存第 {frame} 格。")
                contents = encoded.tobytes()
                staged_bytes += len(contents)
                if staged_bytes > MAX_FRAME_BYTES:
                    raise ValueError("補框暫存超過 8 GiB，請降低儲存畫格率。")
                (output / name).write_bytes(contents)
                result["frames"].append({"frame_index": frame, "time_ms": time_ms, "file": name,
                                         "sha256": hashlib.sha256(contents).hexdigest(), "boxes": boxes,
                                         "segment": index})
                saved += 1
            result["missed"] += len(offsets) - saved
    finally:
        capture.release()
    verify_approved(video, source["sha256"], "影片")
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        return run_fill(json.loads(args.request.read_text(encoding="utf-8")), args.model, args.output)
    except Exception as exc:
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
