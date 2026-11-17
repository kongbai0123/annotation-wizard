"""Create SAM requests from explicit, reviewed video anchors."""

import json
from pathlib import Path

from video_dialog import ROOT, StagedVideoJob, console_python


def video_frames(assets, selected):
    video = selected.get("video")
    if not video:
        raise ValueError("請先選擇已確認影片的畫格。")
    frames = [a for a in assets if (a.get("video") or {}).get("sha256") == video["sha256"]]
    return sorted(frames, key=lambda a: a["video"]["frame_index"])


def anchor(asset):
    annotations = asset["annotations"]
    if not asset.get("reviewed") or any(a.get("type") != "bbox" or not a.get("video_track")
                                       for a in annotations):
        return None
    boxes = {}
    for row in annotations:
        track = row["video_track"]
        if track in boxes:
            raise ValueError("同一畫格中的物件名稱重複。")
        x, y, w, h = row["bbox"]
        boxes[track] = {"id": row["id"], "label": row["label"], "xyxy": [x, y, x + w, y + h]}
    return {"asset_id": asset["id"], "index": asset["video"]["frame_index"],
            "time_ms": asset["video"]["time_ms"], "boxes": boxes}


def plan_fill(assets, selected, sample_fps, *, preview=False):
    frames = video_frames(assets, selected)
    keyframes = [a for a in frames if a["video"].get("keyframe") or
                 (not a["video"].get("sam_fill") and a.get("reviewed")
                  and any(b.get("video_track") for b in a["annotations"]))]
    source = selected["video"]
    cuts = [a["video"]["frame_index"] for a in frames if a["video"].get("cut_before")]
    existing = [a["video"]["frame_index"] for a in frames]
    present = set(existing)
    plan = {"video": {k: source[k] for k in ("path", "sha256", "fps")}, "width": selected["width"],
            "height": selected["height"], "sample_fps": min(float(sample_fps), source["fps"]),
            "preview": preview, "segments": [], "skipped": 0, "existing": existing}
    if preview:
        current = source["frame_index"]
        left = next((a for a in reversed(keyframes) if a["video"]["frame_index"] < current), None)
        first = anchor(left) if left is not None else None
        if first and first["boxes"] and not any(first["index"] < i <= current for i in cuts):
            plan["segments"].append({"left": first, "right": {"index": current, "asset_id": selected["id"]}})
        return plan

    for first_asset, last_asset in zip(keyframes, keyframes[1:]):
        start, end = first_asset["video"]["frame_index"], last_asset["video"]["frame_index"]
        if end - start < 2:
            continue
        first, last = anchor(first_asset), anchor(last_asset)
        if (first is None or last is None or first["boxes"].keys() != last["boxes"].keys()
                or any(start < i <= end for i in cuts)):
            plan["skipped"] += 1
            continue
        if first["boxes"] and any(i not in present for i in range(start + 1, end)):
            plan["segments"].append({"left": first, "right": last})
    return plan


class VideoFill(StagedVideoJob):
    def __init__(self, plan, model_dir):
        super().__init__()
        self.plan = plan
        self.model_dir = model_dir

    def run(self, project, selected_id):
        def command(directory, out):
            request = directory / "request.json"
            request.write_text(json.dumps(self.plan, ensure_ascii=False), encoding="utf-8")
            return [str(console_python()), str(ROOT / "fill_worker.py"), str(request), str(self.model_dir), str(out)]

        def commit(result):
            if self.plan["preview"]:
                return json.loads(result.read_text(encoding="utf-8"))["suggestions"]
            return project.import_video_fill(result, selected_id, self.plan)

        return self.run_staged(command, commit)
