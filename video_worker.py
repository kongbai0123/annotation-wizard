"""Create a detached YOLO/ByteTrack review result for one approved local video."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parent
TRACKER_DEPENDENCIES = ROOT / ".video-deps"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv", ".webm", ".mts", ".m2ts"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".ptl", ".safetensors", ".ckpt", ".onnx", ".engine", ".tflite", ".torchscript"}
REMOTE_EVENTS = {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto", "urllib.Request"}
MAX_SAMPLED_FRAMES = 20_000
MAX_FRAME_BYTES = 8 * 1024**3
SHA256 = re.compile(r"[A-Fa-f0-9]{64}\Z")


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=True, allow_nan=False), flush=True)


def local_path(path: Path, *, must_exist: bool) -> Path:
    if str(path).startswith("\\\\"):
        raise ValueError("請使用本機磁碟上的路徑。")
    resolved = path.resolve(strict=must_exist)
    if str(resolved).startswith("\\\\"):
        raise ValueError("請使用本機磁碟上的路徑。")
    return resolved


def verify_approved(path: Path, expected: str, label: str) -> dict:
    if not SHA256.fullmatch(expected):
        raise ValueError(f"{label}的審查識別無效。")
    initial = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    if (initial.st_size, initial.st_mtime_ns, initial.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"{label}在讀取期間有變動，請重新審查。")
    if not hmac.compare_digest(digest, expected.lower()):
        raise ValueError(f"{label}和已確認的版本不同，請重新審查。")
    return {"path": str(path), "sha256": digest, "size_bytes": after.st_size}


def install_local_guard(model_path: Path, output: Path) -> None:
    # Must run before importing Ultralytics. Tracker dependencies are provisioned
    # separately, so processing a video cannot trigger a package/model download.
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", YOLO_VERBOSE="false",
                      YOLO_CONFIG_DIR=str(output / ".ultralytics"), HF_HUB_OFFLINE="1")
    if TRACKER_DEPENDENCIES.is_dir():
        sys.path.insert(0, str(TRACKER_DEPENDENCIES))

    def audit(event, args):
        if event in REMOTE_EVENTS:
            raise PermissionError("影片自動標註只使用本機資料和模型。")
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        candidate = Path(os.fsdecode(args[0]))
        if candidate.suffix.lower() in WEIGHT_EXTENSIONS and candidate.resolve() != model_path:
            raise PermissionError(f"無法讀取未經確認的模型權重：{candidate}")

    sys.addaudithook(audit)


def load_detector_and_tracker(model_path: Path, device: str):
    try:
        import lap  # noqa: F401  # Ultralytics' ByteTrack runtime prerequisite.
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
    except ImportError as error:
        raise RuntimeError("影片標註需要本機 Ultralytics、PyTorch 與 LAP（lap==0.5.13）。請先準備影片推論環境。") from error
    tracker = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    if not tracker.is_file():
        raise RuntimeError("本機 Ultralytics 缺少 ByteTrack 設定檔。")
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and (not torch.cuda.is_available() or int(device) >= torch.cuda.device_count()):
        raise RuntimeError("找不到指定的 CUDA 裝置。")
    model = YOLO(str(model_path), verbose=False)
    if model.task != "detect":
        raise ValueError("請選擇能輸出偵測框的 YOLO Detect .pt 權重。")
    return model, BYTETracker(IterableSimpleNamespace(**YAML.load(tracker))), device


def frame_annotations(boxes, names, tracks, width: int, height: int, minimum_confidence: float) -> list[dict]:
    # ByteTrack rows end with the index of the detector box. New or brief detections
    # may have no confirmed track yet, but they are still candidates for human review.
    track_ids = {int(track[-1]): int(track[4]) for track in tracks}
    annotations = []
    detections = zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True)
    for index, (xyxy, score, class_index) in enumerate(detections):
        if not all(math.isfinite(value) for value in [*xyxy, score, class_index]):
            raise RuntimeError("模型輸出了無效的偵測數值。")
        if score < minimum_confidence:
            continue
        x1, y1, x2, y2 = xyxy
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_index)
        label = str(names[class_id]).strip()
        if not label or not label.isprintable():
            raise ValueError(f"模型類別 {class_id} 名稱無效。")
        annotations.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "label": label, "class_id": class_id,
                            "confidence": score, "track_id": track_ids.get(index)})
    return annotations


def frame_time(reported_msec: float, last_msec: float, fps: float) -> float:
    if last_msec < 0:
        return 0.0
    # Some decoders do not provide presentation timestamps or always return 0.
    if math.isfinite(reported_msec) and reported_msec > last_msec:
        return reported_msec
    return last_msec + 1000.0 / fps


def process_video(cv2, capture, model, tracker, device: str, output: Path,
                  sample_fps: float, confidence: float) -> tuple[dict, list[dict]]:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError("無法讀取可靠的影片幀率。")
    declared = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    total = int(declared) if math.isfinite(declared) and declared > 0 else None
    effective_fps = min(sample_fps, fps)
    if total is not None and math.ceil(total * (effective_fps / fps)) > MAX_SAMPLED_FRAMES:
        raise ValueError("預估取樣超過 20,000 張，請降低取樣率或縮短影片。")
    (output / "frames").mkdir()
    frames = []
    count = 0
    last_msec = -1.0
    next_sample_msec = 0.0
    period = 1000.0 / effective_fps
    staged_bytes = 0
    dimensions = None
    last_progress = -math.inf

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError("影片含有無法解碼成彩色圖片的畫格。")
        height, width = frame.shape[:2]
        if dimensions is None:
            dimensions = (width, height)
        elif dimensions != (width, height):
            raise ValueError("影片中途變更解析度，請先轉成固定解析度。")
        msec = frame_time(capture.get(cv2.CAP_PROP_POS_MSEC), last_msec, fps)
        last_msec = msec
        result = model.predict(source=frame, device=device, conf=min(confidence, 0.1), save=False,
                               verbose=False, project=str(output), name="runtime", exist_ok=True)[0]
        boxes = result.boxes.cpu().numpy()
        tracks = tracker.update(boxes, frame)

        if effective_fps == fps or msec + 0.01 >= next_sample_msec:
            if len(frames) >= MAX_SAMPLED_FRAMES:
                raise ValueError("取樣超過 20,000 張，請降低取樣率或縮短影片。")
            rel_path = f"frames/{count:09d}.png"
            success, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 2])
            if not success:
                raise RuntimeError(f"第 {count + 1} 張畫格無法編碼。")
            pixels = encoded.tobytes()
            staged_bytes += len(pixels)
            if staged_bytes > MAX_FRAME_BYTES:
                raise ValueError("畫格暫存已超過 8 GiB，請降低取樣率或縮短影片。")
            (output / rel_path).write_bytes(pixels)
            frames.append({"file": rel_path, "frame_index": count, "time_ms": round(msec, 3), "width": width,
                           "height": height, "sha256": hashlib.sha256(pixels).hexdigest(),
                           "boxes": frame_annotations(boxes, result.names, tracks, width, height, confidence)})
            next_sample_msec = (math.floor((msec + 0.01) / period) + 1) * period

        count += 1
        tick = time.monotonic()
        if tick - last_progress >= 0.25:
            emit("progress", stage="tracking", frames_read=count, total_frames=total, frames_sampled=len(frames))
            last_progress = tick

    if not frames or dimensions is None:
        raise RuntimeError("影片沒有可用的畫格。")
    if total is not None and count < total - max(2, round(total * 0.01)):
        raise RuntimeError(f"影片提早停止解碼，原估計 {total} 幀，只讀到 {count} 幀。")
    emit("progress", stage="verifying", frames_read=count, total_frames=count, frames_sampled=len(frames))
    info = {"width": dimensions[0], "height": dimensions[1], "fps": fps, "frame_count": count,
            "reported_frame_count": total, "effective_sample_fps": effective_fps}
    return info, frames


def execute(args) -> None:
    if not 0.1 <= args.sample_fps <= 60 or not 0.01 <= args.conf <= 1.0:
        raise ValueError("取樣率需介於 0.1–60 FPS，信心門檻需介於 0.01–1。")
    if args.device not in ("auto", "cpu") and not re.fullmatch(r"[0-9]+", args.device):
        raise ValueError("運算裝置需為 auto、cpu 或 CUDA 裝置編號。")
    video = local_path(args.video, must_exist=True)
    model_path = local_path(args.model, must_exist=True)
    if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("請選擇支援的本機影片檔。")
    if not model_path.is_file() or model_path.suffix.lower() != ".pt":
        raise ValueError("請選擇預先準備好的本機 YOLO .pt 偵測權重。")
    destination = local_path(args.out, must_exist=False)
    emit("progress", stage="verifying", frames_read=0, total_frames=None, frames_sampled=0)
    source = verify_approved(video, args.approved_sha256, "影片")
    model_info = verify_approved(model_path, args.approved_model_sha256, "模型")

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("請使用新的空白暫存資料夾。")
    install_local_guard(model_path, destination)
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError("此影片無法開啟，請檢查檔案格式和編碼。")
        model, tracker, device = load_detector_and_tracker(model_path, args.device)
        info, frames = process_video(cv2, capture, model, tracker, device, destination, args.sample_fps, args.conf)
    finally:
        capture.release()

    # Never publish a completed result if the approved files changed mid-run.
    verify_approved(video, args.approved_sha256, "影片")
    verify_approved(model_path, args.approved_model_sha256, "模型")
    manifest = {"schema_version": 1, "source": {**source, **info}, "model": model_info, "tracker": "bytetrack",
                "sample_fps": args.sample_fps, "conf": args.conf, "frames": frames}
    temporary = destination / ".result.json.tmp"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination / "result.json")
    emit("complete", manifest="result.json", frames_read=info["frame_count"], frames_sampled=len(frames))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--approved-sha256", "--approved-video-sha256", dest="approved_sha256", required=True)
    parser.add_argument("--approved-model-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        execute(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        emit("error", message=str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
