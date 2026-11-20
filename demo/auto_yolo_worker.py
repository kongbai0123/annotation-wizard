"""Train YOLO26n-P2 from random initialization on a local AI export."""

import argparse
from collections import Counter
import csv
from hashlib import file_digest
import json
from math import isfinite
import os
from pathlib import Path
import sys
from time import monotonic


MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".torchscript", ".gguf"}
NETWORK_EVENTS = {
    "socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto",
    "urllib.Request", "http.client.connect", "ftplib.connect",
}
SCHEMA = "annotation_wizard.yolo_scratch_smoke.v1"
PROJECT_SPLIT = "explicit_project_split"


def sha256(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def write_json(destination, report):
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def export_path(root, relative, directory):
    if not isinstance(relative, str) or "\n" in relative or "\r" in relative or Path(relative).is_absolute():
        raise ValueError("Each export path must be local and relative.")
    local = (root / relative).resolve(strict=True)
    if not local.is_relative_to(root / directory) or not local.is_file():
        raise ValueError(f"File outside the export's {directory} directory: {relative}")
    return local


def read_dataset(root, *, role=None):
    manifest = root / "dataset.json"
    inventory = json.loads(manifest.read_bytes())
    split_policy = inventory.get("split_policy", "by_shot_smoke")
    if split_policy not in ("by_shot_smoke", "training_only_single_recording", PROJECT_SPLIT):
        raise ValueError("Unsupported YOLO split policy.")
    labels = inventory["labels"]
    if (not isinstance(labels, list) or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("Export must give unique, nonempty labels.")
    frames = []
    files, image_digests, annotation_paths = set(), set(), set()
    if split_policy == PROJECT_SPLIT:
        groups, digests = {}, {}
        for raw in inventory["frames"]:
            split = raw.get("split")
            group, image_hash = raw.get("shot"), raw.get("image_sha256")
            if (split not in {"train", "test"} or not isinstance(group, str) or not group.strip()
                    or not isinstance(image_hash, str) or image_hash in digests
                    or groups.setdefault(group, split) != split):
                raise ValueError("Project train and test must have disjoint source groups and image digests.")
            digests[image_hash] = split
        if role == "test" and "test" in groups.values():
            reference = inventory.get("independent_test")
            if not isinstance(reference, dict) or not reference.get("provenance") or not reference.get("source_sha256"):
                raise ValueError("Independent test images require a reference provenance receipt.")
    for raw in inventory["frames"]:
        if split_policy == PROJECT_SPLIT and role is not None and raw["split"] != role:
            continue
        image = export_path(root, raw["image"], "images")
        annotation = export_path(root, raw["label"], "labels")
        expected = root / "labels" / image.relative_to(root / "images").with_suffix(".txt")
        if (annotation != expected or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}
                or "\n" in str(image) or "\r" in str(image)):
            raise ValueError(f"Image and YOLO annotation do not pair: {image.name}")
        image_digest, label_digest = sha256(image), sha256(annotation)
        if (image_digest != raw["image_sha256"] or label_digest != raw["label_sha256"]
                or image in files or image_digest in image_digests or annotation in annotation_paths):
            raise ValueError(f"Duplicate input or mismatched hash: {image.name}")
        files.add(image)
        image_digests.add(image_digest)
        annotation_paths.add(annotation)
        shot = raw["shot"]
        if not isinstance(shot, str) or not shot.strip():
            raise ValueError(f"Missing source shot: {image.name}")
        counts = Counter()
        for line in annotation.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            category, xc, yc, width, height = line.split()
            class_id = int(category)
            x, y, w, h = map(float, (xc, yc, width, height))
            if (not 0 <= class_id < len(labels) or not all(map(isfinite, (x, y, w, h)))
                    or min(x, y) < 0 or max(x, y) > 1 or min(w, h) <= 0 or max(w, h) > 1
                    or min(x - w / 2, y - h / 2) < -1e-5 or max(x + w / 2, y + h / 2) > 1 + 1e-5):
                raise ValueError(f"Invalid normalized label: {annotation.name}")
            counts[class_id] += 1
        frames.append({"image": image, "annotation": annotation, "shot": shot, "counts": counts})
    return labels, frames, sha256(manifest), split_policy


def count_classes(labels, frames):
    counts = Counter()
    for frame in frames:
        counts.update(frame["counts"])
    return {label: counts[index] for index, label in enumerate(labels)}


def describe_split(root, labels, frames):
    return {
        "images": len(frames),
        "shots": sorted({frame["shot"] for frame in frames}),
        "boxes_by_class": count_classes(labels, frames),
        "image_files": [frame["image"].relative_to(root).as_posix() for frame in frames],
    }


def split_frames(frames):
    shots = sorted(Counter(frame["shot"] for frame in frames).items(),
                   key=lambda item: (abs(item[1] - len(frames) * 0.25), item[0]))
    for shot, _ in shots:
        train = [frame for frame in frames if frame["shot"] != shot]
        validation = [frame for frame in frames if frame["shot"] == shot]
        if any(frame["counts"] for frame in train) and any(frame["counts"] for frame in validation):
            return train, validation
    return frames, []


class SerialPool:
    """Use in-process image loading where AppContainer disallows named pipes."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    @staticmethod
    def imap(func, iterable):
        return map(func, iterable)


def configure_offline(output, audit):
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled")
    for key, folder in {"YOLO_CONFIG_DIR": "settings", "MPLCONFIGDIR": "matplotlib"}.items():
        directory = output / folder
        directory.mkdir()
        os.environ[key] = str(directory)

    def guard(event, args):
        if event in NETWORK_EVENTS:
            audit["network_events"].append(event)
            raise PermissionError(f"Training is offline: {event}")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                path = Path(os.fsdecode(args[0])).resolve()
            except (OSError, ValueError):
                return
            if path.suffix.lower() in MODEL_SUFFIXES:
                if not path.is_relative_to(output):
                    audit["blocked_external_checkpoints"].append(path.name)
                    raise PermissionError(f"External weight file refused: {path.name}")
                mode = str(args[1]) if len(args) > 1 else ""
                if mode.startswith("r"):
                    audit["local_checkpoint_reads"].append(path.relative_to(output).as_posix())

    sys.addaudithook(guard)


def read_epoch_results(csv_path, publish_metrics):
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    result = []
    for row in rows:
        entry = {}
        for key, value in row.items():
            key = key.strip()
            if key == "epoch" or key.startswith("train/") or (publish_metrics and key.startswith(("metrics/", "val/"))):
                parsed = float(value)
                entry[key] = parsed if isfinite(parsed) else None
        result.append(entry)
    return result


def evaluate_saved_checkpoint(args, checkpoint, labels, validation):
    """Read the newly saved checkpoint and score its visible outputs at fixed thresholds."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix
    from ultralytics.utils.ops import xywh2xyxy

    confidence, match_iou, nms_iou = .25, .50, .70
    predictor = YOLO(str(checkpoint), task="detect", verbose=False)
    paths = [str(frame["image"]) for frame in validation]
    results = predictor.predict(paths, imgsz=args.imgsz, device="0", conf=confidence, iou=nms_iou,
                                agnostic_nms=True, save=False, verbose=False)
    matrix = ConfusionMatrix(dict(enumerate(labels)), task="detect")
    details = []
    for result, frame in zip(results, validation, strict=True):
        if Path(result.path).name != frame["image"].name or result.boxes is None:
            raise RuntimeError("Checkpoint prediction does not match the validation image.")
        truth = torch.tensor([list(map(float, line.split())) for line in
                              frame["annotation"].read_text(encoding="utf-8").splitlines() if line.strip()],
                             dtype=torch.float32).reshape(-1, 5)
        height, width = result.orig_shape
        xyxy = xywh2xyxy(truth[:, 1:5] * torch.tensor([width, height, width, height]))
        boxes = result.boxes.cpu()
        if not torch.isfinite(boxes.data).all():
            raise RuntimeError("Checkpoint output contains nonfinite detection values.")
        matrix.process_batch({"bboxes": boxes.xyxy, "cls": boxes.cls, "conf": boxes.conf},
                             {"bboxes": xyxy, "cls": truth[:, 0]}, conf=confidence, iou_thres=match_iou)
        details.append({"image": frame["image"].relative_to(args.dataset).as_posix(),
                        "shot": frame["shot"], "width": width, "height": height,
                        "reference": [{"label": labels[int(category)], "xyxy": box.tolist()}
                                      for category, box in zip(truth[:, 0], xyxy, strict=True)],
                        "predictions": [{"label": labels[int(category)], "xyxy": box.tolist(), "score": float(score)}
                                        for category, box, score in zip(boxes.cls, boxes.xyxy, boxes.conf, strict=True)]})
    cells = matrix.matrix.astype(int)
    by_class = {}
    for row, name in enumerate(labels):
        tp = int(cells[row, row])
        fp, fn = int(cells[row, :].sum()) - tp, int(cells[:, row].sum()) - tp
        by_class[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / (tp + fn) if tp + fn else None}
    image_report = args.output / "checkpoint-validation-frames.json"
    write_json(image_report, {"frames": details})
    return {"checkpoint": checkpoint.relative_to(args.output).as_posix(), "confidence": confidence,
            "match_iou": match_iou, "nms_iou": nms_iou, "class_agnostic_nms": True,
            "axis_order": "predicted_rows_reference_columns", "labels": [*labels, "background"],
            "matrix": cells.tolist(), "by_class": by_class, "images": len(validation),
            "frames_report": image_report.name, "frames_report_sha256": sha256(image_report)}


def checkpoint_detections(rows, labels, width, height):
    from generic_eval import validate_box

    detections, dropped = [], []
    for label, box, score in rows:
        if not isfinite(label) or int(label) != label or not 0 <= label < len(labels):
            reason = "invalid_class"
        elif not isfinite(score) or not 0 <= score <= 1:
            reason = "invalid_confidence"
        elif len(box) != 4 or not all(map(isfinite, box)):
            reason = "nonfinite_geometry"
        else:
            clipped = [max(0., min(limit, value))
                       for value, limit in zip(box, (width, height, width, height), strict=True)]
            if validate_box(clipped):
                detections.append({"label": labels[int(label)], "xyxy": clipped, "score": score})
                continue
            reason = "zero_area_after_clipping"
        dropped.append({"reason": reason, "class_index": label if isfinite(label) else None,
                        "score": score if isfinite(score) else None,
                        "xyxy": [value if isfinite(value) else None for value in box]})
    return detections, dropped


def evaluate_independent_checkpoint(args, checkpoint, labels, test):
    """Score a fixed checkpoint on independent references opened after training."""
    from ultralytics import YOLO

    from generic_eval import score_detection

    predictor = YOLO(str(checkpoint), task="detect", verbose=False)
    outputs = predictor.predict([str(frame["image"]) for frame in test], imgsz=args.imgsz,
                                device="0", conf=.001, iou=.70, max_det=300,
                                agnostic_nms=False, save=False, verbose=False, stream=True)
    details = []
    for frame, result in zip(test, outputs, strict=True):
        if Path(result.path).resolve() != frame["image"] or result.boxes is None:
            raise RuntimeError("Checkpoint output does not match the independent image.")
        height, width = result.orig_shape
        truth = []
        for line in frame["annotation"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                label, center_x, center_y, extent_x, extent_y = map(float, line.split())
                xyxy = [(center_x - extent_x / 2) * width, (center_y - extent_y / 2) * height,
                        (center_x + extent_x / 2) * width, (center_y + extent_y / 2) * height]
                xyxy = [max(0., min(limit, value)) for value, limit in zip(xyxy, (width, height, width, height), strict=True)]
                truth.append({"label": labels[int(label)], "xyxy": xyxy})
        rows = zip(result.boxes.cls.cpu().tolist(), result.boxes.xyxy.cpu().tolist(),
                   result.boxes.conf.cpu().tolist(), strict=True)
        detections, dropped = checkpoint_detections(rows, labels, width, height)
        image = frame["image"].relative_to(args.dataset).as_posix()
        details.append({"id": frame["image"].stem, "image": image, "shot": frame["shot"], "width": width,
                        "height": height, "reference": truth, "predictions": detections,
                        "invalid_predictions": dropped})
    image_report = args.output / "checkpoint-independent-frames.json"
    metrics = score_detection(details, labels, confidence=.25)
    invalid = [box for frame in details for box in frame["invalid_predictions"]]
    write_json(image_report, {"frames": details})
    return {**metrics, "checkpoint": checkpoint.relative_to(args.output).as_posix(), "confidence": .25,
            "prediction_floor": .001, "nms_iou": .70, "max_detections": 300,
            "invalid_predictions": len(invalid),
            "invalid_predictions_at_operating_threshold": sum(box["score"] is not None and box["score"] >= .25
                                                              for box in invalid),
            "frames_report": image_report.name, "frames_report_sha256": sha256(image_report)}


def train_model(args, labels, train, validation, report, report_path):
    import yaml
    import torch
    from unittest.mock import patch

    path_exists = Path.exists

    def sandbox_exists(path, *options, **kwargs):
        try:
            return path_exists(path, *options, **kwargs)
        except PermissionError:
            if path.name == ".git":
                return False
            raise

    # Ultralytics probes parent Git metadata at import. AppContainer denies it.
    with patch.object(Path, "exists", sandbox_exists):
        import ultralytics
        from ultralytics import YOLO, settings
        from ultralytics.data import base, dataset, utils

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the isolated smoke training.")
    base.ThreadPool = dataset.ThreadPool = SerialPool
    utils.check_font = lambda *args, **kwargs: None
    tracking = ("sync", "wandb", "tensorboard", "comet", "clearml", "mlflow", "dvc", "raytune", "neptune")
    settings.update({key: False for key in tracking if key in settings})

    architecture = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "26" / "yolo26-p2.yaml"
    model_cfg = yaml.safe_load(architecture.read_text(encoding="utf-8"))
    model_cfg.update(nc=len(labels), end2end=False)
    model_yaml = args.output / "yolo26n-p2-project.yaml"
    model_yaml.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    train_list, validation_list = args.output / "train.txt", args.output / "val.txt"
    train_list.write_text("".join(frame["image"].as_posix() + "\n" for frame in train), encoding="utf-8")
    # Ultralytics expects nonempty val even for training-only smoke runs.
    validation_list.write_text("".join(frame["image"].as_posix() + "\n" for frame in validation or train), encoding="utf-8")
    data = args.output / "data.yaml"
    data.write_text(yaml.safe_dump({
        "path": args.output.as_posix(), "train": train_list.name, "val": validation_list.name,
        "names": dict(enumerate(labels)),
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report.update(torch=torch.__version__, ultralytics=ultralytics.__version__, device=torch.cuda.get_device_name(0),
                  architecture_sha256=sha256(model_yaml), data_yaml_sha256=sha256(data))
    torch.manual_seed(args.seed)
    model = YOLO(str(model_yaml), task="detect", verbose=False)

    def on_epoch_end(trainer):
        report["epochs_completed"] = trainer.epoch + 1
        write_json(report_path, report)

    model.add_callback("on_train_epoch_end", on_epoch_end)
    batch_size = min(args.batch, len(train))
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=batch_size, nbs=batch_size,
                device="0", workers=0, pretrained=False, amp=False, resume=False, cache=False, val=bool(validation),
                seed=args.seed, warmup_epochs=0, optimizer="AdamW", close_mosaic=0, mosaic=0.0, plots=False,
                project=str(args.output), name="fit", exist_ok=False, verbose=False)

    checkpoints = {}
    for name in ("best", "last"):
        checkpoint = args.output / "fit" / "weights" / f"{name}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Training did not save the expected {name} checkpoint.")
        checkpoints[name] = {"path": checkpoint.relative_to(args.output).as_posix(), "sha256": sha256(checkpoint)}
    report["checkpoints"] = checkpoints
    report["epoch_results"] = read_epoch_results(args.output / "fit" / "results.csv", bool(validation))
    if args.evaluate_checkpoint and validation:
        report["checkpoint_evaluation"] = evaluate_saved_checkpoint(
            args, args.output / checkpoints["best"]["path"], labels, validation)


def run(args):
    args.dataset = args.dataset.resolve(strict=True)
    args.output = args.output.resolve()
    proof = json.loads(args.isolation.read_bytes())
    if (proof["os_network_capabilities"] != 0 or proof.get("network_denied") is not True
            or proof.get("labels_denied") is not True or proof.get("cuda") is not True
            or not proof.get("profile") or proof["profile"] != os.environ.get("TILE_PROFILE")):
        raise ValueError("Training requires matching local AppContainer isolation evidence.")
    if args.output == args.dataset or args.dataset.is_relative_to(args.output):
        raise ValueError("Training output must not replace the export.")
    labels, frames, manifest_digest, split_policy = read_dataset(args.dataset, role="train")
    generic = split_policy == PROJECT_SPLIT
    train, validation = ((frames, []) if generic or split_policy == "training_only_single_recording" else split_frames(frames))
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "training.json"
    report = {
        "schema": SCHEMA, "status": "running", "scope": "generic_project" if generic else "p03_pipeline_smoke",
        "model": "YOLO26n-P2", "initialization": "random", "pretrained": False, "resume": False,
        "dataset_manifest_sha256": manifest_digest, "isolation_sha256": sha256(args.isolation),
        "isolation_profile": proof["profile"], "labels": labels, "epochs_requested": args.epochs,
        "epochs_completed": 0, "imgsz": args.imgsz, "batch": min(args.batch, len(train)), "seed": args.seed,
        "training": describe_split(args.dataset, labels, train),
        "validation": describe_split(args.dataset, labels, validation),
        "split_policy": split_policy,
        "validation_mode": ("independent_test_after_fit" if generic else
                            "training_only_single_recording" if split_policy == "training_only_single_recording"
                            else "other_shot_ai_labels" if validation else "metrics_unavailable_no_positive_separate_shot"),
        "independent_accuracy_measured": False,
        "audit": {"network_events": [], "blocked_external_checkpoints": [], "local_checkpoint_reads": []},
    }
    report["missing_training_classes"] = [label for label, count in report["training"]["boxes_by_class"].items()
                                          if not count]
    if not any(report["training"]["boxes_by_class"].values()) or (generic and report["missing_training_classes"]):
        report["status"] = "blocked_no_accepted_positive"
        write_json(report_path, report)
        print("YOLO smoke training needs at least one accepted positive label.", flush=True)
        return
    write_json(report_path, report)
    started = monotonic()
    try:
        configure_offline(args.output, report["audit"])
        train_model(args, labels, train, validation, report, report_path)
        if generic:
            checkpoint = args.output / report["checkpoints"]["last"]["path"]
            # Publish the chosen final-epoch checkpoint digest before reading independent image/label bytes.
            report["fixed_checkpoint_before_test"] = report["checkpoints"]["last"]
            write_json(report_path, report)
            test_labels, test, test_manifest_digest, _ = read_dataset(args.dataset, role="test")
            if test_manifest_digest != manifest_digest or test_labels != labels:
                raise ValueError("Dataset changed between fit and independent scoring.")
            if test:
                report["independent_evaluation"] = evaluate_independent_checkpoint(args, checkpoint, labels, test)
                report["independent_accuracy_measured"] = True
            else:
                report["validation_mode"] = "no_independent_reference"
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = round(monotonic() - started, 3)
        write_json(report_path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--isolation", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-checkpoint", action="store_true")
    options = parser.parse_args()
    if options.epochs < 1 or options.batch < 1 or options.imgsz < 32 or options.imgsz % 32 or options.seed < 0:
        parser.error("Expected positive epochs, batch, image size divisible by 32, and a nonnegative seed.")
    run(options)
