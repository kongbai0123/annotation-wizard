"""Persistent project-scoped annotation jobs and reproducible YOLO exports."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from shutil import copyfile
from tempfile import TemporaryDirectory
from uuid import uuid4

from generic_confidence import frame_confidence
from project import Project, _box_review_revision, _digest, _project_lock, _read_manifest, _write_json


JOBS_DIRECTORY = "ai-jobs"
JOB_SCHEMA = "annotation_wizard.generic_job.v1"
STATE_SCHEMA = "annotation_wizard.generic_job_state.v1"
INFERENCE_SCHEMA = "annotation_wizard.generic_inference.v1"
REFERENCE_SCHEMA = "annotation_wizard.generic_reference.v1"
HEX_ID = re.compile(r"[0-9a-f]{32}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
EMPTY_EXCEPTION = {"code": "empty_unverified", "reason": "整張畫格尚未有人確認為空畫面。"}
MANUAL_PENDING = {"code": "manual_review_pending", "reason": "已保留人工框，請確認本批類別是否全部標好。"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _load(path):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"記錄格式無效：{path.name}。")
    return value


def _checked_file(root, relative):
    path = root / relative
    if path.resolve(strict=True) != path or not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"檔案必須位於核准的資料夾內：{relative}。")
    return path


def _line(value, *, required=True):
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        raise ValueError("類別及來源內容必須是單行文字。")
    value = value.strip()
    if required and not value:
        raise ValueError("類別或來源不可留白。")
    return value


def parse_classes(text):
    """Read one class per line as name, optional MM prompt, optional Qwen description."""
    if not isinstance(text, str):
        raise ValueError("請逐行輸入類別。")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 2)
        rows.append(dict(zip(("label", "prompt", "description"), fields)))
    return _class_config(rows)


def _class_config(classes):
    if isinstance(classes, str):
        return parse_classes(classes)
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("請至少設定一個類別。")
    result, names = [], set()
    for index, entry in enumerate(classes):
        entry = {"label": entry} if isinstance(entry, str) else entry
        if not isinstance(entry, dict):
            raise ValueError("類別設定無效。")
        label = _line(entry.get("label", ""))
        if label.casefold() in names:
            raise ValueError(f"類別名稱重複：{label}。")
        names.add(label.casefold())
        prompt = _line(entry.get("prompt", ""), required=False) or label
        description = _line(entry.get("description", ""), required=False)
        prompt_provided = entry.get("prompt_provided", bool(entry.get("prompt", "").strip()))
        description_provided = entry.get("description_provided", bool(description))
        if type(prompt_provided) is not bool or type(description_provided) is not bool:
            raise ValueError("類別提示詞或判斷規則的來源記錄無效。")
        result.append({"id": index, "label": label, "prompt": prompt, "description": description,
                       "prompt_provided": prompt_provided, "description_provided": description_provided})
    return result


def _labels(manifest):
    return [item["label"] for item in manifest["config"]["classes"]]


def _boxes(rows, frame, labels):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("標註必須是偵測框清單。")
    clean, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get("label") not in labels:
            raise ValueError("偵測框類別必須來自本次任務。")
        box = row.get("xyxy")
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or any(type(x) not in (float, int) or not isfinite(x) for x in box)
                or not (0 <= box[0] < box[2] <= frame["width"] and 0 <= box[1] < box[3] <= frame["height"])):
            raise ValueError("偵測框座標必須位於原始畫格範圍內且具有正面積。")
        key = row["label"], tuple(box)
        if key in seen:
            raise ValueError("畫格含有重複的偵測框。")
        seen.add(key)
        clean.append({"label": row["label"], "xyxy": list(box)})
    return clean


def _manual_coverage(asset, project_labels):
    review = asset.get("bbox_review_by_class", {})
    covered = set(asset.get("bbox_review_labels", project_labels)) if asset.get("reviewed") else set()
    covered.update(label for label, row in review.items() if row["status"] == "approved")
    # A segmentation-only label needs an explicit detection review before it can become a zero-box class.
    masks = {box["label"] for box in asset["annotations"] if box.get("type") != "bbox"}
    detection_scopes = {label for label, row in review.items() if row.get("job_id") and row["status"] == "approved"}
    covered.difference_update(masks - detection_scopes)
    return covered


def _original_xyxy(box):
    x, y, width, height = box["bbox"]
    return [x, y, x + width, y + height]


def _overlaps_human(first, second):
    a, b = first["xyxy"], second["xyxy"]
    shared = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    a_area, b_area = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - shared
    smaller = min(a_area, b_area)
    return (union > 0 and shared / union >= 0.50
            or first["label"] == second["label"] and smaller > 0 and shared / smaller >= 0.90)


def _saved_by_operator(box):
    source = box.get("source")
    return (box.get("human_touched") or source in {"manual", "sam_forward", "blind_arbitration"}
            or source == "grounding_dino" and not (box.get("suggestion") or {}).get("judge"))


def _linked_decision(frame, asset, labels, project_labels, model, legacy):
    """Use confirmed project boxes, then retain only non-conflicting AI proposals for review."""
    if asset is None or asset.get("image_sha256", frame["image_sha256"]) != frame["image_sha256"]:
        return None
    covered = _manual_coverage(asset, project_labels) & labels
    states = {label: row["status"] for label, row in asset.get("bbox_review_by_class", {}).items() if label in labels}
    if "excluded" in states.values():
        return {"status": "excluded", "boxes": [], "box_origin": "manual"}
    stored = (box for box in asset["annotations"] if box.get("type") == "bbox" and box["label"] in labels and
              (box["label"] in covered or _saved_by_operator(box)))
    saved = {(box["label"], tuple(_original_xyxy(box))): {"label": box["label"], "xyxy": _original_xyxy(box)}
             for box in stored}
    human = [{**box, "source": "manual"} for box in _boxes(list(saved.values()), frame, labels)]
    if covered == labels:
        return {"status": "approved", "boxes": human, "box_origin": "manual"}
    if not (covered or human or states):
        return None
    removed = [{"label": box["label"], "xyxy": _original_xyxy(box)} for box in asset.get("bbox_removed", ())
               if box["label"] in labels]
    legacy_approval = legacy.get("status") == "approved"
    offered = legacy["boxes"] if legacy_approval else model.get("boxes", ())
    remaining = [box for box in offered if box["label"] not in covered
                and not any(_overlaps_human(box, entry) for entry in human)
                and not any(box["label"] == entry["label"] and _overlaps_human(box, entry) for entry in removed)]
    if legacy_approval:
        remaining = [{**box, "source": "manual"} for box in remaining]
    return {"status": "pending", "boxes": human + remaining, "box_origin": "mixed",
            "exceptions": [*model.get("exceptions", ()), MANUAL_PENDING]}


def _source_group(asset, groups):
    video = asset.get("video")
    if video and video.get("sha256"):
        return f"video:{video['sha256']}"
    if asset["id"] in groups:
        return _line(groups[asset["id"]])
    return f"image:{asset.get('source_sha256') or asset['image_sha256']}"


def _frame(project_dir, asset, source_group):
    ident = asset["id"]
    relative = f"images/{ident}.png"
    path = _checked_file(project_dir, relative)
    digest = _digest(path)
    if digest != asset.get("image_sha256", digest):
        raise ValueError("專案畫格與匯入記錄的 SHA-256 不符。")
    frame = {key: asset[key] for key in ("id", "name", "width", "height")}
    frame.update(image=relative, image_sha256=digest, source_group=source_group)
    if video := asset.get("video"):
        frame["video"] = {key: video[key] for key in ("sha256", "time_ms", "frame_index") if key in video}
    return frame


def _reference_input(path):
    source = Path(path).read_bytes()
    report = json.loads(source)
    if (not isinstance(report, dict) or report.get("schema") != REFERENCE_SCHEMA
            or not isinstance(report.get("frames"), list) or not report["frames"]):
        raise ValueError("獨立標註檔缺少正確格式或驗收畫格。")
    return report, sha256(source).hexdigest()


def _merge_reference_groups(assets, groups, report):
    by_digest = report.get("source_groups", {})
    if (not isinstance(by_digest, dict)
            or any(not HASH.fullmatch(digest) for digest in by_digest)):
        raise ValueError("驗收標註的來源分組必須使用原始影像 SHA-256。")
    by_digest = {digest: _line(group) for digest, group in by_digest.items()}
    result = dict(groups)
    for ident, asset in assets.items():
        group = by_digest.get(asset.get("source_sha256"))
        if group is not None and not asset.get("video"):
            if ident in result and _line(result[ident]) != group:
                raise ValueError("驗收標註的來源分組和本次任務設定不符。")
            result[ident] = group
    return result


def _reference(report, digest, frames, assets, labels, groups):
    provenance = _line(report.get("provenance", ""))
    by_image, by_source = defaultdict(list), defaultdict(list)
    for row in frames.values():
        by_image[row["image_sha256"]].append(row)
    for ident, asset in assets.items():
        if asset.get("source_sha256"):
            by_source[asset["source_sha256"]].append(frames[ident])
    reference, used = [], set()
    for requested in report["frames"]:
        if not isinstance(requested, dict):
            raise ValueError("獨立標註畫格無效。")
        image_sha, source_sha = requested.get("image_sha256"), requested.get("source_sha256")
        if image_sha is not None:
            if not isinstance(image_sha, str) or not HASH.fullmatch(image_sha):
                raise ValueError("驗收畫格的標準化圖片 SHA-256 無效。")
            found = by_image[image_sha]
        else:
            if not isinstance(source_sha, str) or not HASH.fullmatch(source_sha):
                raise ValueError("驗收畫格需要原始或標準化圖片 SHA-256。")
            found = by_source[source_sha]
        frame = found[0] if len(found) == 1 else None
        if (frame is None or frame["id"] in used
                or source_sha is not None and assets[frame["id"]].get("source_sha256") != source_sha):
            raise ValueError("驗收畫格必須唯一對應已匯入的圖片。")
        group = _line(requested.get("source_group", ""))
        ident = frame["id"]
        # Prefer the canonical recording identity for video frames.
        if frame.get("video"):
            group = frame["source_group"]
        elif ident in groups and frames[ident]["source_group"] != group:
            raise ValueError("驗收影像的來源分組和本次任務設定不符。")
        frame = dict(frame, source_group=group)
        reference.append({"frame": frame, "boxes": _boxes(requested.get("boxes"), frame, labels)})
        used.add(ident)
    return {"schema": REFERENCE_SCHEMA, "source_sha256": digest, "provenance": provenance,
            "frames": reference}


def create_job(project, classes, *, asset_ids=None, source_groups=None, holdout=None, settings=None, manual_only=False):
    """Freeze current project inputs. Independent references stay outside the inference manifest."""
    project_dir = project.directory.resolve(strict=True)
    if (project_dir / "door-ab-blind.json").is_file():
        raise ValueError("封存盲標專案僅供獨立計分。")
    classes = _class_config(classes)
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or any(not isinstance(key, str) for key in settings):
        raise ValueError("任務設定無效。")
    if not manual_only:
        settings = {"shadow_qc": True, **settings}
    assets = {asset["id"]: asset for asset in project.assets}
    selected = set(assets if asset_ids is None else asset_ids)
    groups = {} if source_groups is None else source_groups
    if not isinstance(groups, dict) or not groups.keys() <= assets.keys() or not selected <= assets.keys():
        raise ValueError("任務選擇了專案以外的圖片或來源分組。")
    reference_input = _reference_input(holdout) if holdout is not None else None
    if reference_input is not None:
        groups = _merge_reference_groups(assets, groups, reference_input[0])
    all_frames = {ident: _frame(project_dir, asset, _source_group(asset, groups))
                  for ident, asset in assets.items()
                  if ident in selected or holdout is not None}
    reference = (_reference(*reference_input, all_frames, assets, [row["label"] for row in classes], groups)
                 if reference_input is not None else None)
    test_frames = [] if reference is None else [item["frame"] for item in reference["frames"]]
    test_ids = {row["id"] for row in test_frames}
    test_groups = {row["source_group"] for row in test_frames}
    test_images = {row["image_sha256"] for row in test_frames}
    frames = []
    used_images = set(test_images)
    duplicates = 0
    for ident, row in all_frames.items():
        if ident not in selected or ident in test_ids or row["source_group"] in test_groups:
            continue
        if row["image_sha256"] in used_images:
            duplicates += 1
            continue
        used_images.add(row["image_sha256"])
        frames.append(row)
    if not frames:
        raise ValueError("匯入圖片或影片後，至少保留一張可供自動標註的畫格。")
    if len(test_images) != len(test_frames):
        raise ValueError("獨立驗收不得含重複影像。")

    jobs = project_dir / JOBS_DIRECTORY
    if jobs.resolve() != jobs:
        raise ValueError("任務資料夾必須位於目前的專案。")
    jobs.mkdir(exist_ok=True)
    ident = uuid4().hex
    final = jobs / ident
    with TemporaryDirectory(prefix=".job-", dir=jobs) as temp:
        temp = Path(temp)
        manifest = {"schema": JOB_SCHEMA, "job_id": ident, "created": _now(), "duplicate_inputs": duplicates,
                    "config": {"classes": classes, "settings": settings}, "frames": frames}
        if manual_only:
            manifest["purpose"] = "manual_training"
        _write_json(temp / "manifest.json", manifest)
        state = {"schema": STATE_SCHEMA, "job_id": ident, "status": "review_ready" if manual_only else "prepared",
                 "manifest_sha256": _digest(temp / "manifest.json"), "inference_sha256": None, "reviews": {},
                 "reference_sha256": None}
        if reference is not None:
            _write_json(temp / "reference.json", reference)
            state["reference_sha256"] = _digest(temp / "reference.json")
        _write_json(temp / "state.json", state)
        temp.rename(final)
    return GenericJob.open(final)


def create_manual_training_job(project, classes, *, holdout=None):
    classes = _class_config(classes)
    labels = {row["label"] for row in classes}
    current = _read_manifest(project.directory)
    previous_tests = [item["frame"] for job in jobs_for(project) if (reference := job.reference)
                      for item in reference["frames"]]
    held_out = {frame["image_sha256"] for frame in previous_tests}
    held_groups = {frame["source_group"] for frame in previous_tests}
    confirmed = [asset["id"] for asset in current["assets"]
                 if labels <= _manual_coverage(asset, current["labels"])
                 and asset.get("image_sha256") not in held_out
                 and _source_group(asset, {}) not in held_groups
                 and all(row["status"] != "excluded" for label, row in asset.get("bbox_review_by_class", {}).items()
                         if label in labels)]
    if not confirmed:
        raise ValueError("請先在標註頁確認整張影像，包含本次訓練的所有類別。")
    return create_job(project, classes, asset_ids=confirmed, holdout=holdout, manual_only=True)


def jobs_for(project):
    folder = project.directory / JOBS_DIRECTORY
    if not folder.is_dir():
        return []
    jobs = [GenericJob.open(path) for path in folder.iterdir()
            if path.is_dir() and HEX_ID.fullmatch(path.name) and (path / "manifest.json").is_file()]
    return sorted(jobs, key=lambda job: (job.manifest["created"], job.directory.name), reverse=True)


class GenericJob:
    """An inference snapshot and editable whole-frame review decisions."""

    @classmethod
    def open(cls, path):
        directory = Path(path).absolute()
        if (directory.resolve(strict=True) != directory or directory.parent.name != JOBS_DIRECTORY
                or not HEX_ID.fullmatch(directory.name)):
            raise ValueError("通用標註任務的資料夾無效。")
        job = cls(directory)
        return job.refresh()

    def __init__(self, directory):
        self.directory = directory
        self.project_dir = directory.parent.parent
        self.manifest_path = directory / "manifest.json"
        self.inference_path = directory / "inference.json"
        self.training_report_path = directory / "training" / "training.json"

    def refresh(self):
        manifest = _load(_checked_file(self.directory, "manifest.json"))
        state = _load(_checked_file(self.directory, "state.json"))
        if (manifest.get("schema") != JOB_SCHEMA or manifest.get("job_id") != self.directory.name
                or state.get("schema") != STATE_SCHEMA or state.get("job_id") != self.directory.name
                or state.get("manifest_sha256") != _digest(self.manifest_path)
                or state.get("status") not in {"prepared", "review_ready"}):
            raise ValueError("標註任務的內容、版本或校驗記錄不符。")
        self.manifest, self.state = manifest, state
        return self

    @property
    def inference(self):
        digest = self.state["inference_sha256"]
        if digest is None:
            return None
        path = _checked_file(self.directory, "inference.json")
        if _digest(path) != digest:
            raise ValueError("模型裁決已變更，請核對封存結果。")
        return _load(path)

    @property
    def reference(self):
        digest = self.state["reference_sha256"]
        if digest is None:
            return None
        path = _checked_file(self.directory, "reference.json")
        if _digest(path) != digest:
            raise ValueError("獨立驗收標註已變更。")
        return _load(path)

    def publish_inference(self, result_path):
        """Record every frame in one sealed prediction result before permitting edits."""
        with _project_lock(self.directory):
            self.refresh()
            report = _load(Path(result_path))
            if self.state["status"] == "review_ready":
                if self.inference == report:
                    return self.summary()
                raise ValueError("這批任務已封存第一次裁決，請開啟新任務重跑。")
            frames = {row["id"]: row for row in self.manifest["frames"]}
            if (report.get("schema") != INFERENCE_SCHEMA or report.get("job_id") != self.manifest["job_id"]
                    or report.get("manifest_sha256") != self.state["manifest_sha256"]
                    or not isinstance(report.get("frames"), list)):
                raise ValueError("模型裁決未對應目前封存的任務。")
            completed = set()
            labels = _labels(self.manifest)
            for row in report["frames"]:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    raise ValueError("模型輸出的畫格識別碼無效。")
                ident = row["id"]
                frame = frames.get(ident)
                if frame is None or ident in completed or row.get("image_sha256") != frame["image_sha256"]:
                    raise ValueError("模型裁決的畫格重複或來源不符。")
                _boxes(row.get("boxes"), frame, labels)
                exceptions = row.get("exceptions")
                if (not isinstance(exceptions, list) or any(not isinstance(item, dict)
                        or not isinstance(item.get("code"), str) or not item["code"].strip() for item in exceptions)):
                    raise ValueError("模型裁決的例外資料無效。")
                completed.add(ident)
            if completed != frames.keys():
                raise ValueError("模型裁決沒有涵蓋任務內的每一張畫格。")
            _write_json(self.inference_path, report)
            self.state.update(status="review_ready", inference_sha256=_digest(self.inference_path))
            _write_json(self.directory / "state.json", self.state)
        return self.summary()

    def _decisions(self, predictions, assets=None, project_labels=None):
        if assets is None:
            project = _read_manifest(self.project_dir)
            assets = {asset["id"]: asset for asset in project["assets"]}
            project_labels = project["labels"]
        labels = set(_labels(self.manifest))
        predictions = {row["id"]: row for row in predictions["frames"]} if predictions is not None else {}
        decisions = {}
        for frame in self.manifest["frames"]:
            ident = frame["id"]
            row = predictions.get(ident, {})
            by_human = self.state["reviews"].get(ident)
            linked = _linked_decision(frame, assets.get(ident), labels, project_labels, row, by_human or {})
            if linked is not None:
                decisions[ident] = linked
            elif by_human:
                decisions[ident] = {**by_human, "box_origin": "manual"}
            elif not row:
                decisions[ident] = {"status": "awaiting_inference", "boxes": [], "exceptions": []}
            elif row["exceptions"] or not row["boxes"]:
                decisions[ident] = {"status": "pending", "boxes": row["boxes"],
                                    "exceptions": row["exceptions"] or [EMPTY_EXCEPTION]}
            else:
                decisions[ident] = {"status": "ai_ready", "boxes": row["boxes"]}
        return decisions

    def frame_rows(self):
        predictions = self.inference
        project = _read_manifest(self.project_dir)
        assets = {asset["id"]: asset for asset in project["assets"]}
        decisions = self._decisions(predictions, assets, project["labels"])
        by_id = {row["id"]: row for row in predictions["frames"]} if predictions else {}
        rows = []
        for frame in self.manifest["frames"]:
            identifier = frame["id"]
            model_row = by_id.get(identifier, {})
            decision = decisions[identifier]
            boxes = [{**box, "source": "manual" if decision.get("box_origin") == "manual"
                      else box.get("source", "mm_qwen")} for box in decision["boxes"]]
            human = sum(box["source"] == "manual" for box in boxes)
            asset = assets.get(identifier)
            rows.append({**model_row, **frame, **decision, "boxes": boxes,
                         "confidence": frame_confidence(model_row) if model_row else None,
                         "ai_exceptions": model_row.get("exceptions", []),
                         "manual_box_count": human, "ai_box_count": len(boxes) - human,
                         "project_annotation_revision": _box_review_revision(asset) if asset else None})
        return rows

    def frames_for_review(self):
        return [row for row in self.frame_rows() if row["status"] == "pending"]

    def review(self, frame_id, boxes=None, *, exclude=False, expected_project_revision=None):
        """Store the operator's complete frame result, or omit the frame from training."""
        with _project_lock(self.directory):
            self.refresh()
            if self.state["status"] != "review_ready":
                raise ValueError("請先完成模型裁決。")
            frame = next((row for row in self.manifest["frames"] if row["id"] == frame_id), None)
            if frame is None:
                raise ValueError("這張畫格不屬於目前可複核的任務。")
            if type(exclude) is not bool or exclude and boxes:
                raise ValueError("排除畫格和提交標註只能擇一。")
            reviewed_boxes = [] if exclude else _boxes(boxes, frame, _labels(self.manifest))
            Project.open(self.project_dir).apply_batch_box_review(
                frame, _labels(self.manifest), reviewed_boxes, job_id=self.manifest["job_id"],
                status="excluded" if exclude else "approved", reviewed_at=_now(),
                expected_revision=expected_project_revision)
        return self.summary()

    def reopen(self, frame_id):
        with _project_lock(self.directory):
            self.refresh()
            frame = next((row for row in self.manifest["frames"] if row["id"] == frame_id), None)
            if frame is not None:
                Project.open(self.project_dir).apply_batch_box_review(
                    frame, _labels(self.manifest), [], job_id=self.manifest["job_id"],
                    status="pending", reviewed_at=_now())
            if self.state["reviews"].pop(frame_id, None) is not None:
                _write_json(self.directory / "state.json", self.state)

    def review_records(self):
        return {ident: row for ident, row in self._decisions(self.inference).items()
                if row["status"] in {"approved", "excluded"}}

    def summary(self):
        decisions = self._decisions(self.inference)
        counts = Counter(row["status"] for row in decisions.values())
        boxes = Counter(box["label"] for row in decisions.values()
                        if row["status"] in {"ai_ready", "approved"} for box in row["boxes"])
        human = Counter(box["label"] for row in decisions.values() if row["status"] == "approved"
                        for box in row["boxes"])
        reference = self.reference
        return {"frames": len(self.manifest["frames"]), "ready": counts["ai_ready"] + counts["approved"],
                "ai_ready": counts["ai_ready"], "approved": counts["approved"], "needs_review": counts["pending"],
                "excluded": counts["excluded"], "boxes_by_class": {label: boxes[label] for label in _labels(self.manifest)},
                "human_boxes_by_class": {label: human[label] for label in _labels(self.manifest)},
                "reference_frames": len(reference["frames"]) if reference else 0,
                "duplicate_inputs": self.manifest.get("duplicate_inputs", 0)}

    def export_dataset(self):
        """Copy eligible images/labels to an immutable training snapshot with explicit splits."""
        with _project_lock(self.directory):
            self.refresh()
            if self.state["status"] != "review_ready":
                raise ValueError("請先完成模型裁決。")
            labels = _labels(self.manifest)
            decisions = self._decisions(self.inference)
            train = [(frame, decisions[frame["id"]]["boxes"]) for frame in self.manifest["frames"]
                     if decisions[frame["id"]]["status"] in {"ai_ready", "approved"}]
            if not any(boxes for _, boxes in train):
                raise ValueError("至少需要一個已放行的訓練正例，才能從零開始訓練。")
            reference = self.reference
            test = [] if reference is None else [(row["frame"], row["boxes"]) for row in reference["frames"]]
            if {frame["source_group"] for frame, _ in train} & {frame["source_group"] for frame, _ in test}:
                raise ValueError("同一來源不能同時進入訓練與獨立驗收。")
            identity = sha256(json.dumps({"manifest": self.state["manifest_sha256"],
                                          "reference": self.state["reference_sha256"],
                                          "train": train}, sort_keys=True, allow_nan=False).encode()).hexdigest()
            snapshots = self.directory / "datasets"
            if snapshots.resolve() != snapshots:
                raise ValueError("訓練資料夾必須位於目前任務。")
            snapshots.mkdir(exist_ok=True)
            target = snapshots / identity
            datasets = [("train", train), ("test", test)]
            if target.exists():
                _verify_export(target, identity, labels, datasets)
                return target
            with TemporaryDirectory(prefix=".export-", dir=snapshots) as temporary:
                staging = Path(temporary)
                (staging / "images").mkdir()
                (staging / "labels").mkdir()
                rows = [_write_frame(self.project_dir, staging, labels, frame, boxes, split)
                        for split, items in datasets for frame, boxes in items]
                _write_json(staging / "dataset.json", {
                    "labels": labels, "frames": rows, "split_policy": "explicit_project_split", "dataset_id": identity,
                    "task_manifest_sha256": self.state["manifest_sha256"],
                    "independent_test": {"provenance": reference["provenance"],
                                         "source_sha256": reference["source_sha256"]} if reference else None,
                })
                staging.rename(target)
        return target


def _yolo_text(labels, frame, boxes):
    width, height = frame["width"], frame["height"]
    indices = {name: index for index, name in enumerate(labels)}
    lines = []
    for row in boxes:
        x1, y1, x2, y2 = row["xyxy"]
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        lines.append(f"{indices[row['label']]} " + " ".join(f"{value:.12g}" for value in values) + "\n")
    return "".join(lines).encode("ascii")


def _inventory_row(frame, label_text, split):
    ident = frame["id"]
    return {"image": f"images/{ident}.png", "label": f"labels/{ident}.txt",
            "image_sha256": frame["image_sha256"], "label_sha256": sha256(label_text).hexdigest(),
            "shot": frame["source_group"], "split": split}


def _write_frame(project, staging, labels, frame, boxes, split):
    text = _yolo_text(labels, frame, boxes)
    row = _inventory_row(frame, text, split)
    source = _checked_file(project, frame["image"])
    image = staging / row["image"]
    copyfile(source, image)
    if _digest(image) != frame["image_sha256"]:
        raise ValueError("畫格檔案與本次任務封存的 SHA-256 不符。")
    (staging / row["label"]).write_bytes(text)
    return row


def _verify_export(root, identity, labels, datasets):
    if root.resolve(strict=True) != root:
        raise ValueError("訓練快照必須位於目前任務。")
    inventory = _load(_checked_file(root, "dataset.json"))
    expected = [_inventory_row(frame, _yolo_text(labels, frame, boxes), split)
                for split, items in datasets for frame, boxes in items]
    if (inventory.get("dataset_id") != identity or inventory.get("split_policy") != "explicit_project_split"
            or inventory.get("labels") != labels or inventory.get("frames") != expected):
        raise ValueError("已封存的訓練快照與目前任務不符。")
    for row in inventory["frames"]:
        for key in ("image", "label"):
            if _digest(_checked_file(root, row[key])) != row[f"{key}_sha256"]:
                raise ValueError("已封存的訓練快照內容已變更。")
