"""Local project persistence and portable COCO instance segmentation export."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps


_MANIFEST = "project.json"
_LOCK = ".project.lock"
_ID = re.compile(r"[a-f0-9]{32}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_SCHEMA_VERSION = 1


@contextmanager
def _project_lock(directory: Path):
    with (directory / _LOCK).open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_json(path: Path, value: dict) -> None:
    """Flush a new snapshot before replacing the previous manifest."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}-",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_manifest(directory: Path) -> dict:
    path = directory / _MANIFEST
    if not path.is_file():
        raise ValueError(f"找不到 {path.name}，請選擇已建立的標註專案資料夾。")
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("專案記錄無法讀取。") from exc
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != _SCHEMA_VERSION):
        raise ValueError("此專案的資料格式版本不受支援。")
    labels = data.get("labels")
    assets = data.get("assets")
    if (not isinstance(labels, list) or any(not isinstance(name, str) or not name for name in labels)
            or len(labels) != len(set(labels)) or not isinstance(assets, list)):
        raise ValueError("專案記錄的類別或圖片清單無效。")
    seen = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("專案記錄有無效的圖片。")
        asset_id = asset.get("id")
        if (not isinstance(asset_id, str) or not _ID.fullmatch(asset_id) or asset_id in seen
                or not isinstance(asset.get("name"), str) or not isinstance(asset.get("annotations"), list)
                or any(type(asset.get(key)) is not int or asset[key] < 1 for key in ("width", "height"))):
            raise ValueError("專案記錄有無效的圖片或重複的圖片識別碼。")
        seen.add(asset_id)
    return data


def _asset(data: dict, asset_id: str) -> dict:
    for asset in data["assets"]:
        if asset["id"] == asset_id:
            return asset
    raise ValueError("此圖片不在目前的標註專案內。")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _normalize_image(source: Path, destination: Path) -> tuple[int, int]:
    try:
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original)
            if "A" in image.getbands() or "transparency" in image.info:
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image.convert("RGBA"))
            rgb = image.convert("RGB")
            rgb.info.clear()
            rgb.save(destination, format="PNG")
            return rgb.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"無法匯入圖片「{source.name}」：{exc}") from exc


def _encode_mask(mask: np.ndarray) -> list[int]:
    pixels = mask.ravel(order="F")
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [pixels.size]))).tolist()
    if pixels[0]:
        counts.insert(0, 0)
    return counts


def _mask_geometry(asset: dict, mask: np.ndarray) -> dict:
    pixels = np.asarray(mask, dtype=bool)
    size = [asset["height"], asset["width"]]
    if pixels.shape != tuple(size) or not pixels.any():
        raise ValueError("遮罩不可為空，尺寸必須和原圖一致。")
    rows = np.flatnonzero(pixels.any(axis=1))
    columns = np.flatnonzero(pixels.any(axis=0))
    return {
        "size": size,
        "counts": _encode_mask(pixels),
        "bbox": [int(columns[0]), int(rows[0]), int(columns[-1] - columns[0] + 1),
                 int(rows[-1] - rows[0] + 1)],
        "area": int(pixels.sum()),
    }


def _box_geometry(asset: dict, box) -> dict:
    try:
        left, top, right, bottom = (float(value) for value in box)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("偵測框需要左上與右下的四個座標。") from exc
    if not (0 <= left < right <= asset["width"] and 0 <= top < bottom <= asset["height"]):
        raise ValueError("偵測框必須位於圖片範圍內，並具有正寬度與高度。")
    width, height = right - left, bottom - top
    return {"bbox": [left, top, width, height], "area": width * height}


def _finite(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"影片結果中的{name}無效。")
    return float(value)


def _video_origin(value, kind: str, approved: str) -> dict:
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str) or not value["path"]
            or not isinstance(approved, str) or not _SHA256.fullmatch(approved)
            or value.get("sha256") != approved):
        raise ValueError(f"{kind}與審查確認的檔案不符。")
    return {"path": value["path"], "sha256": approved}


def _video_annotations(boxes, asset: dict, model_labels: dict[int, str]) -> list[dict]:
    if not isinstance(boxes, list):
        raise ValueError("影片影格缺少有效的偵測框清單。")
    rows = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("影片偵測框格式無效。")
        label = box.get("label")
        class_id, track_id = box.get("class_id"), box.get("track_id")
        if (not isinstance(label, str) or not label or label.strip() != label or not label.isprintable()
                or type(class_id) is not int or class_id < 0
                or track_id is not None and (type(track_id) is not int or track_id < 0)):
            raise ValueError("影片偵測框的類別或追蹤編號無效。")
        if model_labels.setdefault(class_id, label) != label:
            raise ValueError("影片結果中的模型類別對應不一致。")
        confidence = _finite(box.get("confidence"), "信心分數")
        if not 0 <= confidence <= 1:
            raise ValueError("影片偵測框的信心分數超出範圍。")
        bbox = box.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("影片偵測框缺少四個座標。")
        x, y, width, height = (_finite(number, "偵測框座標") for number in bbox)
        geometry = _box_geometry(asset, (x, y, x + width, y + height))
        rows.append({"id": uuid4().hex, "type": "bbox", "label": label, "source": "video_auto",
                     "score": confidence, "model_class_id": class_id, "track_id": track_id, **geometry})
    return rows


class Project:
    """Images are immutable PNGs and edits replace a single versioned JSON manifest."""

    def __init__(self, directory: Path, data: dict):
        self.directory = directory
        self._data = data

    @classmethod
    def create(cls, directory: Path) -> Project:
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any(entry.name != _LOCK for entry in directory.iterdir()):
            raise ValueError("請選擇空白或新建的資料夾來建立專案。")
        with _project_lock(directory):
            if any(entry.name != _LOCK for entry in directory.iterdir()):
                raise ValueError("請選擇空白或新建的資料夾來建立專案。")
            data = {"schema_version": _SCHEMA_VERSION, "labels": [], "assets": []}
            _write_json(directory / _MANIFEST, data)
        return cls(directory, data)

    @classmethod
    def open(cls, directory: Path) -> Project:
        directory = Path(directory).resolve()
        return cls(directory, _read_manifest(directory))

    @property
    def assets(self) -> list[dict]:
        return self._data["assets"]

    @property
    def labels(self) -> list[str]:
        return self._data["labels"].copy()

    def _commit(self, data: dict) -> None:
        _write_json(self.directory / _MANIFEST, data)
        self._data = data

    def _image_directory(self) -> Path:
        directory = self.directory / "images"
        if directory.resolve() != directory:
            raise ValueError("專案圖片資料夾必須在專案內。")
        return directory

    def image_path(self, asset_id: str) -> Path:
        _asset(self._data, asset_id)
        return self._image_directory() / f"{asset_id}.png"

    def add_images(self, paths) -> int:
        sources = [Path(path) for path in paths]
        if not sources:
            return 0
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            known_sources = {asset.get("source_sha256") for asset in data["assets"]}
            known_images = {asset.get("image_sha256") for asset in data["assets"]}
            staged = []
            with tempfile.TemporaryDirectory(prefix=".import-", dir=self.directory) as staging:
                staging = Path(staging)
                for source in sources:
                    try:
                        source_sha = _digest(source)
                    except OSError as exc:
                        raise ValueError(f"無法讀取圖片「{source.name}」：{exc}") from exc
                    if source_sha in known_sources:
                        continue
                    asset_id = uuid4().hex
                    candidate = staging / f"{asset_id}.png"
                    width, height = _normalize_image(source, candidate)
                    image_sha = _digest(candidate)
                    known_sources.add(source_sha)
                    if image_sha in known_images:
                        continue
                    known_images.add(image_sha)
                    data["assets"].append({
                        "id": asset_id, "name": source.name, "width": width, "height": height,
                        "source_sha256": source_sha, "image_sha256": image_sha, "reviewed": False,
                        "annotations": [],
                    })
                    staged.append(candidate)
                self._commit_images(staged, data)
            return len(staged)

    def _commit_images(self, staged: list[Path], data: dict) -> None:
        if not staged:
            self._data = data
            return
        images = self._image_directory()
        images.mkdir(exist_ok=True)
        installed = []
        try:
            for candidate in staged:
                destination = images / candidate.name
                candidate.rename(destination)
                installed.append(destination)
            self._commit(data)
        except BaseException:
            for destination in installed:
                destination.unlink(missing_ok=True)
            raise

    def import_video_annotations(self, manifest_path: Path, *, approved_video_sha256: str,
                                 approved_model_sha256: str | None) -> dict[str, int]:
        """Commit a completed, approved worker result as pending video frames."""
        manifest_path = Path(manifest_path).resolve(strict=True)
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("影片處理結果無法讀取。") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("不支援此影片處理結果格式。")
        source_info = manifest.get("source")
        source = _video_origin(source_info, "影片", approved_video_sha256)
        keyframes = manifest.get("mode") == "keyframes"
        if keyframes != (approved_model_sha256 is None) or (keyframes and manifest.get("model") is not None):
            raise ValueError("影片處理模式與確認內容不符。")
        model = None if keyframes else _video_origin(manifest.get("model"), "模型", approved_model_sha256)
        dimensions = (source_info.get("width"), source_info.get("height"))
        fps = _finite(source_info.get("fps"), "幀率")
        frame_count = source_info.get("frame_count")
        sample_fps = _finite(manifest.get("sample_fps"), "取樣幀率")
        confidence = _finite(manifest.get("conf"), "信心門檻")
        frames = manifest.get("frames")
        if (any(type(size) is not int or size <= 0 for size in dimensions) or fps <= 0
                or type(frame_count) is not int or frame_count <= 0 or sample_fps <= 0
                or not 0 <= confidence <= 1 or not isinstance(frames, list) or not frames):
            raise ValueError("影片結果中的尺寸、取樣設定或影格清單無效。")

        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            known = {(asset["video"]["sha256"], asset["video"]["frame_index"])
                     for asset in data["assets"] if asset.get("video")}
            skipped = box_count = 0
            previous_index, previous_time = -1, -1.0
            model_labels: dict[int, str] = {}
            with tempfile.TemporaryDirectory(prefix=".video-", dir=self.directory) as staging:
                staged = []
                for row in frames:
                    if not isinstance(row, dict):
                        raise ValueError("影片影格記錄無效。")
                    index = row.get("frame_index")
                    time_ms = _finite(row.get("time_ms"), "時間戳記")
                    if (type(index) is not int or not previous_index < index < frame_count
                            or time_ms < 0 or time_ms < previous_time
                            or (row.get("width"), row.get("height")) != dimensions):
                        raise ValueError("影片影格順序、時間或尺寸不一致。")
                    previous_index, previous_time = index, time_ms
                    filename = f"frames/{index:09d}.png"
                    digest = row.get("sha256")
                    if row.get("file") != filename or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                        raise ValueError("影片影格的檔名或雜湊無效。")
                    asset_id = uuid4().hex
                    asset = {
                        "id": asset_id, "name": f"{Path(source['path']).stem} · {time_ms / 1000:.3f}s",
                        "width": dimensions[0], "height": dimensions[1], "source_sha256": digest,
                        "image_sha256": digest, "reviewed": False,
                        "video": {**source, "frame_index": index, "time_ms": time_ms, "fps": fps,
                                  "sample_fps": sample_fps, "conf": confidence, "model": model},
                    }
                    if keyframes:
                        if row.get("boxes") != []:
                            raise ValueError("關鍵畫格匯入含有未確認的標註。")
                        asset["video"]["keyframe"] = True
                        asset["annotations"] = []
                    else:
                        asset["annotations"] = _video_annotations(row.get("boxes"), asset, model_labels)
                    if (source["sha256"], index) in known:
                        skipped += 1
                        continue
                    frame_path = manifest_path.parent / filename
                    if frame_path.resolve() != frame_path or not frame_path.is_file():
                        raise ValueError("處理結果缺少影格檔案，或影格路徑超出工作區。")
                    candidate = Path(staging) / f"{asset_id}.png"
                    shutil.copyfile(frame_path, candidate)
                    if _digest(candidate) != digest:
                        raise ValueError("影片影格在匯入前已變更。")
                    with Image.open(candidate) as frame:
                        if frame.format != "PNG" or frame.size != dimensions:
                            raise ValueError("影片影格檔案與記錄的格式或尺寸不符。")
                        frame.verify()
                    for annotation in asset["annotations"]:
                        if annotation["label"] not in data["labels"]:
                            data["labels"].append(annotation["label"])
                    data["assets"].append(asset)
                    box_count += len(asset["annotations"])
                    staged.append(candidate)

                self._commit_images(staged, data)
            return {"imported_frames": len(staged), "added_boxes": box_count, "skipped_frames": skipped}

    def add_label(self, name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("類別名稱必須是文字。")
        name = name.strip()
        if not name or not name.isprintable():
            raise ValueError("請輸入有效的類別名稱。")
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            if name not in data["labels"]:
                data["labels"].append(name)
                self._commit(data)
            else:
                self._data = data
        return name

    def add_annotation(self, asset_id: str, label: str, mask: np.ndarray,
                       source: str, score: float | None = None) -> str:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("標註來源無效。")
        if score is not None:
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("模型分數無效。")
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            asset = _asset(data, asset_id)
            if label not in data["labels"]:
                raise ValueError("請先選擇已建立的類別。")
            annotation_id = uuid4().hex
            asset["annotations"].append({
                "id": annotation_id, "type": "mask", "label": label, "source": source.strip(), "score": score,
                **_mask_geometry(asset, mask),
            })
            asset["reviewed"] = False
            self._commit(data)
        return annotation_id

    def replace_annotation_mask(self, asset_id: str, annotation_id: str, mask: np.ndarray, *,
                                expected_counts: list[int] | None = None) -> str:
        """Replace a confirmed annotation in place and invalidate its review if the mask changed."""
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            asset = _asset(data, asset_id)
            annotation = next((item for item in asset["annotations"] if item["id"] == annotation_id), None)
            if annotation is None or annotation.get("type") == "bbox":
                raise ValueError("這筆遮罩標註已不存在，請重新開啟圖片。")
            if expected_counts is not None and annotation["counts"] != expected_counts:
                raise ValueError("這筆標註已在其他視窗更新，請重新開啟專案再修整。")
            geometry = _mask_geometry(asset, mask)
            if geometry["counts"] == annotation["counts"]:
                self._data = data
            else:
                annotation.update(geometry, score=None, refined=True)
                asset["reviewed"] = False
                self._commit(data)
        return annotation_id

    @property
    def grounding_prompt(self) -> str:
        return self._data.get("grounding_prompt", "car door")

    def set_grounding_prompt(self, prompt: str) -> str:
        prompt = " ".join(prompt.split()).strip(".").strip()
        if not prompt or len(prompt) > 120:
            raise ValueError("請輸入 1 至 120 字元的物件提示詞。")
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            if data.get("grounding_prompt") != prompt:
                data["grounding_prompt"] = prompt
                self._commit(data)
        return prompt

    def add_box(self, asset_id: str, label: str, box, *, suggestion=None) -> str:
        """Save a confirmed detection box in original-image pixels."""
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            asset = _asset(data, asset_id)
            if label not in data["labels"]:
                raise ValueError("請先選擇已建立的類別。")
            annotation_id = uuid4().hex
            record = {"id": annotation_id, "type": "bbox", "label": label,
                      "source": "manual", "score": None, **_box_geometry(asset, box)}
            if suggestion is not None:
                prompt, rank, score = suggestion
                if not isinstance(prompt, str) or not prompt.strip() or type(rank) is not int or not 1 <= rank <= 5:
                    raise ValueError("文字候選來源無效。")
                score = _finite(score, "候選分數")
                if not 0 <= score <= 1:
                    raise ValueError("候選分數無效。")
                record.update(source="grounding_dino", score=score, suggestion={"prompt": prompt, "rank": rank})
            asset["annotations"].append(record)
            asset["reviewed"] = False
            self._commit(data)
        return annotation_id

    def replace_box(self, asset_id: str, annotation_id: str, box, *, expected_bbox=None) -> str:
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            asset = _asset(data, asset_id)
            annotation = next((item for item in asset["annotations"] if item["id"] == annotation_id), None)
            if annotation is None or annotation.get("type") != "bbox":
                raise ValueError("這筆偵測框已不存在，請重新開啟圖片。")
            if expected_bbox is not None and annotation["bbox"] != list(expected_bbox):
                raise ValueError("這個框已在其他視窗更新，請重新開啟專案再修改。")
            geometry = _box_geometry(asset, box)
            if geometry["bbox"] == annotation["bbox"]:
                self._data = data
            else:
                annotation.update(geometry)
                asset["reviewed"] = False
                self._commit(data)
        return annotation_id

    def delete_annotation(self, asset_id: str, ann_id: str) -> None:
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            asset = _asset(data, asset_id)
            remaining = [ann for ann in asset["annotations"] if ann["id"] != ann_id]
            if len(remaining) == len(asset["annotations"]):
                raise ValueError("找不到要移除的標註。")
            asset["annotations"] = remaining
            asset["reviewed"] = False
            self._commit(data)

    def set_reviewed(self, asset_id: str, reviewed: bool) -> None:
        if type(reviewed) is not bool:
            raise ValueError("圖片核對狀態必須為是或否。")
        with _project_lock(self.directory):
            data = _read_manifest(self.directory)
            asset = _asset(data, asset_id)
            if asset.get("reviewed", False) != reviewed:
                asset["reviewed"] = reviewed
                self._commit(data)
            else:
                self._data = data

    @staticmethod
    def decode_mask(annotation: dict) -> np.ndarray:
        size, counts = annotation.get("size"), annotation.get("counts")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(length) is not int or length <= 0 for length in size)
                or not isinstance(counts, list) or not counts
                or any(type(run) is not int or run < 0 for run in counts)
                or sum(counts) != size[0] * size[1]):
            raise ValueError("遮罩的 RLE 編碼與圖片尺寸不符。")
        runs = np.arange(len(counts), dtype=np.int64) % 2 == 1
        return np.repeat(runs, counts).reshape(size, order="F")

    def _export_data(self) -> tuple[list[str], list[dict]]:
        data = _read_manifest(self.directory)
        assets = [asset for asset in data["assets"] if asset.get("reviewed")
                  or (asset["annotations"] and not asset.get("video"))]
        if not assets:
            raise ValueError("請先儲存圖片標註，或將至少一張影片影格標記為已審核。")
        return data["labels"], assets

    @contextmanager
    def _export_target(self, directory: Path):
        destination = Path(directory).resolve()
        image_directory = self._image_directory()
        if destination == self.directory or destination.is_relative_to(image_directory):
            raise ValueError("匯出位置不可使用專案或原圖資料夾。")
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise ValueError("請選擇空白或新建的匯出資料夾。")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".labels-", dir=destination.parent) as staging:
            staging = Path(staging) / "export"
            (staging / "images").mkdir(parents=True)
            yield staging
            # Publish the complete export together. rmdir succeeds only if still empty.
            already_empty = destination.is_dir()
            if already_empty:
                destination.rmdir()
            try:
                staging.rename(destination)
            except OSError:
                if already_empty:
                    destination.mkdir(exist_ok=True)
                raise

    def _export_image(self, staging: Path, asset: dict) -> str:
        filename = f"{asset['id']}.png"
        exported = staging / "images" / filename
        shutil.copyfile(self._image_directory() / filename, exported)
        if _digest(exported) != asset.get("image_sha256"):
            raise ValueError(f"專案圖片「{asset['name']}」已修改，無法匯出對應的標註。")
        return filename

    def export_coco(self, directory: Path) -> Path:
        labels, assets = self._export_data()
        categories = {name: i for i, name in enumerate(labels, start=1)}
        coco = {
            "info": {"description": "Annotation Wizard export", "version": "1.0"},
            "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": category_id, "name": name, "supercategory": "object"}
                           for name, category_id in categories.items()],
        }
        with self._export_target(directory) as staging:
            for image_id, asset in enumerate(assets, start=1):
                filename = self._export_image(staging, asset)
                coco["images"].append({"id": image_id, "file_name": f"images/{filename}",
                                       "width": asset["width"], "height": asset["height"]})
                for annotation in asset["annotations"]:
                    row = {"id": len(coco["annotations"]) + 1, "image_id": image_id,
                           "category_id": categories[annotation["label"]], "iscrowd": 0,
                           "bbox": annotation["bbox"], "area": annotation["area"]}
                    if annotation.get("type") != "bbox":
                        row["segmentation"] = {"size": annotation["size"], "counts": annotation["counts"]}
                    coco["annotations"].append(row)
            _write_json(staging / "annotations.json", coco)
        return Path(directory).resolve() / "annotations.json"

    def export_yolo(self, directory: Path) -> Path:
        """Export paired images and YOLO detection labels with a zero-based class list."""
        labels, assets = self._export_data()
        categories = {name: i for i, name in enumerate(labels)}
        with self._export_target(directory) as staging:
            (staging / "labels").mkdir()
            for asset in assets:
                name = self._export_image(staging, asset)
                image_width, image_height = asset["width"], asset["height"]
                rows = []
                for annotation in asset["annotations"]:
                    x, y, width, height = annotation["bbox"]
                    coords = ((x + width / 2) / image_width, (y + height / 2) / image_height,
                              width / image_width, height / image_height)
                    normalized = " ".join(f"{value:.10f}" for value in coords)
                    rows.append(f"{categories[annotation['label']]} {normalized}\n")
                (staging / "labels" / f"{Path(name).stem}.txt").write_text("".join(rows), encoding="utf-8", newline="\n")
            (staging / "classes.txt").write_text("".join(f"{label}\n" for label in labels), encoding="utf-8", newline="\n")
        return Path(directory).resolve() / "classes.txt"
