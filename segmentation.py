"""Prompted segmentation for one local image at a time."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from math import ceil, floor, isfinite
from pathlib import Path
from threading import RLock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from mask_cleanup import clean_mask


Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    mask: np.ndarray
    score: float | None
    elapsed_ms: float
    raw_mask: np.ndarray | None = None
    cleanup_stats: dict[str, int] = field(default_factory=dict)


def _points(points: list[Point], width: int, height: int) -> tuple[Point, ...]:
    coords = tuple((float(x), float(y)) for x, y in points)
    if any(not (0 <= x < width and 0 <= y < height) for x, y in coords):
        raise ValueError("提示點必須在圖片範圍內。")
    return coords


def _pixel(point: Point) -> tuple[int, int]:
    return int(point[0]), int(point[1])


def _check_prompts(
    size: tuple[int, int], box: Box | None, positive: list[Point], negative: list[Point]
) -> tuple[Box | None, tuple[Point, ...], tuple[Point, ...]]:
    width, height = size
    if box is not None:
        left, top, right, bottom = (float(value) for value in box)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("提示框需要位於圖片內，並具有正寬度與高度。")
        box = left, top, right, bottom
    foreground = _points(positive, width, height)
    background = _points(negative, width, height)
    if set(map(_pixel, foreground)) & set(map(_pixel, background)):
        raise ValueError("同一像素不能同時是前景點和背景點。")
    if box is None and not foreground:
        raise ValueError("請先框住物件，或至少標一個前景點。")
    return box, foreground, background


def _honors_points(mask: np.ndarray, positive: tuple[Point, ...], negative: tuple[Point, ...]) -> bool:
    return all(mask[int(y), int(x)] for x, y in positive) and not any(
        mask[int(y), int(x)] for x, y in negative
    )


def _grabcut(
    image: Image.Image, box: Box | None, positive: tuple[Point, ...], negative: tuple[Point, ...]
) -> np.ndarray:
    width, height = image.size
    if min(width, height) < 8:
        raise ValueError("GrabCut 需要至少 8 × 8 像素的圖片。")
    if box is None:
        crop = (0, 0, width, height)
    else:
        x1, y1, x2, y2 = box
        pad = max(8, ceil(min(x2 - x1, y2 - y1) * .12))
        pixels = tuple(map(_pixel, positive + negative))
        xs = [x for x, _ in pixels]
        ys = [y for _, y in pixels]
        # Include correction clicks even when the initial box missed part of the object.
        crop = (
            max(0, min([floor(x1), *xs]) - pad),
            max(0, min([floor(y1), *ys]) - pad),
            min(width, max([ceil(x2), *(x + 1 for x in xs)]) + pad),
            min(height, max([ceil(y2), *(y + 1 for y in ys)]) + pad),
        )

    left, top, right, bottom = crop
    rgb = np.asarray(image.crop(crop))
    seeds = np.full(rgb.shape[:2], cv2.GC_PR_FGD if box is None else cv2.GC_PR_BGD, dtype=np.uint8)
    if box is None:
        border = max(1, min(width, height) // 100)
        seeds[:border, :] = seeds[-border:, :] = cv2.GC_BGD
        seeds[:, :border] = seeds[:, -border:] = cv2.GC_BGD
    else:
        seeds[floor(y1) - top:ceil(y2) - top, floor(x1) - left:ceil(x2) - left] = cv2.GC_PR_FGD
    radius = max(1, min(seeds.shape) // 250)
    for points, value in ((positive, cv2.GC_FGD), (negative, cv2.GC_BGD)):
        for x, y in map(_pixel, points):
            cv2.circle(seeds, (x - left, y - top), radius, value, -1)
    # A background circle can overlap an adjacent foreground point and vice versa.
    for points, value in ((positive, cv2.GC_FGD), (negative, cv2.GC_BGD)):
        for x, y in map(_pixel, points):
            seeds[y - top, x - left] = value
    if not np.any((seeds == cv2.GC_BGD) | (seeds == cv2.GC_PR_BGD)):
        raise ValueError("GrabCut 需要框外背景或背景提示點，請縮小提示框或新增背景點。")
    try:
        cv2.grabCut(rgb, seeds, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:
        raise RuntimeError("GrabCut 無法分出物件，請調整提示框或前景／背景點。") from error
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = (seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD)
    return mask


class SegmentationEngine:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir).resolve()
        self._lock = RLock()
        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"
        self._image_key: tuple[str, tuple[int, int]] | None = None
        self._embeddings = None

    def generate(
        self, engine: str, image: Image.Image, *, box: Box | None, positive: list[Point],
        negative: list[Point], image_key: str | None = None, cleanup: str = "balanced",
    ) -> SegmentationResult:
        """Return a mask for these prompts. A stable image_key must identify immutable RGB pixels."""
        started = perf_counter()
        if engine not in ("sam2", "grabcut"):
            raise ValueError("分割引擎必須為 SAM 2.1 或 GrabCut。")
        box, foreground, background = _check_prompts(image.size, box, positive, negative)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if engine == "grabcut":
            mask, score = _grabcut(image, box, foreground, background), None
        else:
            with self._lock:
                self._load_sam()
                try:
                    mask, score = self._predict_sam(image, box, foreground, background, image_key)
                except self._torch.OutOfMemoryError:
                    if self._device == "cpu":
                        raise
                    self._use_cpu()
                    mask, score = self._predict_sam(image, box, foreground, background, image_key)
        if not mask.any():
            raise RuntimeError("分割沒有找到前景，請調整提示框或新增前景點。")
        cleaned = clean_mask(mask, positive=foreground, negative=background, strength=cleanup)
        stats = {"removed_components": cleaned.removed_components, "filled_holes": cleaned.filled_holes,
                 "changed_pixels": cleaned.changed_pixels}
        return SegmentationResult(cleaned.mask, score, (perf_counter() - started) * 1000, mask, stats)

    def _load_sam(self) -> None:
        if self._model is not None:
            return
        required = ("config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json")
        if not all((self.model_dir / name).is_file() for name in required):
            raise RuntimeError(f"SAM 2.1 本機權重不完整。請執行安裝或選用 GrabCut。\n{self.model_dir}")
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as error:
            raise RuntimeError("SAM 2.1 執行環境未完整安裝。請執行安裝或選用 GrabCut。") from error
        self._torch = torch
        self._processor = Sam2Processor.from_pretrained(self.model_dir, local_files_only=True)
        self._model = Sam2Model.from_pretrained(self.model_dir, local_files_only=True, use_safetensors=True).eval()
        if torch.cuda.is_available():
            try:
                self._model.to(device="cuda:0", dtype=torch.float16)
                self._device = "cuda:0"
            except torch.OutOfMemoryError:
                self._use_cpu()

    def _use_cpu(self) -> None:
        self._image_key = None
        self._embeddings = None
        self._model.to(device="cpu", dtype=self._torch.float32)
        self._device = "cpu"
        self._torch.cuda.empty_cache()

    def _predict_sam(
        self, image: Image.Image, box: Box | None, positive: tuple[Point, ...],
        negative: tuple[Point, ...], image_key: str | None,
    ) -> tuple[np.ndarray, float | None]:
        torch, model, processor = self._torch, self._model, self._processor
        width, height = image.size
        key = (image_key, image.size) if image_key is not None else None
        cached = key is not None and key == self._image_key and self._embeddings is not None
        if cached:
            source = {"original_sizes": [[height, width]]}
        else:
            self._image_key = self._embeddings = None
            source = {"images": image}
        prompt = {}
        if box is not None:
            prompt["input_boxes"] = [[list(box)]]
        if positive or negative:
            prompt["input_points"] = [[[list(point) for point in positive + negative]]]
            prompt["input_labels"] = [[[1] * len(positive) + [0] * len(negative)]]

        inputs = processor(**source, **prompt, return_tensors="pt")
        original_sizes = inputs.pop("original_sizes")
        pixels = inputs.pop("pixel_values", None)
        autocast = torch.autocast("cuda", dtype=torch.float16) if self._device == "cuda:0" else nullcontext()
        with torch.inference_mode(), autocast:
            embeddings = self._embeddings if cached else model.get_image_embeddings(pixels.to(self._device))
            if key is not None and not cached:
                self._image_key, self._embeddings = key, embeddings
            inputs = inputs.to(self._device)
            output = model(**inputs, image_embeddings=embeddings, multimask_output=True)
        masks = processor.post_process_masks(output.pred_masks.float().cpu(), original_sizes)[0]
        if tuple(masks.shape[-2:]) != (height, width):
            raise RuntimeError("SAM 2.1 回傳的遮罩尺寸與圖片不符。")
        masks = masks.reshape(-1, height, width).numpy().astype(bool, copy=False)
        scores = output.iou_scores.float().cpu().reshape(-1).numpy()
        if len(masks) != len(scores):
            raise RuntimeError("SAM 2.1 回傳的遮罩與品質數量不符。")

        viable = [i for i, mask in enumerate(masks) if mask.any() and _honors_points(mask, positive, negative)]
        if box is not None:
            x1, y1, x2, y2 = box
            viable = [i for i in viable if masks[i, floor(y1):ceil(y2), floor(x1):ceil(x2)].any()]
        if not viable:
            raise RuntimeError("SAM 2.1 沒有產生符合這組提示的遮罩，請調整提示框或提示點。")
        best = max(viable, key=lambda i: float(scores[i]) if isfinite(scores[i]) else float("-inf"))
        return masks[best], float(scores[best]) if isfinite(scores[best]) else None
