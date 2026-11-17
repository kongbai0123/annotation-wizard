"""Conservative native-resolution cleanup for an instance mask."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import cv2
import numpy as np


@dataclass(frozen=True)
class CleanupResult:
    mask: np.ndarray
    removed_components: int
    filled_holes: int
    changed_pixels: int


_RING = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_SQUARE = np.ones((3, 3), dtype=np.uint8)
_RING_KERNEL = _SQUARE.copy()
_RING_KERNEL[1, 1] = 0


def _connected_ring(indices: list[int], diagonal: bool) -> bool:
    if not indices:
        return False
    unseen = {_RING[index] for index in indices}
    pending = [unseen.pop()]
    while pending:
        y, x = pending.pop()
        neighbors = [pair for pair in unseen if (
            max(abs(y - pair[0]), abs(x - pair[1])) == 1 if diagonal else
            abs(y - pair[0]) + abs(x - pair[1]) == 1
        )]
        unseen.difference_update(neighbors)
        pending.extend(neighbors)
    return not unseen


def _simple_points() -> np.ndarray:
    # Permitting a toggle only with both local connectivities intact keeps a
    # smoothing pass from severing a thin bar or closing a real opening.
    return np.array([
        _connected_ring([i for i in range(8) if value & (1 << i)], diagonal=True)
        and _connected_ring([i for i in range(8) if not value & (1 << i)], diagonal=False)
        for value in range(256)
    ])


_SIMPLE = _simple_points()


def _prompt_pixels(points, height: int, width: int) -> list[tuple[int, int]]:
    result = []
    for x, y in points:
        if isfinite(x) and isfinite(y) and 0 <= x < width and 0 <= y < height:
            result.append((int(y), int(x)))
    return result


def _thin_nearby_fragment(component: np.ndarray, distance: np.ndarray, span_limit: float) -> bool:
    """Leave plausible short breaks in slender object details for human review."""
    y, x = np.nonzero(component)
    if y.size < 7 or max(np.ptp(y), np.ptp(x)) < 5 or distance[y, x].min() > span_limit:
        return False
    short, long = np.linalg.eigvalsh(np.cov(np.stack((x, y))))
    return long >= 7 * max(short, 0.25)


def _remove_specks(mask: np.ndarray, prompts: list[tuple[int, int]], limit: int, budget: int, span: float):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    anchored = areas > limit
    anchored[0] = False
    anchored[1 + int(np.argmax(areas[1:]))] = True
    for y, x in prompts:
        if labels[y, x]:
            anchored[labels[y, x]] = True

    small = np.flatnonzero(~anchored)[1:]
    if small.size == 0:
        return 0, 0
    distance = cv2.distanceTransform((~anchored[labels]).astype(np.uint8), cv2.DIST_L2, 3)
    deleted, pixels = 0, 0
    for index in small[np.argsort(areas[small], kind="stable")]:
        x, y, w, h, area = map(int, stats[index])
        if pixels + area > budget:
            break
        # The decision is local to the component's bounding rectangle.
        roi = np.s_[y:y + h, x:x + w]
        component = labels[roi] == index
        if _thin_nearby_fragment(component, distance[roi], span):
            continue
        mask[roi][component] = 0
        deleted += 1
        pixels += area
    return deleted, pixels


def _fill_pinholes(mask: np.ndarray, prompts: list[tuple[int, int]], limit: int, max_span: int, budget: int):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    important = {int(labels[y, x]) for y, x in prompts}
    height, width = mask.shape
    filled, pixels = 0, 0
    for index in sorted(range(1, len(stats)), key=lambda item: stats[item, cv2.CC_STAT_AREA]):
        x, y, w, h, area = map(int, stats[index])
        if area > limit or pixels + area > budget:
            break
        if (index in important or x == 0 or y == 0 or x + w == width or y + h == height or
                max(w, h) > max_span):
            continue
        roi = np.s_[y:y + h, x:x + w]
        mask[roi][labels[roi] == index] = 1
        filled += 1
        pixels += area
    return filled, pixels


def _smooth_edge(mask: np.ndarray, protected: np.ndarray, *, aggressive: bool, budget: int) -> None:
    if min(mask.shape) < 3 or budget <= 0:
        return
    core = cv2.erode(mask, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    near_core = cv2.dilate(core, _SQUARE, iterations=2) != 0
    # Centers of any one parity do not occur in one another's 3x3 window.
    for offset_y, offset_x in ((0, 0), (1, 1), (0, 1), (1, 0)):
        count = cv2.filter2D(mask, -1, _RING_KERNEL, borderType=cv2.BORDER_CONSTANT)
        trim = (mask != 0) & (count <= (3 if aggressive else 2)) & near_core
        fill = (mask == 0) & (count >= (6 if aggressive else 7))
        candidates = (trim | fill) & ~protected
        candidates[[0, -1], :] = False
        candidates[:, [0, -1]] = False
        y, x = np.nonzero(candidates[offset_y::2, offset_x::2])
        y, x = 2 * y + offset_y, 2 * x + offset_x
        if y.size == 0:
            continue
        pattern = np.zeros(y.size, dtype=np.uint8)
        for index, (dy, dx) in enumerate(_RING):
            pattern |= mask[y + dy, x + dx] << index
        valid = _SIMPLE[pattern]
        y, x = y[valid], x[valid]
        if y.size > budget:
            # An unusually rough slice can be reviewed without automatic edits.
            continue
        mask[y, x] ^= 1
        budget -= y.size
        if budget == 0:
            return


def clean_mask(mask: np.ndarray, *, positive=(), negative=(), strength: str = "balanced") -> CleanupResult:
    """Remove small specks, fill pinholes, and smooth the boundary around prompts.

    Original-image prompts use (x, y) coordinates. The returned boolean mask
    is always independent of the input. Larger holes and separated parts are
    retained for review. ``off`` (alias ``none``) returns the raw mask.
    """
    if strength not in {"off", "none", "light", "balanced"}:
        raise ValueError(f"Unknown mask cleanup strength: {strength!r}")
    original = np.asarray(mask, dtype=bool)
    if original.ndim != 2:
        raise ValueError("Mask cleanup requires a two-dimensional mask")
    result = original.copy()
    if strength in {"off", "none"} or not result.any():
        return CleanupResult(result, 0, 0, 0)

    # Labels and morphological passes need only see the foreground's bounding
    # rectangle plus a small, untouched border of surrounding background.
    x, y, width, height = cv2.boundingRect(result.astype(np.uint8))
    y0, x0 = max(y - 3, 0), max(x - 3, 0)
    y1, x1 = min(y + height + 3, result.shape[0]), min(x + width + 3, result.shape[1])
    crop = np.s_[y0:y1, x0:x1]
    work = np.ascontiguousarray(original[crop], dtype=np.uint8)
    area = int(work.sum())
    balanced = strength == "balanced"
    speck_limit = min(256, max(2, int(area * 0.0018))) if balanced else min(64, max(1, int(area * 0.00045)))
    hole_limit = min(64, max(1, int(area * 0.00055))) if balanced else min(16, max(1, int(area * 0.00015)))
    max_span = max(2, int(2 * sqrt(hole_limit)))
    budget = min(8000, int(area * (0.035 if balanced else 0.012)))
    if budget == 0:
        return CleanupResult(result, 0, 0, 0)
    prompt_pixels = []
    for prompts in (positive, negative):
        prompt_pixels.extend((py - y0, px - x0) for py, px in _prompt_pixels(prompts, *original.shape)
                             if y0 <= py < y1 and x0 <= px < x1)
    protected = np.zeros_like(work)
    for py, px in prompt_pixels:
        protected[py, px] = 1

    removed, removed_pixels = _remove_specks(work, prompt_pixels, speck_limit, budget,
                                               min(12, max(3, sqrt(area) * 0.025)))
    filled, filled_pixels = _fill_pinholes(work, prompt_pixels, hole_limit, max_span, budget - removed_pixels)
    _smooth_edge(work, cv2.dilate(protected, _SQUARE) != 0, aggressive=balanced,
                 budget=budget - removed_pixels - filled_pixels)
    result[crop] = work != 0
    changed = np.count_nonzero(result[crop] != original[crop])
    return CleanupResult(result, removed, filled, int(changed))
