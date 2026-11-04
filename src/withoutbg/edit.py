"""Smart brush edit (open weights 10.12.4+, sidecar ``stroke_edit``).

The user paints what to keep and what to remove on a matte. The stroke-edit
graph maps RGB + matte + strokes + depth at 448² to the edited matte; it is
bilinearly upsampled and replaces the matte only in the region the strokes
reach (:func:`stroke_region`). Only Remove is clamped: a Keep stroke swiped
across an object onto the background around it must not paint that background
solid.

Masks follow the withoutBG API: a pixel is marked when it is light and opaque
(``>= 128`` composited onto black); where Keep and Remove both mark a pixel,
Remove wins. Internally the strokes are one float32 ``(H, W)`` map with values
``{0, 0.5, 1}`` (remove, untouched, keep).

Same code as the withoutBG reference host, so results match it exactly.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

REMOVE = 0.0
UNTOUCHED = 0.5
KEEP = 1.0
MARKED_MIN = 128

STROKE_RADIUS_FRAC = 0.02
STROKE_RADIUS_MIN = 8
CHANGE_MIN = 0.1
"""The edit counts as changing a pixel where it moves the matte by more than this."""

CHANNELS = ("r", "g", "b", "matte", "keep", "remove", "depth")


class EditInputError(ValueError):
    """A mask or the matte is unreadable or does not match the image."""


def _flatten(img: Image.Image) -> Image.Image:
    """``L`` image of *img* composited onto black, so transparent pixels read as 0."""
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        return Image.alpha_composite(
            Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba
        ).convert("L")
    return img.convert("L")


def _checked(img: Image.Image, size: tuple[int, int], what: str) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.size != tuple(size):
        raise EditInputError(
            f"{what} size {img.size[0]}x{img.size[1]} does not match image size {size[0]}x{size[1]}"
        )
    return img


def mask_from_image(
    mask: Image.Image, size: tuple[int, int], what: str = "Mask"
) -> np.ndarray:
    """Bool ``(H, W)`` marks of a Smart brush mask; *size* is the photo's ``(w, h)`` after EXIF."""
    return np.asarray(_flatten(_checked(mask, size, what))) >= MARKED_MIN


def matte_from_image(matte: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Float32 ``(H, W)`` matte in ``[0, 1]`` (alpha dropped, converted to ``L``)."""
    return (
        np.asarray(_checked(matte, size, "Matte").convert("L"), dtype=np.float32)
        / 255.0
    )


def combine_masks(
    keep: np.ndarray | None, remove: np.ndarray | None, shape: tuple[int, int]
) -> np.ndarray:
    """Bool masks → float32 ``{0, 0.5, 1}`` strokes; Remove wins where both mark a pixel."""
    out = np.full(shape, UNTOUCHED, dtype=np.float32)
    if keep is not None:
        out[keep] = KEEP
    if remove is not None:
        out[remove] = REMOVE
    return out


def net_clamp(strokes: np.ndarray) -> np.ndarray:
    """The strokes the edit clamps: Remove only (Keep marks become untouched)."""
    return np.where(strokes == KEEP, np.float32(UNTOUCHED), strokes).astype(np.float32)


def apply_strokes(alpha: np.ndarray, clamp: np.ndarray) -> np.ndarray:
    """Force *alpha* (``(..., H, W)``) to 1 under Keep and 0 under Remove in *clamp*."""
    alpha = np.where(clamp == KEEP, np.float32(1.0), alpha)
    return np.where(clamp == REMOVE, np.float32(0.0), alpha).astype(np.float32)


def resize_area(x: np.ndarray, size: int) -> np.ndarray:
    """``(C, H, W)`` → ``(C, size, size)`` like ``F.interpolate(mode="area")`` (adaptive average pool)."""

    def axis(v: np.ndarray, ax: int) -> np.ndarray:
        n = v.shape[ax]
        if n == size:
            return v
        i = np.arange(size)
        start = (i * n) // size
        end = -((-(i + 1) * n) // size)
        csum = np.concatenate(
            [np.zeros_like(np.take(v, [0], axis=ax)), np.cumsum(v, axis=ax)], axis=ax
        )
        sums = np.take(csum, end, axis=ax) - np.take(csum, start, axis=ax)
        shape = [1] * v.ndim
        shape[ax] = size
        return sums / (end - start).reshape(shape)

    return axis(axis(x.astype(np.float64), 1), 2).astype(np.float32)


def soften_hard_strokes(marks: np.ndarray) -> np.ndarray:
    """Tent-filter (``[1, 2, 1] / 4``, edge-replicated) ``(C, S, S)`` marks that have no partial pixels.

    Training strokes are area-downsampled from 2×, so their edges are soft; an
    image already at the net's size would give hard 0/1 edges the net is brittle on.
    """
    if ((marks > 0) & (marks < 1)).any():
        return marks
    p = np.pad(marks, ((0, 0), (0, 0), (1, 1)), mode="edge")
    soft = (p[:, :, :-2] + 2 * p[:, :, 1:-1] + p[:, :, 2:]) / 4
    p = np.pad(soft, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return ((p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4).astype(np.float32)


def stroke_edit_input(
    rgb: np.ndarray,
    matte: np.ndarray,
    strokes: np.ndarray,
    depth: np.ndarray,
    spec: dict,
) -> np.ndarray:
    """``(1, 7, S, S)`` graph input from native ``(H, W, 3)`` *rgb*, ``(H, W)`` *matte* / *strokes*.

    *depth* is the coarse graph's ``edit_depth`` (normalised inverse depth, ``S``²).
    """
    if (
        tuple(spec["channels"]) != CHANNELS
        or spec["resize"] != "area"
        or spec["soften_hard_strokes"] != "tent_1_2_1"
    ):
        raise ValueError("Unsupported stroke-edit input spec")
    size = int(spec["size"])
    native = np.concatenate(
        [
            rgb.transpose(2, 0, 1),
            np.clip(matte, 0, 1)[None],
            (strokes == KEEP)[None].astype(np.float32),
            (strokes == REMOVE)[None].astype(np.float32),
        ]
    ).astype(np.float32)
    x = resize_area(native, size)
    d = np.asarray(depth, dtype=np.float32).reshape(-1, *np.shape(depth)[-2:])[:1]
    if d.shape[-2:] != (size, size):
        raise ValueError(f"edit depth must be {size}x{size}; got {d.shape[-2:]}")
    parts = [x[:4], soften_hard_strokes(x[4:6]), np.clip(d, 0, 1)]
    return np.concatenate(parts)[None].astype(np.float32)


def stroke_radius(height: int, width: int) -> int:
    return max(STROKE_RADIUS_MIN, int(round(STROKE_RADIUS_FRAC * max(height, width))))


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not mask.any():
        return mask
    return (
        cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    )


def stroke_region(
    matte: np.ndarray, guided: np.ndarray, strokes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(region, weight)`` for ``(H, W)`` *matte*, edited *guided* and *strokes*.

    ``region``: pixels near a stroke, grown by the 8-connected pieces the edit
    changes that they touch (one stroke on an object reaches all of it, a
    change elsewhere does not), plus the same margin. ``weight`` feathers it
    to 0 over its outer half margin: the blend weight of the edit.
    """
    h, w = matte.shape
    r = stroke_radius(h, w)
    seeds = _dilate(strokes != UNTOUCHED, r)
    if not seeds.any():
        empty = np.zeros((h, w), dtype=bool)
        return empty, empty.astype(np.float32)
    grow = seeds | (np.abs(guided - matte) > CHANGE_MIN)
    _, labels = cv2.connectedComponents(grow.astype(np.uint8), connectivity=8)
    region = _dilate(np.isin(labels, np.unique(labels[seeds])) & grow, r)
    inside = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    return region, np.clip(inside / max(1.0, r / 2), 0.0, 1.0).astype(np.float32)


def edit_alpha(
    rgb: np.ndarray,
    matte: np.ndarray,
    strokes: np.ndarray,
    depth: np.ndarray,
    spec: dict,
    run_editor: Callable[[np.ndarray], np.ndarray],
    upsample: Callable[[np.ndarray, int, int], np.ndarray],
    refine: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Edited ``(H, W)`` alpha for native ``(H, W, 3)`` *rgb*, previous *matte* and *strokes*.

    *run_editor* runs the stroke-edit graph (``(1, 7, S, S)`` → ``(1, 1, S, S)``),
    *upsample* is the host's bilinear resize (``(C, s, s), h, w → (C, h, w)``),
    and *refine*, when the bundle ships an edge refiner, maps the upsampled
    ``(1, H, W)`` edit and the clamp strokes to the refined ``(1, H, W)`` alpha.
    Outside :func:`stroke_region` the result is *matte*.
    """
    h, w = matte.shape
    if rgb.shape[:2] != (h, w) or strokes.shape != (h, w):
        raise ValueError(
            f"edit expects matching sizes; got rgb {rgb.shape}, matte {matte.shape}, strokes {strokes.shape}"
        )
    matte = np.clip(matte, 0, 1).astype(np.float32)
    coarse = run_editor(stroke_edit_input(rgb, matte, strokes, depth, spec))
    if not np.isfinite(coarse).all():
        raise ValueError("Non-finite stroke-edit output")
    coarse = np.clip(
        np.asarray(coarse, dtype=np.float32).reshape(1, *coarse.shape[-2:]), 0, 1
    )
    clamp = net_clamp(strokes)
    up = upsample(coarse, h, w)
    guided = apply_strokes(up, clamp)[0]
    alpha = refine(np.clip(up, 0, 1), clamp)[0] if refine is not None else up[0]
    _, weight = stroke_region(matte, guided, strokes)
    alpha = matte + weight * (np.clip(alpha, 0, 1) - matte)
    return apply_strokes(alpha, clamp)


def resize_nearest(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest resize of ``(H, W)`` to ``(h, w)`` (index rule of ``F.interpolate(mode="nearest")``)."""
    ih, iw = x.shape
    if (ih, iw) == (h, w):
        return x
    rows = np.minimum((np.arange(h) * (ih / h)).astype(np.int64), ih - 1)
    cols = np.minimum((np.arange(w) * (iw / w)).astype(np.int64), iw - 1)
    return x[rows[:, None], cols[None, :]]


def clamp_matte(matte: Image.Image, clamp: np.ndarray) -> Image.Image:
    """``L`` *matte* forced to 255 under Keep and 0 under Remove in same-size *clamp*."""
    arr = np.asarray(matte, dtype=np.uint8).copy()
    arr[clamp == KEEP] = 255
    arr[clamp == REMOVE] = 0
    return Image.fromarray(arr)
