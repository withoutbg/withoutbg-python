"""Routed open-weights pipeline (sidecar schema 3, pipeline ``routed``).

Matches the withoutBG reference host (``withoutbg-inference`` ``RoutedPipeline``):
a shared ConvNeXt backbone at 448² yields router logits and features. Fine
strands, soft detail and transparency go to the withoutBG matting graph (Depth
Anything V2 small depth + ConvNeXt-fused matting, reusing those features); hard
opaque objects, flat scenes and vehicles go to BiRefNet at 1024². Only the
selected branch runs; its alpha is bilinearly upsampled to native resolution.

Every graph input is a square stretch of the whole working image, float32 RGB
in [0, 1], NCHW; normalization lives inside the graphs. Graphs are resolved and
SHA256-checked lazily, so the BiRefNet graph is fetched only when needed.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

SCHEMA_VERSION = 3
PIPELINE_NAME = "routed"
GRAPHS = ("router", "coarse", "birefnet")

SessionFactory = Callable[..., Any]


def validate_sidecar(sidecar: dict[str, Any]) -> None:
    """Reject anything but a well-formed open-weights ``routed`` bundle."""
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported open-weights bundle schema")
    pipeline = sidecar.get("pipeline")
    if pipeline == "routed_edge_refine" or "refiner" in sidecar:
        raise ValueError(
            "Edge-refine bundles are withoutBG Enterprise and not supported by this SDK"
        )
    if pipeline != PIPELINE_NAME:
        raise ValueError(f"Unsupported open-weights pipeline: {pipeline!r}")
    for name in GRAPHS:
        spec = sidecar[name]
        path = Path(spec["file"])
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("Bundle model paths must be relative to the bundle")
        digest = spec["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Bundle models require SHA256 checksums")
    router = sidecar["router"]
    if not set(router["birefnet_categories"]) <= set(router["categories"]):
        raise ValueError("BiRefNet categories must be router categories")


def fit_max_size(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Downscale to fit ``max_width × max_height``, keeping the aspect ratio."""
    w, h = image.size
    ar = w / h
    resize = False
    if w > max_width:
        w, h, resize = max_width, int(max_width / ar), True
    if h > max_height:
        h, w, resize = max_height, int(max_height * ar), True
    return image.resize((w, h)) if resize else image


def _bilinear_axis(src: int, dst: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source indices and weights of torch bilinear (``align_corners=False``)."""
    pos = np.maximum((np.arange(dst) + 0.5) * (src / dst) - 0.5, 0.0)
    i0 = pos.astype(np.int64)
    i1 = np.minimum(i0 + 1, src - 1)
    return i0, i1, (pos - i0).astype(np.float32)


def resize_bilinear(x: np.ndarray, height: int, width: int) -> np.ndarray:
    """``(C, H, W)`` float32 → ``(C, height, width)`` like ``F.interpolate``."""
    _, h, w = x.shape
    if (h, w) == (height, width):
        return x
    r0, r1, rl = _bilinear_axis(h, height)
    c0, c1, cl = _bilinear_axis(w, width)
    rows = x[:, r0] * (1.0 - rl)[:, None] + x[:, r1] * rl[:, None]
    out: np.ndarray = rows[:, :, c0] * (1.0 - cl) + rows[:, :, c1] * cl
    return out


def resize_bicubic_antialias(rgb: np.ndarray, size: int) -> np.ndarray:
    """``(H, W, 3)`` → ``(3, size, size)``; PIL "F" bicubic = torch antialias."""
    return np.stack(
        [
            np.asarray(
                Image.fromarray(np.ascontiguousarray(rgb[..., c], np.float32)).resize(
                    (size, size), Image.Resampling.BICUBIC
                )
            )
            for c in range(3)
        ]
    )


def graph_inputs(
    rgb: np.ndarray,
    graph_spec: dict[str, Any],
    cached: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Graph feeds from native ``(H, W, 3)`` float32 RGB in ``[0, 1]``.

    Feeds already in *cached* (same name, same resize spec) are reused.
    """
    feeds = {}
    for name, spec in graph_spec["inputs"].items():
        if cached is not None and name in cached:
            feeds[name] = cached[name]
            continue
        size = int(spec["size"])
        if spec["interpolation"] == "bicubic" and spec["antialias"]:
            x = resize_bicubic_antialias(rgb, size)
        elif spec["interpolation"] == "bilinear" and not spec["antialias"]:
            x = resize_bilinear(
                np.ascontiguousarray(rgb.transpose(2, 0, 1)), size, size
            )
        else:
            raise ValueError(f"Unsupported resize for {name}: {spec}")
        feeds[name] = np.ascontiguousarray(x[None], dtype=np.float32)
    return feeds


class RoutedPipeline:
    """Router → withoutBG matting or BiRefNet, alpha upsampled to native size."""

    def __init__(
        self,
        sidecar: dict[str, Any],
        resolve_file: Callable[[str], Path],
        providers: list[str] | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        validate_sidecar(sidecar)
        self.sidecar = sidecar
        self.resolve_file = resolve_file
        self.providers = providers or ["CPUExecutionProvider"]
        if session_factory is None:
            import onnxruntime as ort  # type: ignore

            session_factory = ort.InferenceSession
        self.session_factory = session_factory
        self.sessions: dict[str, Any] = {}
        router = sidecar["router"]
        self.categories: list[str] = list(router["categories"])
        self.birefnet_categories = set(router["birefnet_categories"])
        max_w, max_h = sidecar["max_inference_size"]
        self.max_size = (int(max_w), int(max_h))
        self.lock = threading.RLock()

    def _session(self, name: str) -> Any:
        with self.lock:
            if name not in self.sessions:
                spec = self.sidecar[name]
                path = Path(self.resolve_file(spec["file"]))
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != spec["sha256"]:
                    raise ValueError(f"Bundle checksum mismatch: {name}")
                session = self.session_factory(str(path), providers=self.providers)
                if session.get_providers()[0] != self.providers[0]:
                    raise RuntimeError(f"Requested provider unavailable for {name}")
                self.sessions[name] = session
            return self.sessions[name]

    def preload(self) -> None:
        """Resolve, verify and load every graph so both branches are ready."""
        for name in GRAPHS:
            self._session(name)

    def _route(
        self, rgb: np.ndarray
    ) -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]:
        spec = self.sidecar["router"]
        feeds = graph_inputs(rgb, spec)
        logits, *feats = self._session("router").run(
            [spec["logits_name"], *spec["feature_names"]], feeds
        )
        if logits.shape != (1, len(self.categories)) or not np.isfinite(logits).all():
            raise ValueError("Router output must be finite [1, n_categories] logits")
        category = self.categories[int(np.argmax(logits[0]))]
        return category, feeds, dict(zip(spec["feature_names"], feats))

    def _coarse(
        self,
        rgb: np.ndarray,
        feeds: dict[str, np.ndarray],
        feats: dict[str, np.ndarray],
    ) -> np.ndarray:
        spec = self.sidecar["coarse"]
        # The backbone already resized rgb_matting identically; reuse it.
        coarse_feeds = graph_inputs(rgb, spec, cached=feeds)
        coarse_feeds.update(feats)
        out: np.ndarray = self._session("coarse").run(
            [spec["output_name"]], coarse_feeds
        )[0][0]
        return out

    def _birefnet(self, rgb: np.ndarray) -> np.ndarray:
        spec = self.sidecar["birefnet"]
        out: np.ndarray = self._session("birefnet").run(
            [spec["output_name"]], graph_inputs(rgb, spec)
        )[0][0]
        return out

    def estimate_alpha(self, image: Image.Image) -> tuple[Image.Image, dict[str, str]]:
        """``L`` alpha matte at *image*'s size plus ``{"category", "pipeline"}``."""
        orig_size = image.size
        work = fit_max_size(image.convert("RGB"), *self.max_size)
        rgb = np.asarray(work, dtype=np.float32) / 255.0
        h, w = rgb.shape[:2]
        with self.lock:
            category, feeds, feats = self._route(rgb)
            pipeline = "birefnet" if category in self.birefnet_categories else "matting"
            if pipeline == "birefnet":
                coarse = self._birefnet(rgb)
            else:
                coarse = self._coarse(rgb, feeds, feats)
        if coarse.ndim != 3 or coarse.shape[0] != 1 or not np.isfinite(coarse).all():
            raise ValueError(f"{pipeline} output must be finite [1, H, W] alpha")
        alpha = np.clip(resize_bilinear(coarse, h, w), 0.0, 1.0)[0]
        matte = Image.fromarray(np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8))
        if matte.size != orig_size:
            matte = matte.resize(orig_size, Image.Resampling.BILINEAR)
        return matte, {"category": category, "pipeline": pipeline}
