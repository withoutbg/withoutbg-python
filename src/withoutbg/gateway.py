"""Community ONNX gateway. Vendored unchanged by Docker and the HF Space.

The release sidecar's ``gateway`` object describes three independent graphs.
All graphs accept float32 RGB [0, 1]; normalization lives inside the graphs.
Only the selected branch executes. A malformed bundle never falls back silently.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CATEGORY_PIPELINES = {
    "fine_strand": "matting",
    "soft_detail": "matting",
    "transparency": "matting",
    "hard_opaque": "birefnet",
    "flat_scene": "birefnet",
    "vehicle": "birefnet",
}


def validate_gateway(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported community gateway schema")
    categories = manifest.get("categories", [])
    if len(categories) != 6 or set(categories) != set(CATEGORY_PIPELINES):
        raise ValueError("Gateway must declare all six trained router categories")
    for name in ("router", "matting", "birefnet"):
        spec = manifest[name]
        path = Path(spec["file"])
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("Gateway model paths must be relative to the bundle")
        if spec["resize"] not in ("stretch", "letterbox"):
            raise ValueError("Unsupported gateway resize mode")
        if spec["interpolation"] not in ("bilinear", "bicubic"):
            raise ValueError("Unsupported gateway interpolation")
        if not isinstance(spec["canvas_size"], int) or spec["canvas_size"] <= 0:
            raise ValueError("Invalid gateway canvas size")
        digest = spec["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Gateway models require SHA256 checksums")
        if not spec["input_name"] or not spec["output_name"]:
            raise ValueError("Gateway models require named inputs and outputs")


def prepare_image(image, spec):
    canvas = spec["canvas_size"]
    interpolation = getattr(Image.Resampling, spec["interpolation"].upper())
    if spec["resize"] == "letterbox":
        scale = canvas / max(image.size)
        width, height = [max(1, round(d * scale)) for d in image.size]
        resized = image.resize((width, height), interpolation)
        prepared = Image.new("RGB", (canvas, canvas))
        prepared.paste(resized, (0, 0))
    else:
        width = height = canvas
        prepared = image.resize((canvas, canvas), interpolation)
    tensor = np.asarray(prepared, dtype=np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[None], (width, height)


class CommunityGateway:
    def __init__(self, manifest, resolve_file, providers=None, session_factory=None):
        validate_gateway(manifest)
        self.manifest = manifest
        self.resolve_file = resolve_file
        self.providers = providers or ["CPUExecutionProvider"]
        if session_factory is None:
            import onnxruntime as ort

            session_factory = ort.InferenceSession
        self.session_factory = session_factory
        self.sessions = {}
        self.lock = threading.RLock()

    def _session(self, name):
        with self.lock:
            if name not in self.sessions:
                spec = self.manifest[name]
                path = Path(self.resolve_file(spec["file"]))
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != spec["sha256"]:
                    raise ValueError(f"Gateway checksum mismatch: {name}")
                session = self.session_factory(str(path), providers=self.providers)
                if session.get_providers()[0] != self.providers[0]:
                    raise RuntimeError(f"Requested provider unavailable for {name}")
                self.sessions[name] = session
            return self.sessions[name]

    def preload(self):
        """Validate/load all assets so service readiness covers both branches."""
        for name in ("router", "matting", "birefnet"):
            self._session(name)

    def _run(self, name, image):
        spec = self.manifest[name]
        tensor, valid = prepare_image(image, spec)
        output = self._session(name).run(
            [spec["output_name"]], {spec["input_name"]: tensor}
        )[0]
        if not np.isfinite(output).all():
            raise ValueError(f"Non-finite gateway output: {name}")
        return output, valid

    def estimate_alpha(self, image):
        """Return an original-size L matte and routing metadata."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        with self.lock:
            logits, _ = self._run("router", image)
            if logits.shape != (1, len(self.manifest["categories"])):
                raise ValueError("Router output must be [1, 6] logits")
            index = int(np.argmax(logits[0]))
            category = self.manifest["categories"][index]
            pipeline = CATEGORY_PIPELINES[category]
            output, (width, height) = self._run(pipeline, image)
        if output.ndim != 4 or output.shape[:2] != (1, 1):
            raise ValueError("Branch output must be [1, 1, H, W] alpha")
        alpha = output[0, 0]
        canvas = self.manifest[pipeline]["canvas_size"]
        crop_w = max(1, round(width * alpha.shape[1] / canvas))
        crop_h = max(1, round(height * alpha.shape[0] / canvas))
        alpha = np.clip(alpha[:crop_h, :crop_w] * 255, 0, 255).astype(np.uint8)
        matte = Image.fromarray(alpha).resize(image.size, Image.Resampling.BILINEAR)
        return matte, {"category": category, "pipeline": pipeline}
