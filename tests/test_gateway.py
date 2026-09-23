"""Behavioral checks for routing, preprocessing, and bundle failures."""

import copy
import hashlib

import numpy as np
import pytest
from PIL import Image

from withoutbg.gateway import CATEGORY_PIPELINES, CommunityGateway, prepare_image


@pytest.fixture
def bundle(tmp_path):
    manifest = {"schema_version": 1, "categories": list(CATEGORY_PIPELINES)}
    for name in ("router", "matting", "birefnet"):
        data = name.encode()
        (tmp_path / f"{name}.onnx").write_bytes(data)
        manifest[name] = {
            "file": f"{name}.onnx",
            "sha256": hashlib.sha256(data).hexdigest(),
            "canvas_size": 8,
            "resize": "letterbox" if name == "matting" else "stretch",
            "interpolation": "bicubic" if name == "router" else "bilinear",
            "input_name": "rgb",
            "output_name": "logits" if name == "router" else "alpha",
        }
    return manifest, tmp_path


@pytest.mark.parametrize("category,pipeline", CATEGORY_PIPELINES.items())
def test_only_selected_branch_runs(bundle, category, pipeline):
    manifest, root = bundle
    # Exported checkpoint order, rather than a hardcoded numeric class mapping.
    manifest["categories"].reverse()
    calls = []

    class Session:
        def __init__(self, path, providers):
            self.name = path.split("/")[-1].split(".")[0]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, names, inputs):
            calls.append(self.name)
            assert inputs["rgb"].shape == (1, 3, 8, 8)
            if self.name == "router":
                logits = np.zeros((1, 6))
                logits[0, manifest["categories"].index(category)] = 10
                return [logits]
            # Different output resolution exercises scaled letterbox cropping.
            alpha = np.ones((1, 1, 4, 4), dtype=np.float32)
            if self.name == "matting":
                alpha[:, :, 2:, :] = 0  # Padding must not leak into output.
            return [alpha]

    gateway = CommunityGateway(
        manifest, lambda name: root / name, session_factory=Session
    )
    matte, route = gateway.estimate_alpha(Image.new("RGB", (20, 10), "white"))
    assert calls == ["router", pipeline]
    assert route == {"category": category, "pipeline": pipeline}
    assert matte.size == (20, 10)
    assert np.asarray(matte).min() == 255


def test_bad_checksum_fails_before_session_creation(bundle):
    manifest, root = bundle
    (root / "router.onnx").write_bytes(b"corrupt")
    gateway = CommunityGateway(
        manifest,
        lambda name: root / name,
        session_factory=lambda *a, **kw: pytest.fail("Loaded corrupt graph"),
    )
    with pytest.raises(ValueError, match="checksum"):
        gateway.estimate_alpha(Image.new("RGB", (8, 8)))


@pytest.mark.parametrize("mutation", ["version", "category", "path", "resize"])
def test_invalid_manifest_rejected(bundle, mutation):
    original, root = bundle
    manifest = copy.deepcopy(original)
    if mutation == "version":
        manifest["schema_version"] = 2
    elif mutation == "category":
        manifest["categories"][0] = "unknown"
    elif mutation == "path":
        manifest["router"]["file"] = "../escape.onnx"
    else:
        manifest["router"]["resize"] = "crop"
    with pytest.raises(ValueError):
        CommunityGateway(manifest, lambda name: root / name)


def test_stretch_and_letterbox_are_distinct(bundle):
    manifest, _ = bundle
    image = Image.new("RGB", (20, 10), "white")
    stretched, dimensions = prepare_image(image, manifest["router"])
    assert dimensions == (8, 8)
    assert stretched.min() == 1
    boxed, dimensions = prepare_image(image, manifest["matting"])
    assert dimensions == (8, 4)
    assert boxed[:, :, 4:].max() == 0
