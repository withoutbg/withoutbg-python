"""Behavioral checks for the routed bundle: routing, feeds, and bundle failures."""

import copy
import hashlib

import numpy as np
import pytest
from PIL import Image

from withoutbg.routed import (
    RoutedPipeline,
    fit_max_size,
    resize_bilinear,
    validate_sidecar,
)

CATEGORIES = [
    "fine_strand",
    "soft_detail",
    "transparency",
    "hard_opaque",
    "flat_scene",
    "vehicle",
]
BIREFNET_CATEGORIES = ["hard_opaque", "flat_scene", "vehicle"]
FEATURES = ["f0", "f1", "f2", "f3"]
BILINEAR = {"resize": "stretch", "interpolation": "bilinear", "antialias": False}
BICUBIC_AA = {"resize": "stretch", "interpolation": "bicubic", "antialias": True}


@pytest.fixture
def bundle(tmp_path):
    files = {
        "router": "withoutbg-open-weights-backbone.onnx",
        "coarse": "withoutbg-open-weights.onnx",
        "birefnet": "birefnet-general.onnx",
    }
    sidecar = {
        "schema_version": 3,
        "pipeline": "routed",
        "variant": "oss",
        "max_inference_size": [64, 64],
        "router": {
            "inputs": {"rgb_matting": {"size": 8, **BILINEAR}},
            "logits_name": "route_logits",
            "feature_names": FEATURES,
            "categories": list(CATEGORIES),
            "birefnet_categories": BIREFNET_CATEGORIES,
        },
        "coarse": {
            "inputs": {
                "rgb_depth": {"size": 10, **BICUBIC_AA},
                "rgb_matting": {"size": 8, **BILINEAR},
            },
            "feature_names": FEATURES,
            "output_name": "coarse_alpha",
        },
        "birefnet": {
            "inputs": {"rgb": {"size": 12, **BILINEAR}},
            "output_name": "alpha",
        },
    }
    for name, filename in files.items():
        data = name.encode()
        (tmp_path / filename).write_bytes(data)
        sidecar[name]["file"] = filename
        sidecar[name]["sha256"] = hashlib.sha256(data).hexdigest()
    return sidecar, tmp_path


def make_session_factory(sidecar, category, calls):
    graph_of = {sidecar[n]["file"]: n for n in ("router", "coarse", "birefnet")}

    class Session:
        def __init__(self, path, providers):
            self.graph = graph_of[path.replace("\\", "/").split("/")[-1]]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, names, feeds):
            calls.append((self.graph, {k: v.shape for k, v in feeds.items()}))
            if self.graph == "router":
                logits = np.zeros((1, 6), dtype=np.float32)
                logits[0, sidecar["router"]["categories"].index(category)] = 10
                return [logits] + [
                    np.full((1, 2, 2, 2), i, np.float32) for i in range(4)
                ]
            value = 0.25 if self.graph == "coarse" else 0.75
            return [np.full((1, 1, 4, 4), value, dtype=np.float32)]

    return Session


@pytest.mark.parametrize("category", CATEGORIES)
def test_only_selected_branch_runs(bundle, category):
    sidecar, root = bundle
    # Sidecar category order decides the mapping, not a hardcoded index.
    sidecar["router"]["categories"].reverse()
    calls = []
    pipeline = RoutedPipeline(
        sidecar,
        lambda name: root / name,
        session_factory=make_session_factory(sidecar, category, calls),
    )

    matte, route = pipeline.estimate_alpha(Image.new("RGB", (20, 10), (90, 40, 10)))

    expected = "birefnet" if category in BIREFNET_CATEGORIES else "matting"
    assert route == {"category": category, "pipeline": expected}
    assert [graph for graph, _ in calls] == [
        "router",
        "birefnet" if expected == "birefnet" else "coarse",
    ]
    assert set(pipeline.sessions) == {"router", calls[1][0]}
    assert matte.mode == "L" and matte.size == (20, 10)
    value = 191 if expected == "birefnet" else 64
    assert np.all(np.asarray(matte) == value)


def test_matting_gets_depth_input_reused_matting_input_and_features(bundle):
    sidecar, root = bundle
    calls = []
    pipeline = RoutedPipeline(
        sidecar,
        lambda name: root / name,
        session_factory=make_session_factory(sidecar, "fine_strand", calls),
    )
    pipeline.estimate_alpha(Image.new("RGB", (20, 10)))

    assert calls[0] == ("router", {"rgb_matting": (1, 3, 8, 8)})
    assert calls[1] == (
        "coarse",
        {
            "rgb_depth": (1, 3, 10, 10),
            "rgb_matting": (1, 3, 8, 8),
            **dict.fromkeys(FEATURES, (1, 2, 2, 2)),
        },
    )


def test_birefnet_gets_its_own_stretch(bundle):
    sidecar, root = bundle
    calls = []
    pipeline = RoutedPipeline(
        sidecar,
        lambda name: root / name,
        session_factory=make_session_factory(sidecar, "vehicle", calls),
    )
    pipeline.estimate_alpha(Image.new("RGB", (20, 10)))

    assert calls[1] == ("birefnet", {"rgb": (1, 3, 12, 12)})


def test_preload_loads_every_graph(bundle):
    sidecar, root = bundle
    pipeline = RoutedPipeline(
        sidecar,
        lambda name: root / name,
        session_factory=make_session_factory(sidecar, "vehicle", []),
    )
    pipeline.preload()

    assert set(pipeline.sessions) == {"router", "coarse", "birefnet"}


def test_bad_checksum_fails_before_session_creation(bundle):
    sidecar, root = bundle
    (root / sidecar["router"]["file"]).write_bytes(b"corrupt")

    def factory(*args, **kwargs):
        raise AssertionError("session must not be created")

    pipeline = RoutedPipeline(
        sidecar, lambda name: root / name, session_factory=factory
    )
    with pytest.raises(ValueError, match="checksum"):
        pipeline.estimate_alpha(Image.new("RGB", (4, 4)))


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda s: s.update(schema_version=1), "schema"),
        (lambda s: s.update(pipeline="routed_edge_refine"), "Enterprise"),
        (lambda s: s.update(refiner={"file": "r.onnx"}), "Enterprise"),
        (lambda s: s.update(pipeline="coarse_edge_refine"), "pipeline"),
        (lambda s: s["coarse"].update(file="/tmp/x.onnx"), "relative"),
        (lambda s: s["birefnet"].update(file="../x.onnx"), "relative"),
        (lambda s: s["router"].update(sha256="abc"), "SHA256"),
        (lambda s: s["router"].update(birefnet_categories=["cars"]), "categories"),
    ],
)
def test_invalid_sidecar_rejected(bundle, mutation, message):
    sidecar = copy.deepcopy(bundle[0])
    mutation(sidecar)
    with pytest.raises(ValueError, match=message):
        validate_sidecar(sidecar)


def test_large_images_fit_max_size_and_matte_keeps_original_size(bundle):
    sidecar, root = bundle
    pipeline = RoutedPipeline(
        sidecar,
        lambda name: root / name,
        session_factory=make_session_factory(sidecar, "vehicle", []),
    )
    matte, _ = pipeline.estimate_alpha(Image.new("RGB", (200, 50)))

    assert matte.size == (200, 50)
    assert fit_max_size(Image.new("RGB", (200, 50)), 64, 64).size == (64, 16)
    assert fit_max_size(Image.new("RGB", (50, 200)), 64, 64).size == (16, 64)
    assert fit_max_size(Image.new("RGB", (30, 20)), 64, 64).size == (30, 20)


def test_resize_bilinear_matches_torch_half_pixel_centers():
    x = np.array([[[0.0, 1.0]]], dtype=np.float32)  # (C=1, H=1, W=2)
    out = resize_bilinear(x, 1, 4)
    # F.interpolate(mode="bilinear", align_corners=False) → [0, .25, .75, 1]
    np.testing.assert_allclose(out[0, 0], [0.0, 0.25, 0.75, 1.0])
