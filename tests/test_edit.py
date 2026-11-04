"""Keep / Remove edit: mask contract, routed edit pass, sidecar validation."""

import copy
import hashlib

import numpy as np
import pytest
from PIL import Image

from withoutbg.edit import (
    KEEP,
    REMOVE,
    UNTOUCHED,
    EditInputError,
    combine_masks,
    mask_from_image,
    net_clamp,
    resize_nearest,
    stroke_edit_input,
)
from withoutbg.routed import RoutedPipeline, validate_sidecar

from .test_routed import BICUBIC_AA, BILINEAR, CATEGORIES, FEATURES

STROKE_SPEC = {
    "input_name": "stroke_input",
    "output_name": "edited_alpha",
    "size": 16,
    "channels": ["r", "g", "b", "matte", "keep", "remove", "depth"],
    "resize": "area",
    "soften_hard_strokes": "tent_1_2_1",
    "depth": "coarse.edit_depth",
}


@pytest.fixture
def edit_bundle(tmp_path):
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
            "birefnet_categories": ["hard_opaque", "flat_scene", "vehicle"],
        },
        "coarse": {
            "inputs": {
                "rgb_depth": {"size": 10, **BICUBIC_AA},
                "rgb_matting": {"size": 8, **BILINEAR},
            },
            "feature_names": FEATURES,
            "output_name": "coarse_alpha",
            "edit_depth_name": "edit_depth",
        },
        "birefnet": {
            "inputs": {"rgb": {"size": 12, **BILINEAR}},
            "output_name": "alpha",
        },
        "stroke_edit": dict(STROKE_SPEC),
    }
    for name in ("router", "coarse", "birefnet", "stroke_edit"):
        data = name.encode()
        (tmp_path / f"{name}.onnx").write_bytes(data)
        sidecar[name]["file"] = f"{name}.onnx"
        sidecar[name]["sha256"] = hashlib.sha256(data).hexdigest()
    return sidecar, tmp_path


def fake_factory(calls):
    """Router picks fine_strand; coarse is solid; the editor follows the strokes."""

    class Session:
        def __init__(self, path, providers):
            self.graph = path.replace("\\", "/").split("/")[-1].removesuffix(".onnx")

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, names, feeds):
            calls.append((self.graph, tuple(names)))
            if self.graph == "router":
                logits = np.zeros((1, len(CATEGORIES)), np.float32)
                logits[0, 0] = 1
                return [logits] + [np.zeros((1, 2, 2, 2), np.float32)] * 4
            if self.graph == "coarse":
                out = {
                    "coarse_alpha": np.ones((1, 1, 8, 8), np.float32),
                    "edit_depth": np.full((1, 1, 16, 16), 0.5, np.float32),
                }
                return [out[n] for n in names]
            if self.graph == "stroke_edit":
                x = feeds["stroke_input"]
                assert x.shape == (1, 7, 16, 16)
                return [np.clip(x[:, 3:4] + x[:, 4:5] - x[:, 5:6], 0, 1)]
            raise AssertionError(self.graph)

    return Session


def test_remove_wins_where_both_mark():
    keep = np.zeros((4, 4), bool)
    remove = np.zeros((4, 4), bool)
    keep[:2] = True
    remove[1:3] = True
    s = combine_masks(keep, remove, (4, 4))
    assert (s[0] == KEEP).all() and (s[1] == REMOVE).all() and (s[2] == REMOVE).all()
    assert (s[3] == UNTOUCHED).all()


def test_mask_marks_light_opaque_pixels_only():
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[0, 0] = (255, 255, 255, 255)  # marked
    rgba[0, 1] = (255, 255, 255, 0)  # transparent: not marked
    rgba[0, 2] = (60, 60, 60, 255)  # dark: not marked
    marks = mask_from_image(Image.fromarray(rgba, "RGBA"), (3, 2))
    assert marks.tolist() == [[True, False, False], [False, False, False]]


def test_mask_size_mismatch_raises():
    with pytest.raises(EditInputError, match="does not match"):
        mask_from_image(Image.new("L", (5, 5)), (4, 4))


def test_keep_is_not_clamped():
    s = np.array([[KEEP, REMOVE, UNTOUCHED]], np.float32)
    assert net_clamp(s).tolist() == [[UNTOUCHED, REMOVE, UNTOUCHED]]


def test_stroke_input_rejects_unknown_spec():
    with pytest.raises(ValueError, match="Unsupported"):
        stroke_edit_input(
            np.zeros((4, 4, 3), np.float32),
            np.zeros((4, 4)),
            np.zeros((4, 4)),
            np.zeros((1, 1, 16, 16)),
            {**STROKE_SPEC, "resize": "nearest"},
        )


def test_resize_nearest_keeps_values():
    x = np.array([[0, 1], [0.5, 0]], np.float32)
    assert set(np.unique(resize_nearest(x, 7, 5))) <= {0.0, 0.5, 1.0}


def test_edit_removes_marked_object_and_leaves_far_pixels(edit_bundle):
    sidecar, root = edit_bundle
    calls = []
    pipe = RoutedPipeline(
        sidecar, lambda n: root / n, session_factory=fake_factory(calls)
    )
    assert pipe.supports_edit
    image = Image.new("RGB", (48, 32), (200, 100, 50))
    strokes = np.full((32, 48), UNTOUCHED, np.float32)
    strokes[:, :8] = REMOVE
    matte = np.ones((32, 48), np.float32)
    out, route = pipe.edit_alpha(image, strokes, matte)
    a = np.asarray(out)
    assert route["pipeline"] == "edit"
    assert (a[:, :8] == 0).all()
    assert (a[:, 40:] == 255).all()
    assert ("coarse", ("coarse_alpha", "edit_depth")) in calls
    assert any(g == "stroke_edit" for g, _ in calls)


def test_edit_without_matte_mattes_first(edit_bundle):
    sidecar, root = edit_bundle
    pipe = RoutedPipeline(sidecar, lambda n: root / n, session_factory=fake_factory([]))
    strokes = np.full((20, 20), UNTOUCHED, np.float32)
    strokes[5:8, 5:8] = REMOVE
    out, _ = pipe.edit_alpha(Image.new("RGB", (20, 20)), strokes)
    assert np.asarray(out)[6, 6] == 0


def test_edit_rejects_stroke_size_mismatch(edit_bundle):
    sidecar, root = edit_bundle
    pipe = RoutedPipeline(sidecar, lambda n: root / n, session_factory=fake_factory([]))
    with pytest.raises(ValueError, match="do not match"):
        pipe.edit_alpha(Image.new("RGB", (20, 20)), np.zeros((10, 10), np.float32))


def test_bundle_without_stroke_edit_cannot_edit(edit_bundle):
    sidecar, root = edit_bundle
    old = copy.deepcopy(sidecar)
    del old["stroke_edit"]
    pipe = RoutedPipeline(old, lambda n: root / n, session_factory=fake_factory([]))
    assert not pipe.supports_edit
    with pytest.raises(ValueError, match="no stroke-edit graph"):
        pipe.edit_alpha(Image.new("RGB", (4, 4)), np.zeros((4, 4), np.float32))


def test_stroke_edit_needs_edit_depth_output(edit_bundle):
    sidecar, _ = edit_bundle
    bad = copy.deepcopy(sidecar)
    del bad["coarse"]["edit_depth_name"]
    with pytest.raises(ValueError, match="edit_depth"):
        validate_sidecar(bad)


def _loaded_model(sidecar, root):
    from withoutbg.models import OpenWeightsModel

    model = OpenWeightsModel(model_path=root / "coarse.onnx")
    model.routed = RoutedPipeline(
        sidecar, lambda n: root / n, session_factory=fake_factory([])
    )
    model._models_loaded = True
    return model


def test_model_edit_returns_rgba_and_accepts_rgba_matte(edit_bundle):
    sidecar, root = edit_bundle
    model = _loaded_model(sidecar, root)
    image = Image.new("RGB", (24, 16), (10, 20, 30))
    previous = image.convert("RGBA")  # fully opaque previous result
    remove = np.zeros((16, 24), bool)
    remove[:, :4] = True
    out = model.edit(image, remove=remove, matte=previous)
    assert out.mode == "RGBA" and out.size == (24, 16)
    alpha = np.asarray(out.getchannel("A"))
    assert (alpha[:, :4] == 0).all() and (alpha[:, 20:] == 255).all()


def test_model_edit_needs_a_mask(edit_bundle):
    from withoutbg.exceptions import WithoutBGError

    sidecar, root = edit_bundle
    with pytest.raises(WithoutBGError, match="keep and/or remove"):
        _loaded_model(sidecar, root).edit(Image.new("RGB", (4, 4)))


def test_model_edit_reports_bad_mask_size(edit_bundle):
    from withoutbg.exceptions import WithoutBGError

    sidecar, root = edit_bundle
    with pytest.raises(WithoutBGError, match="does not match"):
        _loaded_model(sidecar, root).edit(
            Image.new("RGB", (8, 8)), keep=Image.new("L", (4, 4))
        )


def test_model_edit_on_old_bundle_explains_version(edit_bundle):
    from withoutbg.exceptions import WithoutBGError

    sidecar, root = edit_bundle
    old = copy.deepcopy(sidecar)
    del old["stroke_edit"]
    with pytest.raises(WithoutBGError, match="10.12.4"):
        _loaded_model(old, root).edit(
            Image.new("RGB", (4, 4)), keep=np.ones((4, 4), bool)
        )


def test_model_edit_accepts_float_and_bytes_masks(edit_bundle, tmp_path):
    import io

    sidecar, root = edit_bundle
    model = _loaded_model(sidecar, root)
    buf = io.BytesIO()
    Image.new("L", (8, 8), 0).save(buf, "PNG")
    out = model.edit(
        Image.new("RGB", (8, 8)),
        keep=np.zeros((8, 8), np.float32),
        remove=buf.getvalue(),
    )
    assert out.size == (8, 8)


def test_cli_edit_rejects_api_and_matte_alone(tmp_path):
    from click.testing import CliRunner

    from withoutbg.cli import main

    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    runner = CliRunner()
    r = runner.invoke(main, [str(img), "--matte", str(img)])
    assert r.exit_code == 1 and "--matte needs" in r.output
    r = runner.invoke(
        main, [str(img), "--remove", str(img), "--use-api", "--api-key", "k"]
    )
    assert r.exit_code == 1 and "open-weights" in r.output


def test_cli_edit_runs_model_edit(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from withoutbg import cli

    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    seen = {}

    def fake_edit(self, input_image, keep=None, remove=None, matte=None):
        seen.update(keep=keep, remove=remove)
        return Image.new("RGBA", (4, 4))

    monkeypatch.setattr(cli.WithoutBGOpenWeights, "edit", fake_edit)
    r = CliRunner().invoke(cli.main, [str(img), "--remove", str(img)])
    assert r.exit_code == 0, r.output
    assert seen["remove"] == img and (tmp_path / "a-withoutbg.png").exists()
