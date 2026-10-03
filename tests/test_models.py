"""Unit tests for OpenWeightsModel with mocked ONNX session."""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from withoutbg.exceptions import ModelNotFoundError, WithoutBGError
from withoutbg.models import OpenWeightsModel

FAKE_MODEL_PATH = "/fake/withoutbg-open-weights.onnx"
DEFAULT_SIDECAR = {
    "canvas_size": 448,
    "output_shape": [1, 1, 448, 448],
    "input_name": "rgb",
    "model_version": "10.0.0",
}


def _make_mock_session(output: np.ndarray, input_name: str = "rgb") -> Mock:
    session = Mock()
    session.get_inputs.return_value = [Mock(name=input_name)]
    session.run.return_value = [output]
    return session


def _make_model_sidecar(model: OpenWeightsModel) -> None:
    model.sidecar = DEFAULT_SIDECAR.copy()
    model._models_loaded = True
    model.model_path = __import__("pathlib").Path(FAKE_MODEL_PATH)


@pytest.fixture
def sample_rgb_image():
    return Image.new("RGB", (256, 192), color=(120, 80, 200))


@pytest.fixture
def mock_model_path():
    return FAKE_MODEL_PATH


class TestLazyModelLoading:
    """Tests for deferred ONNX model loading."""

    @patch("withoutbg.models.ort.InferenceSession")
    @patch("withoutbg.models.OpenWeightsModel._load_sidecar")
    def test_models_not_loaded_on_init(
        self, mock_load_sidecar, mock_session, mock_model_path
    ):
        model = OpenWeightsModel(model_path=mock_model_path)

        assert model.models_loaded is False
        mock_session.assert_not_called()

    @patch("withoutbg.models.ort.InferenceSession")
    @patch("withoutbg.models.OpenWeightsModel._load_sidecar")
    def test_preload_loads_session(
        self, mock_load_sidecar, mock_session, mock_model_path
    ):
        mock_session.return_value = _make_mock_session(
            np.full((1, 1, 448, 448), 0.8, dtype=np.float32)
        )
        model = OpenWeightsModel(model_path=mock_model_path)

        model.preload()

        assert model.models_loaded is True
        mock_session.assert_called_once()

    @patch("withoutbg.models.ort.InferenceSession")
    def test_remove_background_triggers_lazy_load(
        self,
        mock_session,
        tmp_path,
        sample_rgb_image,
    ):
        model_path = tmp_path / "withoutbg-open-weights.onnx"
        model_path.touch()
        sidecar_path = model_path.with_suffix(model_path.suffix + ".json")
        sidecar_path.write_text(json.dumps(DEFAULT_SIDECAR))

        mock_session.return_value = _make_mock_session(
            np.full((1, 1, 448, 448), 0.8, dtype=np.float32)
        )
        model = OpenWeightsModel(model_path=model_path)

        result = model.remove_background(sample_rgb_image)

        assert model.models_loaded is True
        assert mock_session.call_count == 1
        assert result.mode == "RGBA"
        assert result.size == sample_rgb_image.size


class TestInferencePipeline:
    """Tests for letterbox preprocessing and alpha estimation."""

    @patch("withoutbg.models.ort.InferenceSession")
    def test_letterbox_image_produces_expected_tensor_shape(
        self, mock_session, mock_model_path, sample_rgb_image
    ):
        model = OpenWeightsModel(model_path=mock_model_path)
        _make_model_sidecar(model)

        rgb, new_w, new_h = model._letterbox_image(sample_rgb_image)

        assert rgb.shape == (1, 3, 448, 448)
        assert rgb.dtype == np.float32
        assert 0.0 <= rgb.min() <= rgb.max() <= 1.0
        assert new_w > 0 and new_h > 0

    @patch("withoutbg.models.ort.InferenceSession")
    def test_postprocess_alpha_returns_original_size(
        self, mock_session, mock_model_path, sample_rgb_image
    ):
        model = OpenWeightsModel(model_path=mock_model_path)
        _make_model_sidecar(model)
        alpha_canvas = np.full((448, 448), 0.75, dtype=np.float32)

        # 256x192 letterboxed onto 448 → new_w=448, new_h=336
        alpha = model._postprocess_alpha(
            alpha_canvas, new_w=448, new_h=336, orig_w=256, orig_h=192
        )

        assert alpha.mode == "L"
        assert alpha.size == (256, 192)

    @patch("withoutbg.models.ort.InferenceSession")
    def test_estimate_alpha_runs_inference(
        self, mock_session, mock_model_path, sample_rgb_image
    ):
        mock_session.return_value = _make_mock_session(
            np.full((1, 1, 448, 448), 0.8, dtype=np.float32)
        )
        model = OpenWeightsModel(model_path=mock_model_path)
        _make_model_sidecar(model)
        model.session = mock_session.return_value
        progress_values: list[float] = []

        alpha = model.estimate_alpha(
            sample_rgb_image, progress_callback=progress_values.append
        )

        assert alpha.mode == "L"
        assert alpha.size == sample_rgb_image.size
        assert progress_values[0] == 0.0
        assert progress_values[-1] == 1.0
        mock_session.return_value.run.assert_called_once()

    @patch("withoutbg.models.ort.InferenceSession")
    @patch("withoutbg.models.OpenWeightsModel._load_sidecar")
    def test_model_not_found_error_propagates_from_inference(
        self, mock_load_sidecar, mock_session, mock_model_path, sample_rgb_image
    ):
        mock_session.side_effect = ModelNotFoundError("Failed to load model")
        model = OpenWeightsModel(model_path=mock_model_path)

        with pytest.raises(ModelNotFoundError, match="Failed to load model"):
            model.remove_background(sample_rgb_image)

    @patch("withoutbg.models.ort.InferenceSession")
    def test_unsupported_input_type_raises_withoutbg_error(
        self, mock_session, mock_model_path
    ):
        model = OpenWeightsModel(model_path=mock_model_path)

        with pytest.raises(WithoutBGError, match="Unsupported input type"):
            model.remove_background({"not": "an image"})

    @patch("withoutbg.models.OpenWeightsModel._download_from_hf")
    def test_load_sidecar_reads_local_json(self, mock_download, tmp_path):
        model_path = tmp_path / "withoutbg-open-weights.onnx"
        model_path.touch()
        sidecar_path = model_path.with_suffix(model_path.suffix + ".json")
        sidecar_path.write_text(json.dumps(DEFAULT_SIDECAR))

        model = OpenWeightsModel(model_path=model_path)
        model.model_path = model_path
        model._load_sidecar()

        assert model.sidecar["canvas_size"] == 448
        mock_download.assert_not_called()


class TestRoutedBundle:
    """A sidecar with a ``pipeline`` selects the routed matting / BiRefNet bundle."""

    @staticmethod
    def _write_bundle(tmp_path, with_birefnet=True):
        import hashlib

        sidecar = {
            "schema_version": 3,
            "pipeline": "routed",
            "max_inference_size": [4096, 4096],
            "router": {
                "file": "withoutbg-open-weights-backbone.onnx",
                "categories": ["fine_strand", "vehicle"],
                "birefnet_categories": ["vehicle"],
            },
            "coarse": {"file": "withoutbg-open-weights.onnx"},
            "birefnet": {"file": "birefnet-general.onnx"},
        }
        for name in ("router", "coarse", "birefnet"):
            data = name.encode()
            if name != "birefnet" or with_birefnet:
                (tmp_path / sidecar[name]["file"]).write_bytes(data)
            sidecar[name]["sha256"] = hashlib.sha256(data).hexdigest()
        model_path = tmp_path / "withoutbg-open-weights.onnx"
        (tmp_path / "withoutbg-open-weights.onnx.json").write_text(json.dumps(sidecar))
        return model_path

    @patch("withoutbg.models.ort.InferenceSession")
    def test_routed_sidecar_builds_routed_pipeline(self, mock_session, tmp_path):
        model = OpenWeightsModel(model_path=self._write_bundle(tmp_path))
        model._ensure_models_loaded()

        assert model.routed is not None
        assert model.session is None
        mock_session.assert_not_called()  # graphs load lazily

    def test_missing_local_asset_raises_model_not_found(self, tmp_path):
        model = OpenWeightsModel(
            model_path=self._write_bundle(tmp_path, with_birefnet=False)
        )
        model._ensure_models_loaded()

        with pytest.raises(ModelNotFoundError, match="birefnet-general.onnx"):
            model.routed.resolve_file("birefnet-general.onnx")

    def test_invalid_routed_sidecar_raises_withoutbg_error(self, tmp_path):
        model_path = self._write_bundle(tmp_path)
        sidecar_path = tmp_path / "withoutbg-open-weights.onnx.json"
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["pipeline"] = "routed_edge_refine"
        sidecar_path.write_text(json.dumps(sidecar))

        with pytest.raises(WithoutBGError, match="Enterprise"):
            OpenWeightsModel(model_path=model_path)._ensure_models_loaded()

    @patch("withoutbg.models.hf_hub_download")
    def test_hub_downloads_are_pinned(self, mock_download, tmp_path, monkeypatch):
        from withoutbg.models import HF_REVISION

        monkeypatch.delenv("WITHOUTBG_MODEL_PATH", raising=False)
        mock_download.return_value = str(tmp_path / "x.onnx")
        OpenWeightsModel()._download_from_hf("x.onnx", "x")

        assert mock_download.call_args.kwargs["revision"] == HF_REVISION
