"""Unit tests for WithoutBG core API."""

import warnings
from unittest.mock import patch

import pytest
from PIL import Image

from withoutbg import WithoutBG
from withoutbg.exceptions import (
    APIError,
    ConfigurationError,
    ModelNotFoundError,
    WithoutBGError,
)


class TestWithoutBGAPIFactory:
    """Tests for WithoutBG.api() factory method."""

    def test_api_uses_explicit_key(self):
        model = WithoutBG.api(api_key="sk_explicit")
        assert model.api_client.api_key == "sk_explicit"

    def test_api_uses_environment_variable(self, monkeypatch):
        monkeypatch.setenv("WITHOUTBG_API_KEY", "sk_from_env")
        model = WithoutBG.api()
        assert model.api_client.api_key == "sk_from_env"

    def test_explicit_key_overrides_environment(self, monkeypatch):
        monkeypatch.setenv("WITHOUTBG_API_KEY", "sk_from_env")
        model = WithoutBG.api(api_key="sk_explicit")
        assert model.api_client.api_key == "sk_explicit"

    def test_api_raises_configuration_error_without_key(self, monkeypatch):
        monkeypatch.delenv("WITHOUTBG_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="WITHOUTBG_API_KEY"):
            WithoutBG.api()


class TestOpenWeightsModelFactory:
    """Tests for WithoutBG.open_weights() factory method."""

    def test_open_weights_returns_open_weights_instance(self):
        from withoutbg.core import WithoutBGOpenWeights

        model = WithoutBG.open_weights()
        assert isinstance(model, WithoutBGOpenWeights)

    def test_opensource_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WithoutBG.opensource()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "open_weights" in str(w[0].message)

    def test_opensource_still_returns_open_weights_instance(self):
        from withoutbg.core import WithoutBGOpenWeights

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = WithoutBG.opensource()
        assert isinstance(model, WithoutBGOpenWeights)


class TestExceptionReraising:
    """Tests that typed exceptions propagate through WithoutBG wrappers."""

    @patch("withoutbg.core.WithoutBGAPIClient.remove_background")
    def test_api_error_is_not_wrapped(self, mock_remove):
        mock_remove.side_effect = APIError("Invalid API key")
        model = WithoutBG.api(api_key="sk_test")

        with pytest.raises(APIError, match="Invalid API key"):
            model.remove_background(Image.new("RGB", (64, 64)))

    @patch("withoutbg.models.OpenWeightsModel.remove_background")
    def test_model_not_found_error_is_not_wrapped(self, mock_remove):
        mock_remove.side_effect = ModelNotFoundError("Models missing")
        model = WithoutBG.open_weights()

        with pytest.raises(ModelNotFoundError, match="Models missing"):
            model.remove_background(Image.new("RGB", (64, 64)))

    @patch("withoutbg.core.WithoutBGAPIClient.remove_background")
    def test_unknown_errors_are_wrapped(self, mock_remove):
        mock_remove.side_effect = ValueError("unexpected")
        model = WithoutBG.api(api_key="sk_test")

        with pytest.raises(
            WithoutBGError, match="Background removal failed"
        ) as exc_info:
            model.remove_background(Image.new("RGB", (64, 64)))

        assert isinstance(exc_info.value.__cause__, ValueError)

    @patch("withoutbg.models.OpenWeightsModel.remove_background")
    def test_unknown_open_weights_errors_are_wrapped(self, mock_remove):
        mock_remove.side_effect = RuntimeError("unexpected")
        model = WithoutBG.open_weights()

        with pytest.raises(
            WithoutBGError, match="Background removal failed"
        ) as exc_info:
            model.remove_background(Image.new("RGB", (64, 64)))

        assert isinstance(exc_info.value.__cause__, RuntimeError)
