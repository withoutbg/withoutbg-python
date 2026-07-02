"""Integration tests for the WithoutBGAPIClient class."""

import base64
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.withoutbg.api import WithoutBGAPIClient
from src.withoutbg.exceptions import APIError


class TestWithoutBGAPIClient:
    """Test cases for WithoutBGAPIClient integration."""

    @pytest.fixture
    def api_client(self):
        """Create WithoutBGAPIClient with test API key."""
        return WithoutBGAPIClient(api_key="test_api_key")

    @pytest.fixture
    def test_image_path(self):
        """Path to test ice cream image."""
        return Path(__file__).parent / "fixtures" / "images" / "test-ice-cream.png"

    @pytest.fixture
    def mock_alpha_path(self):
        """Path to mock alpha channel image."""
        return (
            Path(__file__).parent
            / "fixtures"
            / "images"
            / "test-ice-cream-api-service-mock-alpha.png"
        )

    @pytest.fixture
    def test_image(self, test_image_path):
        """Load test image."""
        img = Image.open(test_image_path)
        yield img
        img.close()

    @pytest.fixture
    def mock_alpha_image(self, mock_alpha_path):
        """Load mock alpha channel image."""
        img = Image.open(mock_alpha_path)
        yield img
        img.close()

    def _create_mock_alpha_response(self, mock_alpha_image):
        """Create mock API response using real alpha channel image."""
        # Convert alpha image to base64
        buffer = io.BytesIO()
        mock_alpha_image.save(buffer, format="PNG")
        alpha_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"alpha_base64": alpha_base64}

    @patch("src.withoutbg.api.requests.Session.post")
    def test_remove_background_success(
        self, mock_post, api_client, test_image, mock_alpha_image
    ):
        """Test successful background removal with real alpha channel."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = self._create_mock_alpha_response(
            mock_alpha_image
        )
        mock_post.return_value = mock_response

        # Test background removal
        result = api_client.remove_background(test_image)

        # Verify result
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == test_image.size

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0].endswith("/v1.0/alpha-channel-base64")
        assert "image_base64" in call_args[1]["json"]

    @patch("src.withoutbg.api.requests.Session.post")
    def test_remove_background_with_file_path(
        self, mock_post, api_client, test_image_path, mock_alpha_image
    ):
        """Test background removal using file path."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = self._create_mock_alpha_response(
            mock_alpha_image
        )
        mock_post.return_value = mock_response

        # Test with file path
        result = api_client.remove_background(test_image_path)

        # Verify result
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    @patch("src.withoutbg.api.requests.Session.post")
    def test_remove_background_with_bytes(
        self, mock_post, api_client, test_image_path, mock_alpha_image
    ):
        """Test background removal using bytes input."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = self._create_mock_alpha_response(
            mock_alpha_image
        )
        mock_post.return_value = mock_response

        # Load image as bytes
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()

        # Test with bytes
        result = api_client.remove_background(image_bytes)

        # Verify result
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    @patch("src.withoutbg.api.requests.Session.post")
    def test_image_resizing_for_large_images(
        self, mock_post, api_client, mock_alpha_image
    ):
        """Test that large images are resized for API transmission."""
        # Create large test image
        large_image = Image.new("RGB", (2048, 1536), color=(255, 0, 0))

        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = self._create_mock_alpha_response(
            mock_alpha_image
        )
        mock_post.return_value = mock_response

        # Test background removal
        result = api_client.remove_background(large_image)

        # Verify result maintains original size
        assert result.size == large_image.size
        assert result.mode == "RGBA"

        # Verify API was called (image should be resized internally)
        mock_post.assert_called_once()

    @patch("src.withoutbg.api.requests.Session.post")
    def test_alpha_channel_application(
        self, mock_post, api_client, test_image, mock_alpha_image
    ):
        """Test that alpha channel is correctly applied to original image."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = self._create_mock_alpha_response(
            mock_alpha_image
        )
        mock_post.return_value = mock_response

        # Test background removal
        result = api_client.remove_background(test_image)

        # Verify alpha channel was applied
        assert result.mode == "RGBA"

        # Check that result has alpha channel data
        alpha_channel = result.split()[-1]  # Get alpha channel
        alpha_array = list(alpha_channel.getdata())

        # Should have varying alpha values (not all 255 or all 0)
        unique_alpha_values = set(alpha_array)
        assert len(unique_alpha_values) > 1, "Alpha channel should have varying values"

    @patch("src.withoutbg.api.requests.Session.post")
    def test_api_error_401_unauthorized(self, mock_post, api_client, test_image):
        """Test 401 unauthorized error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.ok = False
        mock_post.return_value = mock_response

        with pytest.raises(APIError, match="Invalid API key"):
            api_client.remove_background(test_image)

    @patch("src.withoutbg.api.requests.Session.post")
    def test_api_error_402_insufficient_credits(
        self, mock_post, api_client, test_image
    ):
        """Test 402 insufficient credits error handling."""
        mock_response = Mock()
        mock_response.status_code = 402
        mock_response.ok = False
        mock_post.return_value = mock_response

        with pytest.raises(APIError, match="Insufficient credits"):
            api_client.remove_background(test_image)

    @patch("src.withoutbg.api.requests.Session.post")
    def test_api_error_403_expired_credits(self, mock_post, api_client, test_image):
        """Test 403 expired credits error handling."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.ok = False
        mock_post.return_value = mock_response

        with pytest.raises(APIError, match="Credits expired"):
            api_client.remove_background(test_image)

    @patch("src.withoutbg.api.requests.Session.post")
    def test_api_error_429_rate_limit(self, mock_post, api_client, test_image):
        """Test 429 rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_post.return_value = mock_response

        with pytest.raises(APIError, match="Rate limit exceeded"):
            api_client.remove_background(test_image)

    @patch("src.withoutbg.api.requests.Session.post")
    def test_api_error_500_server_error(self, mock_post, api_client, test_image):
        """Test 500 server error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.ok = False
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_post.return_value = mock_response

        with pytest.raises(APIError, match="API request failed"):
            api_client.remove_background(test_image)

    @patch("src.withoutbg.api.requests.Session.post")
    def test_invalid_response_format(self, mock_post, api_client, test_image):
        """Test handling of invalid API response format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"status": "success"}  # Missing alpha_base64
        mock_post.return_value = mock_response

        with pytest.raises(APIError, match="Invalid API response format"):
            api_client.remove_background(test_image)

    @patch("src.withoutbg.api.requests.Session.post")
    def test_malformed_json_response(self, mock_post, api_client, test_image):
        """Test handling of malformed JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "not valid json"
        mock_post.return_value = mock_response

        with pytest.raises(APIError, match="Unexpected error"):
            api_client.remove_background(test_image)

    def test_no_api_key_error(self):
        """Test error when no API key is provided."""
        api_client = WithoutBGAPIClient()  # No API key
        test_image = Image.new("RGB", (100, 100))

        with pytest.raises(APIError, match="API key required"):
            api_client.remove_background(test_image)

    @patch("src.withoutbg.api.requests.Session.get")
    def test_get_usage_success(self, mock_get, api_client):
        """Test successful usage information retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "credits_used": 150,
            "credits_total": 1000,
            "credits_remaining": 850,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.get_usage()

        assert result["credits_used"] == 150
        assert result["credits_remaining"] == 850
        mock_get.assert_called_once_with("https://api.withoutbg.com/available-credit")

    @patch("src.withoutbg.api.requests.Session.get")
    def test_get_models_success(self, mock_get, api_client):
        """Test successful models information retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": ["open_source", "pro", "premium"],
            "default": "pro",
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.get_models()

        assert "models" in result
        assert result["default"] == "pro"
        mock_get.assert_called_once_with("https://api.withoutbg.com/v1/models")

    def test_get_usage_no_api_key(self):
        """Test usage retrieval without API key."""
        api_client = WithoutBGAPIClient()  # No API key

        with pytest.raises(APIError, match="API key required"):
            api_client.get_usage()

    @patch("src.withoutbg.api.requests.Session.post")
    def test_network_error_handling(self, mock_post, api_client, test_image):
        """Test network error handling."""
        import requests

        mock_post.side_effect = requests.ConnectionError("Connection failed")

        with pytest.raises(APIError, match="Network error"):
            api_client.remove_background(test_image)

    def test_unsupported_image_type(self, api_client):
        """Test error handling for unsupported image types."""
        with pytest.raises(APIError, match="Unsupported image type"):
            api_client.remove_background({"invalid": "type"})

    @patch("src.withoutbg.api.requests.Session.post")
    def test_custom_base_url(self, mock_post, mock_alpha_image):
        """Test API client with custom base URL."""
        custom_api = WithoutBGAPIClient(
            api_key="test_key", base_url="https://custom.api.com/"
        )

        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = self._create_mock_alpha_response(
            mock_alpha_image
        )
        mock_post.return_value = mock_response

        test_image = Image.new("RGB", (100, 100))
        result = custom_api.remove_background(test_image)

        # Verify custom URL was used
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://custom.api.com/v1.0/alpha-channel-base64"
        assert isinstance(result, Image.Image)
