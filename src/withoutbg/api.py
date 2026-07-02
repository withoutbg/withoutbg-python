"""withoutBG API client for cloud-based background removal."""

import base64
import io
from pathlib import Path
from typing import Any, Callable, Optional, Union

import requests
from PIL import Image

from .exceptions import APIError
from .models import _apply_exif_orientation


class WithoutBGAPIClient:
    """Client for the withoutBG API."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: str = "https://api.withoutbg.com"
    ):
        """Initialize the withoutBG API client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoints
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def _encode_image(self, image: Union[str, Path, Image.Image, bytes]) -> str:
        """Encode image to base64 for API transmission."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        elif isinstance(image, bytes):
            image_bytes = image
        else:
            raise APIError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    def _decode_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        image_bytes = base64.b64decode(base64_string)
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.copy()

    def _resize_for_api(
        self, image: Image.Image, max_size: int = 1024
    ) -> tuple[Image.Image, tuple[int, int]]:
        """Resize image for API transmission while preserving aspect ratio.

        Args:
            image: Input PIL Image
            max_size: Maximum dimension size (default 1024)

        Returns:
            Tuple of (resized_image, original_size)
        """
        original_size = image.size
        width, height = original_size

        # If image is already smaller than max_size, return as-is
        if max(width, height) <= max_size:
            return image, original_size

        # Calculate new dimensions preserving aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image, original_size

    def _apply_alpha_channel(
        self, original_image: Image.Image, alpha_image: Image.Image
    ) -> Image.Image:
        """Apply alpha channel to original image to remove background.

        Args:
            original_image: Original RGB/RGBA image
            alpha_image: Grayscale alpha channel image

        Returns:
            RGBA image with background removed
        """
        # Ensure original image is in RGB mode
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")

        # Ensure alpha image is grayscale
        if alpha_image.mode != "L":
            alpha_image = alpha_image.convert("L")

        # Resize alpha to match original image if needed
        if alpha_image.size != original_image.size:
            alpha_image = alpha_image.resize(
                original_image.size, Image.Resampling.LANCZOS
            )

        # Create RGBA image by adding alpha channel
        rgba_image = original_image.copy()
        rgba_image.putalpha(alpha_image)

        return rgba_image

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background using the withoutBG API.

        Args:
            input_image: Input image
            progress_callback: Optional callback for progress updates
            **kwargs: Additional API parameters

        Returns:
            PIL Image with background removed

        Raises:
            APIError: If API request fails
        """
        if not self.api_key:
            raise APIError("API key required for withoutBG API service")

        try:
            if progress_callback:
                progress_callback(0.1)

            # Store original image for local alpha application
            if isinstance(input_image, (str, Path)):
                with Image.open(input_image) as img:
                    original_image = img.copy()
            elif isinstance(input_image, Image.Image):
                original_image = input_image.copy()
            elif isinstance(input_image, bytes):
                with Image.open(io.BytesIO(input_image)) as img:
                    original_image = img.copy()
            else:
                raise APIError(f"Unsupported image type: {type(input_image)}")

            # Apply EXIF orientation correction right after loading
            original_image = _apply_exif_orientation(original_image)

            # Resize image for API transmission to optimize latency
            if progress_callback:
                progress_callback(0.2)
            api_image, original_size = self._resize_for_api(original_image)

            # Encode resized image
            if progress_callback:
                progress_callback(0.3)
            encoded_image = self._encode_image(api_image)

            # Prepare request for base64 endpoint
            payload = {"image_base64": encoded_image}

            # Make API request to base64 endpoint
            if progress_callback:
                progress_callback(0.5)
            response = self.session.post(
                f"{self.base_url}/v1.0/alpha-channel-base64", json=payload, timeout=30
            )

            if response.status_code == 401:
                raise APIError("Invalid API key")
            elif response.status_code == 429:
                raise APIError("Rate limit exceeded (20 requests per minute)")
            elif response.status_code == 402:
                raise APIError(
                    "Insufficient credits. Get more credits at https://withoutbg.com/login"
                )
            elif response.status_code == 403:
                raise APIError(
                    "Credits expired. Top up your account to reactivate "
                    "frozen credits at https://withoutbg.com/login"
                )
            elif not response.ok:
                try:
                    error_msg = response.json().get("error", "Unknown API error")
                except Exception:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                raise APIError(f"API request failed: {error_msg}")

            # Decode response
            if progress_callback:
                progress_callback(0.7)
            result_data = response.json()

            if "alpha_base64" not in result_data:
                raise APIError(
                    f"Invalid API response format\n\n"
                    f"More info: sample response: {result_data}"
                )

            # Decode alpha channel
            if progress_callback:
                progress_callback(0.8)
            alpha_image = self._decode_image(result_data["alpha_base64"])

            # Resize alpha channel back to original dimensions if needed
            if alpha_image.size != original_size:
                alpha_image = alpha_image.resize(
                    original_size, Image.Resampling.LANCZOS
                )

            # Apply alpha channel to original image locally
            if progress_callback:
                progress_callback(0.9)
            result = self._apply_alpha_channel(original_image, alpha_image)

            if progress_callback:
                progress_callback(1.0)

            return result

        except requests.RequestException as e:
            raise APIError(f"Network error: {str(e)}") from e
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"Unexpected error: {str(e)}") from e

    def get_usage(self) -> dict[str, Any]:
        """Get current API usage statistics.

        Returns:
            Dictionary with usage information
        """
        if not self.api_key:
            raise APIError("API key required")

        try:
            response = self.session.get(f"{self.base_url}/available-credit")
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        except requests.RequestException as e:
            raise APIError(f"Failed to get usage: {str(e)}") from e

    def get_models(self) -> dict[str, Any]:
        """Get available models and their capabilities.

        Returns:
            Dictionary with model information
        """
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        except requests.RequestException as e:
            raise APIError(f"Failed to get models: {str(e)}") from e


# Deprecated alias — will be removed in a future release
ProAPI = WithoutBGAPIClient
