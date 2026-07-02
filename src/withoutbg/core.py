"""Core background removal functionality with class-based API."""

import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Union

from PIL import Image

from .api import WithoutBGAPIClient
from .exceptions import ConfigurationError, WithoutBGError
from .models import OpenWeightsModel


class WithoutBG:
    """Base class for background removal.

    Use factory methods to create instances:
    - WithoutBG.open_weights() for local withoutBG Open Weights Model
    - WithoutBG.api(api_key) for withoutBG API
    """

    @staticmethod
    def open_weights(
        model_path: Optional[Union[str, Path]] = None,
    ) -> "WithoutBGOpenWeights":
        """Create instance using the local withoutBG Open Weights Model.

        Args:
            model_path: Optional path to withoutbg-open-weights.onnx

        Returns:
            WithoutBGOpenWeights: Instance for local background removal

        Example:
            >>> model = WithoutBG.open_weights()
            >>> result = model.remove_background("input.jpg")
        """
        return WithoutBGOpenWeights(model_path=model_path)

    @staticmethod
    def opensource(
        model_path: Optional[Union[str, Path]] = None,
    ) -> "WithoutBGOpenWeights":
        """Create instance using the local withoutBG Open Weights Model.

        .. deprecated::
            Use :meth:`open_weights` instead. This alias will be removed in a
            future release.
        """
        warnings.warn(
            "WithoutBG.opensource() is deprecated. "
            "Use WithoutBG.open_weights() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return WithoutBGOpenWeights(model_path=model_path)

    @staticmethod
    def api(
        api_key: Optional[str] = None, base_url: str = "https://api.withoutbg.com"
    ) -> "WithoutBGAPI":
        """Create instance using the withoutBG API.

        Args:
            api_key: API key for the withoutBG API service. If omitted, reads
                WITHOUTBG_API_KEY from the environment.
            base_url: Base URL for API endpoints (optional)

        Returns:
            WithoutBGAPI: Instance for cloud-based background removal

        Raises:
            ConfigurationError: If no API key is provided or found in the
                environment.

        Example:
            >>> model = WithoutBG.api(api_key="sk_...")
            >>> result = model.remove_background("input.jpg")
        """
        resolved_key = api_key or os.getenv("WITHOUTBG_API_KEY")
        if not resolved_key:
            raise ConfigurationError(
                "API key required. Pass api_key or set WITHOUTBG_API_KEY "
                "environment variable."
            )
        return WithoutBGAPI(resolved_key, base_url=base_url)

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background from a single image.

        Args:
            input_image: Input image as file path, PIL Image, or bytes
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments passed to the model/API

        Returns:
            PIL Image with background removed
        """
        raise NotImplementedError("Subclass must implement remove_background()")

    def remove_background_batch(
        self,
        input_images: list[Union[str, Path, Image.Image, bytes]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Remove background from multiple images.

        Args:
            input_images: List of input images
            output_dir: Directory to save results (optional)
            **kwargs: Additional arguments

        Returns:
            List of PIL Images with backgrounds removed
        """
        raise NotImplementedError("Subclass must implement remove_background_batch()")


class WithoutBGOpenWeights(WithoutBG):
    """withoutBG Open Weights Model implementation.

    Uses ONNX-based models running locally for background removal.
    Models are downloaded and loaded on first use, then reused for all inferences.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize with the withoutBG Open Weights Model.

        Args:
            model_path: Path to withoutbg-open-weights.onnx (optional)

        Note:
            The model is loaded lazily on the first call to remove_background()
            or preload(). If a path is not provided, the ONNX graph and sidecar
            metadata are downloaded from Hugging Face on first use.
        """
        self.model = OpenWeightsModel(model_path=model_path)

    def preload(self) -> None:
        """Download (if needed) and load all ONNX models into memory."""
        self.model.preload()

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background from image using the withoutBG Open Weights Model.

        Args:
            input_image: Input image as file path, PIL Image, or bytes
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments (unused for withoutBG Open Weights Model)

        Returns:
            PIL Image with background removed

        Example:
            >>> model = WithoutBG.open_weights()
            >>> result = model.remove_background("input.jpg")
            >>> result.save("output.png")
        """
        try:
            return self.model.remove_background(
                input_image, progress_callback=progress_callback, **kwargs
            )
        except WithoutBGError:
            raise
        except Exception as e:
            raise WithoutBGError(f"Background removal failed: {str(e)}") from e

    def remove_background_batch(
        self,
        input_images: list[Union[str, Path, Image.Image, bytes]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Remove background from multiple images using the withoutBG Open
        Weights Model.

        The model is loaded once and reused for all images, making this
        much more efficient than processing images separately.

        Args:
            input_images: List of input images
            output_dir: Directory to save results (optional)
            **kwargs: Additional arguments

        Returns:
            List of PIL Images with backgrounds removed

        Example:
            >>> model = WithoutBG.open_weights()
            >>> results = model.remove_background_batch(["img1.jpg", "img2.jpg"])
        """
        results = []

        for i, input_image in enumerate(input_images):
            output_path = None
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)

                # Try to get original filename
                if isinstance(input_image, (str, Path)):
                    input_path = Path(input_image)
                    stem = input_path.stem
                    suffix = input_path.suffix or ".png"
                    output_filename = f"{stem}-withoutbg{suffix}"
                else:
                    # For PIL Images or bytes, use numbered fallback
                    output_filename = f"output_{i:04d}-withoutbg.png"

                output_path = output_dir_path / output_filename

            # Process image (reusing self.model for efficiency)
            result = self.remove_background(input_image, **kwargs)

            if output_path:
                result.save(output_path)
                # Note: Keep result in memory for return, don't close it yet

            results.append(result)

        return results


class WithoutBGAPI(WithoutBG):
    """withoutBG API implementation.

    Uses cloud-based withoutBG API for high-quality background removal.
    API client is initialized once and reused for all requests.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.withoutbg.com"):
        """Initialize withoutBG API client.

        Args:
            api_key: API key for the withoutBG API service
            base_url: Base URL for API endpoints (optional)

        Note:
            API client is initialized once and reused for all requests.
        """
        self.api_client = WithoutBGAPIClient(api_key=api_key, base_url=base_url)

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background using the withoutBG API.

        Args:
            input_image: Input image as file path, PIL Image, or bytes
            progress_callback: Optional callback for progress updates
            **kwargs: Additional API parameters

        Returns:
            PIL Image with background removed

        Example:
            >>> model = WithoutBG.api(api_key="sk_...")
            >>> result = model.remove_background("input.jpg")
            >>> result.save("output.png")
        """
        try:
            return self.api_client.remove_background(
                input_image, progress_callback=progress_callback, **kwargs
            )
        except WithoutBGError:
            raise
        except Exception as e:
            raise WithoutBGError(f"Background removal failed: {str(e)}") from e

    def remove_background_batch(
        self,
        input_images: list[Union[str, Path, Image.Image, bytes]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Remove background from multiple images using the withoutBG API.

        The API client is reused for all images. Automatically adds a 3-second
        delay between requests to respect the 20 requests/minute rate limit.

        Args:
            input_images: List of input images
            output_dir: Directory to save results (optional)
            **kwargs: Additional arguments

        Returns:
            List of PIL Images with backgrounds removed

        Example:
            >>> model = WithoutBG.api(api_key="sk_...")
            >>> results = model.remove_background_batch(["img1.jpg", "img2.jpg"])
        """
        results = []

        for i, input_image in enumerate(input_images):
            output_path = None
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)

                # Try to get original filename
                if isinstance(input_image, (str, Path)):
                    input_path = Path(input_image)
                    stem = input_path.stem
                    suffix = input_path.suffix or ".png"
                    output_filename = f"{stem}-withoutbg{suffix}"
                else:
                    # For PIL Images or bytes, use numbered fallback
                    output_filename = f"output_{i:04d}-withoutbg.png"

                output_path = output_dir_path / output_filename

            # Process image (reusing self.api_client for efficiency)
            result = self.remove_background(input_image, **kwargs)

            if output_path:
                result.save(output_path)
                # Note: Keep result in memory for return, don't close it yet

            results.append(result)

            # Rate limit: 20 requests/minute = 3s per request
            # Skip delay after the last image
            if i < len(input_images) - 1:
                time.sleep(3)

        return results


# Deprecated alias — will be removed in a future release
WithoutBGOpenSource = WithoutBGOpenWeights
