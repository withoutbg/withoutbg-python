"""Local model implementations."""

import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import onnxruntime as ort  # type: ignore
from huggingface_hub import hf_hub_download
from PIL import ExifTags, Image

from .exceptions import ModelNotFoundError, WithoutBGError
from .routed import RoutedPipeline

HF_REPO_ID = "withoutbg/withoutbg-openweights-onnx"
HF_MODEL_FILENAME = "withoutbg-open-weights.onnx"
HF_SIDECAR_FILENAME = "withoutbg-open-weights.onnx.json"
# Pinned Hub commit: a new upload never changes what a released SDK runs.
HF_REVISION = "93afc91c44e0e27706159ead1623918b49f31d15"


def _apply_exif_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation to rotate image correctly.

    Args:
        image: PIL Image that may contain EXIF orientation data

    Returns:
        PIL Image rotated according to EXIF orientation, or original if
        no orientation data
    """
    try:
        # Get EXIF data
        exif = image.getexif()
        if not exif:
            return image

        # Find orientation tag
        orientation_key = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orientation_key = tag
                break

        if orientation_key is None or orientation_key not in exif:
            return image

        orientation = exif[orientation_key]

        # Apply rotation based on orientation value
        # EXIF orientation values:
        # 1 = Normal (no rotation)
        # 2 = Mirrored horizontally
        # 3 = Rotated 180°
        # 4 = Mirrored vertically
        # 5 = Mirrored horizontally and rotated 90° CCW
        # 6 = Rotated 90° CW
        # 7 = Mirrored horizontally and rotated 90° CW
        # 8 = Rotated 90° CCW

        if orientation == 2:
            # Horizontal mirror
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # 180° rotation
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            # Vertical mirror
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            # Horizontal mirror + 90° CCW
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = image.rotate(90, expand=True)
        elif orientation == 6:
            # 90° CW rotation
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            # Horizontal mirror + 90° CW
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = image.rotate(-90, expand=True)
        elif orientation == 8:
            # 90° CCW rotation
            image = image.rotate(90, expand=True)

        return image

    except Exception:
        # If any error occurs during EXIF processing, return original image
        return image


class OpenWeightsModel:
    """Local ONNX-based background removal model (withoutBG Open Weights Model)."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize the withoutBG Open Weights Model.

        Args:
            model_path: Path to withoutbg-open-weights.onnx (optional)

        Note:
            Model paths can be provided explicitly. Otherwise the bundle is
            downloaded from Hugging Face (pinned to ``HF_REVISION``) on first
            use. The sidecar JSON next to the model decides the pipeline: a
            ``routed`` bundle runs the router and then the matting or BiRefNet
            graph from the same folder; a legacy sidecar runs the single
            letterboxed graph. Models load lazily on first use via
            remove_background() or preload().
        """
        self._model_path_override = model_path
        self.model_path: Optional[Path] = None
        self.sidecar: dict[str, Any] = {}
        self.session: Optional[ort.InferenceSession] = None
        self._models_loaded = False
        self.routed: Optional[RoutedPipeline] = None
        self._hf_revision: Optional[str] = HF_REVISION

    @property
    def models_loaded(self) -> bool:
        """Return whether the ONNX model has been loaded into memory."""
        return self._models_loaded

    def preload(self) -> None:
        """Download (if needed) and load all ONNX models into memory."""
        self._ensure_models_loaded()
        if self.routed is not None:
            self.routed.preload()

    def _ensure_models_loaded(self) -> None:
        """Resolve model path and load ONNX session on first use."""
        if self._models_loaded:
            return

        model_path = (
            Path(self._model_path_override)
            if self._model_path_override
            else self._get_default_model_path()
        )
        self.model_path = model_path

        self._load_sidecar()
        if "pipeline" in self.sidecar:

            def resolve_file(filename: str) -> Path:
                local = model_path.parent / filename
                if local.is_file():
                    return local
                if self._model_path_override or os.getenv("WITHOUTBG_MODEL_PATH"):
                    raise ModelNotFoundError(f"Bundle asset not found: {local}")
                return self._download_from_hf(filename, f"Open Weights {filename}")

            try:
                self.routed = RoutedPipeline(self.sidecar, resolve_file)
            except (KeyError, TypeError, ValueError) as e:
                raise WithoutBGError(f"Invalid model bundle: {e}") from e
        else:
            self._load_model()
        self._models_loaded = True

    def _get_default_model_path(self) -> Path:
        """Get path to the unified ONNX model from env variable or Hugging Face.

        Checks WITHOUTBG_MODEL_PATH environment variable first.
        If not set, downloads from Hugging Face.
        """
        env_path = os.getenv("WITHOUTBG_MODEL_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
            raise ModelNotFoundError(
                f"Model not found at path specified in WITHOUTBG_MODEL_PATH: {env_path}"
            )
        return self._download_from_hf(
            HF_MODEL_FILENAME, "withoutBG Open Weights Model ONNX model"
        )

    def _download_from_hf(self, filename: str, model_name: str) -> Path:
        """Download model from Hugging Face Hub with caching.

        Args:
            filename: Name of the model file to download
            model_name: Human-readable name for error messages

        Returns:
            Path to the downloaded model file

        Raises:
            ModelNotFoundError: If download fails or HF Hub is not available
        """
        try:
            model_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    revision=self._hf_revision,
                    cache_dir=None,
                )
            )
            # Keep the sidecar and all branch downloads on the same Hub snapshot.
            if model_path.parent.parent.name == "snapshots":
                self._hf_revision = model_path.parent.name
            return model_path

        except Exception as e:
            raise ModelNotFoundError(
                f"Failed to download {model_name} from Hugging Face: {str(e)}\n"
                f"You can manually download models from: "
                f"https://huggingface.co/{HF_REPO_ID}"
            ) from e

    def _load_sidecar(self) -> None:
        """Load sidecar metadata JSON alongside the ONNX model."""
        if self.model_path is None:
            raise ModelNotFoundError("Model path not resolved")

        sidecar_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        if sidecar_path.exists():
            self.sidecar = json.loads(sidecar_path.read_text())
            return

        try:
            sidecar_hf_path = self._download_from_hf(
                HF_SIDECAR_FILENAME, "Model sidecar metadata"
            )
            self.sidecar = json.loads(Path(sidecar_hf_path).read_text())
        except Exception as e:
            raise ModelNotFoundError(
                f"Sidecar metadata not found at {sidecar_path}: {e}"
            ) from e

    def _load_model(self) -> None:
        """Load the unified ONNX model."""
        try:
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                str(self.model_path), providers=providers
            )
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model: {str(e)}") from e

    def _letterbox_image(self, image: Image.Image) -> tuple[np.ndarray, int, int]:
        """Prepare letterboxed RGB tensor for ONNX inference.

        Returns:
            Tuple of (rgb_tensor, new_w, new_h) where new_w/new_h are the
            resized image dimensions before padding.
        """
        canvas = int(self.sidecar.get("canvas_size", 448))
        orig_w, orig_h = image.size
        scale = canvas / max(orig_w, orig_h)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))

        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (canvas, canvas), (0, 0, 0))
        padded.paste(resized, (0, 0))

        rgb = np.asarray(padded, dtype=np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))[None, ...]

        return rgb, new_w, new_h

    def _postprocess_alpha(
        self,
        alpha_canvas: np.ndarray,
        new_w: int,
        new_h: int,
        orig_w: int,
        orig_h: int,
    ) -> Image.Image:
        """Crop and resize alpha output to original image dimensions."""
        canvas = int(self.sidecar.get("canvas_size", 448))
        output_shape = self.sidecar.get("output_shape", [1, 1, canvas, canvas])
        output_canvas = int(self.sidecar.get("output_canvas_size", output_shape[2]))

        # v10: input and output canvas are equal (448), so this scale is 1.0.
        # Kept for back-compat with older sidecars where output was smaller.
        crop_h = max(1, round(new_h * output_canvas / canvas))
        crop_w = max(1, round(new_w * output_canvas / canvas))
        alpha_crop = alpha_canvas[:crop_h, :crop_w]

        alpha_u8 = np.clip(alpha_crop * 255.0, 0, 255).astype(np.uint8)
        alpha = Image.fromarray(alpha_u8, "L").resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
        return alpha

    def estimate_alpha(
        self, image: Image.Image, progress_callback: Optional[Callable] = None
    ) -> Image.Image:
        """Run ONNX inference to estimate the alpha channel.

        Parameters:
        - image (PIL.Image.Image): Input RGB image.
        - progress_callback: Optional callback function for progress updates.

        Returns:
        - PIL.Image.Image: Alpha channel at the original image resolution.
        """
        if progress_callback:
            progress_callback(0.0)

        self._ensure_models_loaded()

        if self.routed is not None:
            alpha, _route = self.routed.estimate_alpha(image)
            if progress_callback:
                progress_callback(1.0)
            return alpha

        orig_w, orig_h = image.size
        rgb, new_w, new_h = self._letterbox_image(image)

        if progress_callback:
            progress_callback(0.3)

        if self.session is None:
            raise ModelNotFoundError("ONNX model not loaded")

        input_name = self.sidecar.get("input_name", "rgb")
        alpha_output = self.session.run(None, {input_name: rgb})[0][0, 0]

        if progress_callback:
            progress_callback(0.8)

        alpha = self._postprocess_alpha(alpha_output, new_w, new_h, orig_w, orig_h)

        if progress_callback:
            progress_callback(1.0)

        return alpha

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background from image using the withoutBG Open Weights Model.

        Args:
            input_image: Input image
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments (unused for withoutBG Open Weights Model)

        Returns:
            PIL Image with background removed
        """
        # Load image
        if isinstance(input_image, (str, Path)):
            with Image.open(input_image) as img:
                image = img.copy()
        elif isinstance(input_image, bytes):
            with Image.open(io.BytesIO(input_image)) as img:
                image = img.copy()
        elif isinstance(input_image, Image.Image):
            image = input_image.copy()
        else:
            raise WithoutBGError(f"Unsupported input type: {type(input_image)}")

        # Apply EXIF orientation correction right after loading
        image = _apply_exif_orientation(image)

        # Convert to RGB if needed (model expects RGB only)
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            alpha_channel = self.estimate_alpha(
                image, progress_callback=progress_callback
            )

            if image.mode != "RGBA":
                image = image.convert("RGBA")

            image_array = np.array(image)
            image_array[:, :, 3] = np.array(alpha_channel)
            result_image = Image.fromarray(image_array, "RGBA")

            return result_image

        except WithoutBGError:
            raise
        except Exception as e:
            raise WithoutBGError(f"Model inference failed: {str(e)}") from e


# Deprecated alias — will be removed in a future release
OpenSourceModel = OpenWeightsModel
