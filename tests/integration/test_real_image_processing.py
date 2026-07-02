"""Real image processing tests using actual ONNX models."""

import os
from typing import Optional

import numpy as np
import pytest
from PIL import Image

from withoutbg import WithoutBG
from withoutbg.exceptions import ModelNotFoundError, WithoutBGError
from withoutbg.models import OpenWeightsModel


def assert_alpha_channel_valid(result_image: Image.Image) -> None:
    """Validate alpha channel has proper background removal characteristics."""
    assert result_image.mode == "RGBA", f"Expected RGBA, got {result_image.mode}"

    result_array = np.array(result_image)
    alpha = result_array[:, :, 3]

    # Alpha should have variation (not all 0 or all 255)
    alpha_min, alpha_max = alpha.min(), alpha.max()
    assert (
        alpha_min < alpha_max
    ), f"Alpha channel lacks variation: min={alpha_min}, max={alpha_max}"

    # Should have some transparency (background removal)
    transparent_pixels = np.sum(alpha < 255)
    assert transparent_pixels > 0, "No background pixels detected (all opaque)"

    # Should preserve some foreground
    opaque_pixels = np.sum(alpha > 0)
    assert opaque_pixels > 0, "No foreground pixels preserved (all transparent)"

    # Alpha values should be in valid range
    assert 0 <= alpha_min <= 255, f"Invalid alpha min: {alpha_min}"
    assert 0 <= alpha_max <= 255, f"Invalid alpha max: {alpha_max}"


def calculate_alpha_iou(result: Image.Image, expected: Image.Image) -> float:
    """Calculate IoU (Intersection over Union) for alpha channel masks.

    Args:
        result: Result image with alpha channel
        expected: Expected image with alpha channel

    Returns:
        IoU score between 0.0 and 1.0
    """
    assert (
        result.size == expected.size
    ), f"Size mismatch: {result.size} vs {expected.size}"
    assert result.mode == "RGBA", f"Result must be RGBA, got {result.mode}"
    assert expected.mode == "RGBA", f"Expected must be RGBA, got {expected.mode}"

    # Extract alpha channels and normalize to [0,1]
    result_alpha = np.array(result)[:, :, 3] / 255.0
    expected_alpha = np.array(expected)[:, :, 3] / 255.0

    # Binarize by rounding (>0.5 = foreground, <=0.5 = background)
    result_binary = (result_alpha > 0.5).astype(np.uint8)
    expected_binary = (expected_alpha > 0.5).astype(np.uint8)

    # Calculate IoU
    intersection = np.sum(result_binary & expected_binary)
    union = np.sum(result_binary | expected_binary)

    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return float(iou)


def assert_alpha_iou(
    result: Image.Image, expected: Image.Image, min_iou: float = 0.97
) -> None:
    """Assert that alpha channel IoU meets minimum threshold."""
    iou = calculate_alpha_iou(result, expected)
    assert iou >= min_iou, f"Alpha IoU too low: {iou:.4f} < {min_iou}"


def assert_images_similar(
    result: Image.Image, expected: Image.Image, tolerance: float = 0.15
) -> None:
    """Compare two images with tolerance for minor differences."""
    assert (
        result.size == expected.size
    ), f"Size mismatch: {result.size} vs {expected.size}"
    assert (
        result.mode == expected.mode
    ), f"Mode mismatch: {result.mode} vs {expected.mode}"

    result_array = np.array(result, dtype=np.float32) / 255.0
    expected_array = np.array(expected, dtype=np.float32) / 255.0

    # Calculate mean absolute difference
    diff = np.mean(np.abs(result_array - expected_array))
    assert diff <= tolerance, f"Images too different: MAE={diff:.3f} > {tolerance}"


@pytest.fixture(scope="session")
def real_open_weights() -> Optional[OpenWeightsModel]:
    """Load actual ONNX model if available, skip if not."""
    try:
        model = OpenWeightsModel()
        model.preload()
        assert model.session is not None, "ONNX model not loaded"
        return model
    except (FileNotFoundError, ModelNotFoundError, Exception) as e:
        pytest.skip(f"Real ONNX model not available: {e}")


@pytest.mark.integration
@pytest.mark.real_processing
@pytest.mark.skipif(
    os.getenv("CI") == "true" and not os.getenv("RUN_REAL_PROCESSING"),
    reason="Real processing tests skipped in CI (set RUN_REAL_PROCESSING=1 to enable)",
)
class TestRealImageProcessing:
    """Integration tests using actual ONNX models and real images."""

    def test_ice_cream_processing(
        self, real_open_weights, test_images_dir, expected_outputs_dir
    ):
        """Test processing the ice cream image with real models."""
        input_path = test_images_dir / "test-ice-cream.png"
        assert input_path.exists(), f"Test image not found: {input_path}"

        # Process with real model
        result = real_open_weights.remove_background(input_path)

        # Validate result properties
        assert_alpha_channel_valid(result)

        # Check against expected output with IoU metric
        expected_path = expected_outputs_dir / "test-ice-cream.png"
        if expected_path.exists():
            with Image.open(expected_path) as expected:
                # Use IoU for interpretable alpha channel comparison
                assert_alpha_iou(result, expected, min_iou=0.97)

    def test_core_api_with_real_model(self, test_images_dir):
        """Test core API WithoutBG class with real processing."""
        input_path = test_images_dir / "test-ice-cream.png"

        # Test with withoutBG Open Weights Model
        model = WithoutBG.open_weights()
        result = model.remove_background(input_path)
        assert_alpha_channel_valid(result)

        # Result should preserve original dimensions
        with Image.open(input_path) as original:
            assert result.size == original.size

    def test_different_input_formats(self, real_open_weights, test_images_dir):
        """Test processing with different input formats."""
        input_path = test_images_dir / "test-ice-cream.png"
        with Image.open(input_path) as original_image:
            # Test with PIL Image
            result_pil = real_open_weights.remove_background(original_image)
            assert_alpha_channel_valid(result_pil)

        # Test with file path (string)
        result_path = real_open_weights.remove_background(str(input_path))
        assert_alpha_channel_valid(result_path)

        # Test with Path object
        result_pathobj = real_open_weights.remove_background(input_path)
        assert_alpha_channel_valid(result_pathobj)

        # Test with bytes
        with open(input_path, "rb") as f:
            image_bytes = f.read()
        result_bytes = real_open_weights.remove_background(image_bytes)
        assert_alpha_channel_valid(result_bytes)

        # All results should have same dimensions
        assert (
            result_pil.size
            == result_path.size
            == result_pathobj.size
            == result_bytes.size
        )

    @pytest.mark.parametrize(
        "format_name,pil_format",
        [
            ("JPEG", "JPEG"),
            ("PNG", "PNG"),
            ("WebP", "WebP"),
        ],
    )
    def test_different_image_formats(
        self, real_open_weights, sample_test_image, format_name, pil_format
    ):
        """Test processing various image formats."""
        # Convert test image to RGB for JPEG compatibility
        if format_name == "JPEG" and sample_test_image.mode == "RGBA":
            test_img = sample_test_image.convert("RGB")
        else:
            test_img = sample_test_image

        # Save to bytes in specific format
        import io

        buffer = io.BytesIO()
        test_img.save(buffer, format=pil_format)
        buffer.seek(0)

        # Process the formatted image
        result = real_open_weights.remove_background(buffer.getvalue())
        assert_alpha_channel_valid(result)

    @pytest.mark.parametrize(
        "size",
        [
            (256, 256),  # Small square
            (512, 384),  # Medium landscape
            (384, 512),  # Medium portrait
            (1024, 768),  # Large landscape
        ],
    )
    def test_different_resolutions(self, real_open_weights, size):
        """Test processing different image sizes."""
        # Create test image with specific size
        test_img = Image.new("RGB", size, color=(128, 64, 192))

        # Add some pattern to make it more realistic
        import numpy as np

        img_array = np.array(test_img)
        # Add gradient pattern
        gradient = np.linspace(0, 255, size[0]).astype(np.uint8)
        img_array[:, :, 0] = gradient[None, :]  # Horizontal gradient in red channel
        test_img = Image.fromarray(img_array)

        result = real_open_weights.remove_background(test_img)

        # Should preserve original dimensions
        assert result.size == size
        assert_alpha_channel_valid(result)

    def test_processing_consistency(self, real_open_weights, test_images_dir):
        """Test that multiple runs produce consistent results."""
        input_path = test_images_dir / "test-ice-cream.png"

        # Process same image multiple times
        result1 = real_open_weights.remove_background(input_path)
        result2 = real_open_weights.remove_background(input_path)

        # Results should be identical (deterministic) - perfect IoU
        iou = calculate_alpha_iou(result1, result2)
        assert iou == 1.0, f"Processing not deterministic: IoU={iou:.6f} < 1.0"

    def test_pipeline_stages_integration(self, real_open_weights, test_images_dir):
        """Test that inference produces reasonable outputs."""
        input_path = test_images_dir / "test-ice-cream.png"
        with Image.open(input_path) as original:
            result = real_open_weights.remove_background(original)

        # Validate the result has expected characteristics for ice cream image
        result_array = np.array(result)
        alpha = result_array[:, :, 3]

        # Should have significant variation in alpha channel
        alpha_std = np.std(alpha)
        assert alpha_std > 30, f"Alpha channel lacks variation (std={alpha_std:.1f})"

        # Should have both very transparent and very opaque regions
        very_transparent = np.sum(alpha < 50)
        very_opaque = np.sum(alpha > 200)

        assert very_transparent > 100, "Not enough background removal"
        assert very_opaque > 100, "Not enough foreground preservation"

    def test_alpha_iou_metric(
        self, real_open_weights, test_images_dir, expected_outputs_dir
    ):
        """Test alpha IoU metric calculation and interpretation."""
        input_path = test_images_dir / "test-ice-cream.png"
        expected_path = expected_outputs_dir / "test-ice-cream.png"

        # Process image
        result = real_open_weights.remove_background(input_path)

        if expected_path.exists():
            with Image.open(expected_path) as expected:
                # Calculate IoU (should be very high since expected was generated
                # with same model)
                iou = calculate_alpha_iou(result, expected)
                print(f"Alpha IoU score: {iou:.4f}")

            # Should be perfect since we generated expected output with same model
            assert iou >= 0.99, f"IoU should be near-perfect: {iou:.4f}"

            # Demonstrate interpretability: IoU represents overlap of binary masks
            result_alpha = np.array(result)[:, :, 3] / 255.0
            expected_alpha = np.array(expected)[:, :, 3] / 255.0

            result_binary = (result_alpha > 0.5).astype(np.uint8)
            expected_binary = (expected_alpha > 0.5).astype(np.uint8)

            intersection = np.sum(result_binary & expected_binary)
            union = np.sum(result_binary | expected_binary)

            print(f"Intersection pixels: {intersection}")
            print(f"Union pixels: {union}")
            print(f"IoU = {intersection}/{union} = {intersection/union:.4f}")

    def test_error_handling_with_real_model(self, real_open_weights):
        """Test error handling with real models."""
        # Test with invalid input
        with pytest.raises((WithoutBGError, ValueError, TypeError)):
            real_open_weights.remove_background(None)

        # Test with invalid file path
        with pytest.raises((FileNotFoundError, WithoutBGError)):
            real_open_weights.remove_background("/nonexistent/file.png")


@pytest.mark.performance
@pytest.mark.real_processing
class TestRealProcessingPerformance:
    """Performance tests for real image processing."""

    def test_processing_speed_benchmark(
        self, real_open_weights, test_images_dir, performance_tracker
    ):
        """Benchmark processing speed with real models."""
        input_path = test_images_dir / "test-ice-cream.png"

        import time

        start_time = time.time()
        result = real_open_weights.remove_background(input_path)
        processing_time = time.time() - start_time

        # Record performance metric
        performance_tracker.record_metric("processing_time", processing_time, "seconds")

        # Basic validation
        assert_alpha_channel_valid(result)

        # Performance assertion (adjust based on your requirements)
        performance_tracker.assert_performance(
            "processing_time",
            max_value=30.0,  # 30 seconds max for ice cream image
            message=f"Processing too slow: {processing_time:.2f}s",
        )

    def test_memory_usage_real_processing(self, real_open_weights, test_images_dir):
        """Test memory usage during real processing."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        input_path = test_images_dir / "test-ice-cream.png"
        result = real_open_weights.remove_background(input_path)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Basic validation
        assert_alpha_channel_valid(result)

        # Memory should not increase excessively (adjust threshold as needed)
        assert (
            memory_increase < 500
        ), f"Memory usage too high: {memory_increase:.1f}MB increase"
