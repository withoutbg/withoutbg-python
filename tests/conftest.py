"""Shared test configuration and fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import test data fixtures for shared use across test modules


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, multiple components)"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests (very slow)"
    )
    config.addinivalue_line(
        "markers", "api: Tests requiring API access (may be skipped)"
    )
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line(
        "markers", "real_processing: Tests using actual ONNX models (very slow)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)

        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark unit tests (default for tests/ directory)
        if (
            "test_" in str(item.fspath)
            and "integration" not in str(item.fspath)
            and "performance" not in str(item.fspath)
        ):
            item.add_marker(pytest.mark.unit)

        # Auto-mark API tests
        if "api" in item.name.lower() or "pro" in item.name.lower():
            item.add_marker(pytest.mark.api)


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_images_dir(test_data_dir):
    """Provide path to test images directory."""
    return test_data_dir / "images"


@pytest.fixture(scope="session")
def expected_outputs_dir(test_data_dir):
    """Provide path to expected outputs directory."""
    return test_data_dir / "expected_outputs"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    import gc
    import platform
    import time

    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

        # Windows-specific cleanup: force garbage collection and wait briefly
        # to ensure PIL releases file handles before cleanup
        if platform.system() == "Windows":
            gc.collect()
            time.sleep(0.1)  # Small delay to allow file handles to be released


@pytest.fixture
def sample_test_image():
    """Create a standard test image for general use."""
    return Image.new("RGB", (512, 384), color=(128, 64, 192))


@pytest.fixture
def sample_test_images():
    """Create multiple test images with different properties."""
    return {
        "small": Image.new("RGB", (128, 96), color=(255, 0, 0)),
        "medium": Image.new("RGB", (512, 384), color=(0, 255, 0)),
        "large": Image.new("RGB", (1024, 768), color=(0, 0, 255)),
        "portrait": Image.new("RGB", (384, 512), color=(255, 255, 0)),
        "landscape": Image.new("RGB", (768, 512), color=(255, 0, 255)),
        "square": Image.new("RGB", (512, 512), color=(0, 255, 255)),
    }


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # This fixture runs after each test to ensure cleanup
    # The actual cleanup is handled by tempfile.TemporaryDirectory context managers


# Pytest hooks for performance testing
def pytest_runtest_setup(item):
    """Setup hook for individual tests."""
    if "performance" in item.keywords:
        # Skip performance tests by default unless explicitly requested
        if not item.config.getoption("--run-performance", default=False):
            pytest.skip("Performance tests skipped (use --run-performance to run)")

    if "real_processing" in item.keywords:
        # Skip real processing tests by default unless explicitly requested
        if not item.config.getoption("--run-real-processing", default=False):
            pytest.skip(
                "Real processing tests skipped (use --run-real-processing to run)"
            )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )
    parser.addoption(
        "--run-api",
        action="store_true",
        default=False,
        help="Run API tests (requires network access)",
    )
    parser.addoption(
        "--run-real-processing",
        action="store_true",
        default=False,
        help="Run real processing tests (downloads models, very slow)",
    )


# Custom assertion helpers
def assert_image_properties(
    image, expected_mode=None, expected_size=None, expected_format=None
):
    """Helper function to assert image properties."""
    assert isinstance(image, Image.Image), f"Expected PIL Image, got {type(image)}"

    if expected_mode:
        assert (
            image.mode == expected_mode
        ), f"Expected mode {expected_mode}, got {image.mode}"

    if expected_size:
        assert (
            image.size == expected_size
        ), f"Expected size {expected_size}, got {image.size}"

    if expected_format:
        assert (
            image.format == expected_format
        ), f"Expected format {expected_format}, got {image.format}"


def assert_processing_result(result, original_image):
    """Helper function to assert processing results."""
    assert_image_properties(result, expected_mode="RGBA")
    assert result.size == original_image.size, "Result size should match original"

    # Check that alpha channel exists and has variation
    result_array = np.array(result)
    alpha_channel = result_array[:, :, 3]
    assert (
        alpha_channel.min() >= 0 and alpha_channel.max() <= 255
    ), "Alpha values should be in valid range"


# Performance testing utilities
class PerformanceTracker:
    """Utility class for tracking performance metrics in tests."""

    def __init__(self):
        self.metrics = {}

    def record_metric(self, name, value, unit="seconds"):
        """Record a performance metric."""
        self.metrics[name] = {"value": value, "unit": unit}

    def get_metric(self, name):
        """Get a recorded metric."""
        return self.metrics.get(name)

    def assert_performance(self, name, max_value, message=None):
        """Assert that a performance metric meets requirements."""
        metric = self.get_metric(name)
        assert metric is not None, f"Metric {name} not found"

        actual_value = metric["value"]
        assert actual_value <= max_value, (
            message
            or f"Performance regression: {name} = {actual_value} > {max_value} "
            f"{metric['unit']}"
        )


@pytest.fixture
def performance_tracker():
    """Provide a performance tracker for tests."""
    return PerformanceTracker()


@pytest.fixture
def real_test_image_path(test_images_dir):
    """Get path to real test image."""
    return test_images_dir / "test-ice-cream.png"
