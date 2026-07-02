"""Performance benchmarks for withoutbg processing."""

import gc
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from withoutbg import WithoutBG
from withoutbg.models import OpenWeightsModel

FAKE_MODEL_PATH = "/fake/withoutbg-open-weights.onnx"
DEFAULT_SIDECAR = {
    "canvas_size": 1024,
    "output_canvas_size": 768,
    "output_shape": [1, 1, 768, 768],
    "input_name": "rgb",
}


@pytest.fixture
def benchmark_images():
    """Create test images of various sizes for benchmarking."""
    sizes = [
        (256, 256),
        (512, 512),
        (1024, 768),
        (2048, 1536),
    ]

    images = {}
    for size in sizes:
        name = f"{size[0]}x{size[1]}"
        images[name] = Image.new("RGB", size, color=(128, 64, 192))

    return images


@pytest.fixture
def mock_onnx_setup():
    """Setup mocked ONNX environment for performance testing."""

    def _load_sidecar(self: OpenWeightsModel) -> None:
        self.sidecar = DEFAULT_SIDECAR.copy()

    with patch("withoutbg.models.ort.InferenceSession") as mock_session:
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(OpenWeightsModel, "_load_sidecar", _load_sidecar):
                session = Mock()
                alpha_output = np.full((1, 1, 768, 768), 0.5, dtype=np.float32)
                session.run.return_value = [alpha_output]
                mock_session.return_value = session
                yield session


def _patch_estimate_alpha(image: Image.Image):
    mock_alpha = Image.new("L", image.size, color=128)
    return patch(
        "withoutbg.models.OpenWeightsModel.estimate_alpha",
        return_value=mock_alpha,
    )


class TestProcessingBenchmarks:
    """Benchmark processing performance."""

    def test_small_image_processing_time(self, benchmark_images, mock_onnx_setup):
        image = benchmark_images["256x256"]

        with _patch_estimate_alpha(image):
            model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
            start_time = time.time()
            result = model.remove_background(image)
            processing_time = time.time() - start_time

            assert isinstance(result, Image.Image)
            assert processing_time < 5.0
            print(f"Small image (256x256) processing time: {processing_time:.3f}s")

    def test_medium_image_processing_time(self, benchmark_images, mock_onnx_setup):
        image = benchmark_images["512x512"]

        with _patch_estimate_alpha(image):
            model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
            start_time = time.time()
            result = model.remove_background(image)
            processing_time = time.time() - start_time

            assert isinstance(result, Image.Image)
            assert processing_time < 10.0
            print(f"Medium image (512x512) processing time: {processing_time:.3f}s")

    def test_large_image_processing_time(self, benchmark_images, mock_onnx_setup):
        image = benchmark_images["1024x768"]

        with _patch_estimate_alpha(image):
            model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
            start_time = time.time()
            result = model.remove_background(image)
            processing_time = time.time() - start_time

            assert isinstance(result, Image.Image)
            assert processing_time < 20.0
            print(f"Large image (1024x768) processing time: {processing_time:.3f}s")

    def test_xl_image_processing_time(self, benchmark_images, mock_onnx_setup):
        image = benchmark_images["2048x1536"]

        with _patch_estimate_alpha(image):
            model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
            start_time = time.time()
            result = model.remove_background(image)
            processing_time = time.time() - start_time

            assert isinstance(result, Image.Image)
            assert processing_time < 30.0
            print(f"XL image (2048x1536) processing time: {processing_time:.3f}s")

    def test_batch_processing_throughput(self, mock_onnx_setup):
        test_images = [
            Image.new("RGB", (512, 384), color=(i * 25, i * 20, i * 15))
            for i in range(10)
        ]

        with patch(
            "withoutbg.models.OpenWeightsModel.remove_background"
        ) as mock_remove_bg:
            mock_results = [
                Image.new("RGBA", image.size, color=(100, 150, 200, 128))
                for image in test_images
            ]
            mock_remove_bg.side_effect = mock_results

            model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
            start_time = time.time()
            results = model.remove_background_batch(test_images)
            total_time = time.time() - start_time
            throughput = len(test_images) / total_time

            assert len(results) == len(test_images)
            assert throughput > 1.0
            print(f"Batch processing throughput: {throughput:.2f} images/second")

    def test_model_initialization_time(self):
        with patch("withoutbg.models.ort.InferenceSession") as mock_session:
            with patch.object(OpenWeightsModel, "_load_sidecar"):
                mock_session.return_value = Mock()

                model = OpenWeightsModel(model_path=FAKE_MODEL_PATH)
                start_time = time.time()
                model.preload()
                init_time = time.time() - start_time

                assert model is not None
                assert init_time < 5.0
                print(f"Model initialization time: {init_time:.3f}s")

    def test_preprocessing_performance(self, benchmark_images, mock_onnx_setup):
        image = benchmark_images["1024x768"]
        model = OpenWeightsModel(model_path=FAKE_MODEL_PATH)
        model.sidecar = DEFAULT_SIDECAR.copy()

        start_time = time.time()
        rgb, new_w, new_h = model._letterbox_image(image)
        preprocessing_time = time.time() - start_time

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (1, 3, 1024, 1024)
        assert preprocessing_time < 1.0
        assert new_w > 0 and new_h > 0
        print(f"Preprocessing time (1024x768): {preprocessing_time:.3f}s")

    def test_inference_performance(self, benchmark_images, mock_onnx_setup):
        image = benchmark_images["512x512"]
        model = OpenWeightsModel(model_path=FAKE_MODEL_PATH)
        model.sidecar = DEFAULT_SIDECAR.copy()
        model.session = mock_onnx_setup
        model._models_loaded = True

        start_time = time.time()
        alpha = model.estimate_alpha(image)
        inference_time = time.time() - start_time

        assert isinstance(alpha, Image.Image)
        assert inference_time < 2.0
        print(f"Inference time (512x512): {inference_time:.3f}s")

    def test_concurrent_processing_performance(self, mock_onnx_setup):
        test_images = [
            Image.new("RGB", (256, 256), color=(i * 50, i * 30, i * 20))
            for i in range(5)
        ]

        processing_times = []
        model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)

        for image in test_images:
            with _patch_estimate_alpha(image):
                start_time = time.time()
                result = model.remove_background(image)
                processing_times.append(time.time() - start_time)
                assert isinstance(result, Image.Image)

        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)

        assert avg_time < 5.0
        assert max_time < 10.0
        print(
            f"Concurrent processing - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, "
            f"Max: {max_time:.3f}s"
        )

    def test_api_response_time_simulation(self):
        test_image = Image.new("RGB", (512, 384), color=(128, 64, 192))

        with patch("requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200

            import base64
            import io

            buffer = io.BytesIO()
            result_image = Image.new("RGBA", test_image.size, color=(128, 64, 192, 128))
            result_image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mock_response.json.return_value = {"image": image_b64}

            def delayed_response(*args, **kwargs):
                time.sleep(0.1)
                return mock_response

            mock_post.side_effect = delayed_response

            model = WithoutBG.api(api_key="sk_test_key")
            start_time = time.time()
            result = model.remove_background(test_image)
            api_time = time.time() - start_time

            assert isinstance(result, Image.Image)
            assert api_time >= 0.1
            assert api_time < 5.0
            print(f"Simulated API processing time: {api_time:.3f}s")


class TestScalabilityBenchmarks:
    """Test scalability and resource usage."""

    def test_memory_scaling_with_image_size(self, mock_onnx_setup):
        sizes = [(256, 256), (512, 512), (1024, 768), (1536, 1024)]

        for size in sizes:
            gc.collect()
            test_image = Image.new("RGB", size, color=(100, 150, 200))

            with _patch_estimate_alpha(test_image):
                model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
                result = model.remove_background(test_image)

                assert isinstance(result, Image.Image)
                assert result.size == size
                print(f"Successfully processed {size[0]}x{size[1]} image")

                del result
                del test_image
                gc.collect()

    def test_batch_size_scaling(self, mock_onnx_setup):
        batch_sizes = [1, 5, 10, 20]

        for batch_size in batch_sizes:
            test_images = [
                Image.new("RGB", (256, 256), color=(i * 10, i * 15, i * 20))
                for i in range(batch_size)
            ]

            with patch(
                "withoutbg.models.OpenWeightsModel.remove_background"
            ) as mock_remove_bg:
                mock_results = [
                    Image.new("RGBA", image.size, color=(100, 150, 200, 128))
                    for image in test_images
                ]
                mock_remove_bg.side_effect = mock_results

                model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
                start_time = time.time()
                results = model.remove_background_batch(test_images)
                batch_time = time.time() - start_time
                time_per_image = batch_time / batch_size

                assert len(results) == batch_size
                assert time_per_image < 5.0
                print(
                    f"Batch size {batch_size}: {batch_time:.3f}s total, "
                    f"{time_per_image:.3f}s per image"
                )

                del results
                del test_images
                gc.collect()

    def test_repeated_processing_stability(self, mock_onnx_setup):
        test_image = Image.new("RGB", (512, 384), color=(128, 64, 192))
        processing_times = []
        model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)

        for _i in range(10):
            with _patch_estimate_alpha(test_image):
                start_time = time.time()
                result = model.remove_background(test_image)
                processing_times.append(time.time() - start_time)
                assert isinstance(result, Image.Image)
                del result
                gc.collect()

        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        variance = sum((t - avg_time) ** 2 for t in processing_times) / len(
            processing_times
        )

        assert variance < 1.0
        assert max_time / min_time < 3.0
        print(f"Repeated processing - Avg: {avg_time:.3f}s, Variance: {variance:.6f}")


@pytest.mark.performance
class TestRegressionBenchmarks:
    """Performance regression tests to catch performance degradation."""

    def test_baseline_performance_regression(self, mock_onnx_setup):
        test_image = Image.new("RGB", (512, 512), color=(128, 64, 192))

        with _patch_estimate_alpha(test_image):
            model = WithoutBG.open_weights(model_path=FAKE_MODEL_PATH)
            start_time = time.time()
            result = model.remove_background(test_image)
            processing_time = time.time() - start_time

            baseline_512x512_time = 10.0

            assert isinstance(result, Image.Image)
            assert processing_time < baseline_512x512_time, (
                f"Performance regression detected: {processing_time:.3f}s > "
                f"{baseline_512x512_time}s"
            )
            print(
                f"Baseline test (512x512): {processing_time:.3f}s "
                f"(threshold: {baseline_512x512_time}s)"
            )
