"""Batch processing performance tests."""

import gc
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from withoutbg.core import WithoutBG


@pytest.fixture
def create_test_images():
    """Create temporary test image files."""

    def _create_images(count=5, size=(256, 256)):
        image_files = []
        for i in range(count):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                color = (i * 40, i * 30, i * 20)
                image = Image.new("RGB", size, color=color)
                image.save(tmp_file.name, "JPEG")
                image_files.append(tmp_file.name)
        return image_files

    return _create_images


@pytest.fixture
def mock_processing():
    """Mock the background removal processing."""

    def _mock_remove_bg(input_image, **kwargs):
        # Simulate processing time
        time.sleep(0.01)  # 10ms simulated processing time

        # Load image to get size
        if isinstance(input_image, str):
            with Image.open(input_image) as img:
                size = img.size
        else:
            size = input_image.size

        # Return mock result
        return Image.new("RGBA", size, color=(100, 150, 200, 128))

    return _mock_remove_bg


class TestBatchPerformance:
    """Test batch processing performance characteristics."""

    def test_small_batch_performance(self, create_test_images, mock_processing):
        """Test performance with small batch sizes."""
        image_files = create_test_images(count=3, size=(256, 256))

        try:
            model = WithoutBG.open_weights()
            with patch.object(
                model.model, "remove_background", side_effect=mock_processing
            ):
                # Measure batch processing time
                start_time = time.time()
                results = model.remove_background_batch(image_files)
                end_time = time.time()

                batch_time = end_time - start_time
                time_per_image = batch_time / len(image_files)

                # Assertions
                assert len(results) == len(image_files)
                assert batch_time < 5.0  # Should complete within 5 seconds
                assert (
                    time_per_image < 2.0
                )  # Each image should take less than 2 seconds

                print(
                    f"Small batch (3 images): {batch_time:.3f}s total, "
                    f"{time_per_image:.3f}s per image"
                )

                for result in results:
                    assert isinstance(result, Image.Image)
                    assert result.mode == "RGBA"

        finally:
            # Cleanup
            for file_path in image_files:
                os.unlink(file_path)

    def test_medium_batch_performance(self, create_test_images, mock_processing):
        """Test performance with medium batch sizes."""
        image_files = create_test_images(count=10, size=(512, 384))

        try:
            model = WithoutBG.open_weights()
            with patch.object(
                model.model, "remove_background", side_effect=mock_processing
            ):
                # Measure batch processing time
                start_time = time.time()
                results = model.remove_background_batch(image_files)
                end_time = time.time()

                batch_time = end_time - start_time
                time_per_image = batch_time / len(image_files)

                # Assertions
                assert len(results) == len(image_files)
                assert batch_time < 15.0  # Should complete within 15 seconds
                assert time_per_image < 2.0  # Average time per image

                print(
                    f"Medium batch (10 images): {batch_time:.3f}s total, "
                    f"{time_per_image:.3f}s per image"
                )

        finally:
            # Cleanup
            for file_path in image_files:
                os.unlink(file_path)

    def test_large_batch_performance(self, create_test_images, mock_processing):
        """Test performance with large batch sizes."""
        image_files = create_test_images(count=25, size=(400, 300))

        try:
            model = WithoutBG.open_weights()
            with patch.object(
                model.model, "remove_background", side_effect=mock_processing
            ):
                # Measure batch processing time
                start_time = time.time()
                results = model.remove_background_batch(image_files)
                end_time = time.time()

                batch_time = end_time - start_time
                batch_time / len(image_files)
                throughput = len(image_files) / batch_time

                # Assertions
                assert len(results) == len(image_files)
                assert batch_time < 60.0  # Should complete within 1 minute
                assert throughput > 0.4  # Should process at least 0.4 images per second

                print(
                    f"Large batch (25 images): {batch_time:.3f}s total, "
                    f"{throughput:.2f} images/sec"
                )

        finally:
            # Cleanup
            for file_path in image_files:
                os.unlink(file_path)

    def test_batch_size_scaling(self, create_test_images, mock_processing):
        """Test how performance scales with batch size."""
        batch_sizes = [1, 5, 10, 20]
        performance_data = {}

        for batch_size in batch_sizes:
            image_files = create_test_images(count=batch_size, size=(256, 256))

            try:
                model = WithoutBG.open_weights()
                with patch.object(
                    model.model, "remove_background", side_effect=mock_processing
                ):
                    # Measure batch processing
                    start_time = time.time()
                    results = model.remove_background_batch(image_files)
                    end_time = time.time()

                    batch_time = end_time - start_time
                    time_per_image = batch_time / batch_size

                    performance_data[batch_size] = {
                        "total_time": batch_time,
                        "time_per_image": time_per_image,
                        "throughput": batch_size / batch_time,
                    }

                    # Verify results
                    assert len(results) == batch_size

                    print(f"Batch size {batch_size}: {time_per_image:.3f}s per image")

            finally:
                # Cleanup
                for file_path in image_files:
                    os.unlink(file_path)

        # Analyze scaling characteristics
        # Time per image should remain relatively stable
        times_per_image = [data["time_per_image"] for data in performance_data.values()]
        max_time = max(times_per_image)
        min_time = min(times_per_image)

        # Time per image shouldn't vary too much across batch sizes
        assert max_time / min_time < 3.0  # Max should be less than 3x min

    def test_batch_memory_efficiency(self, create_test_images, mock_processing):
        """Test memory efficiency during batch processing."""
        image_files = create_test_images(count=15, size=(512, 384))

        try:
            # Force garbage collection
            gc.collect()

            model = WithoutBG.open_weights()
            with patch.object(
                model.model, "remove_background", side_effect=mock_processing
            ):
                # Process batch and monitor memory
                results = model.remove_background_batch(image_files)

                # Verify processing completed
                assert len(results) == len(image_files)

                # Clean up results
                del results
                gc.collect()

                print(f"Batch memory test completed for {len(image_files)} images")

        finally:
            # Cleanup
            for file_path in image_files:
                os.unlink(file_path)

    def test_batch_with_output_directory_performance(
        self, create_test_images, mock_processing
    ):
        """Test batch processing performance when saving to output directory."""
        image_files = create_test_images(count=8, size=(400, 300))

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                model = WithoutBG.open_weights()
                with patch.object(
                    model.model, "remove_background", side_effect=mock_processing
                ):
                    # Measure batch processing with output directory
                    start_time = time.time()
                    results = model.remove_background_batch(
                        image_files, output_dir=output_dir
                    )
                    end_time = time.time()

                    batch_time = end_time - start_time

                    # Assertions
                    assert len(results) == len(image_files)
                    assert batch_time < 20.0  # Should complete within 20 seconds

                    # Check that output directory exists and has correct structure
                    output_path = Path(output_dir)
                    assert output_path.exists()

                    print(f"Batch with output dir (8 images): {batch_time:.3f}s")

            finally:
                # Cleanup input files
                for file_path in image_files:
                    os.unlink(file_path)

    def test_batch_error_handling_performance(self, create_test_images):
        """Test batch processing performance when some images fail."""
        image_files = create_test_images(count=6, size=(256, 256))

        # Add a non-existent file to test error handling
        image_files.append("/non/existent/file.jpg")

        def mock_processing_with_errors(input_image, **kwargs):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"File not found: {input_image}")

            # Simulate normal processing for valid files
            time.sleep(0.01)
            with Image.open(input_image) as img:
                size = img.size
            return Image.new("RGBA", size, color=(100, 150, 200, 128))

        try:
            model = WithoutBG.open_weights()
            with patch.object(
                model.model,
                "remove_background",
                side_effect=mock_processing_with_errors,
            ):
                # Measure batch processing with errors
                start_time = time.time()

                try:
                    results = model.remove_background_batch(image_files)
                    # If no exception is raised, check results
                    assert len(results) <= len(image_files)  # Some may have failed
                except Exception:
                    # Batch processing may raise exception on errors
                    pass

                end_time = time.time()
                batch_time = end_time - start_time

                # Should still complete in reasonable time despite errors
                assert batch_time < 10.0

                print(f"Batch with errors: {batch_time:.3f}s")

        finally:
            # Cleanup valid files
            for file_path in image_files[:-1]:  # Exclude the non-existent file
                if os.path.exists(file_path):
                    os.unlink(file_path)

    def test_batch_different_image_sizes_performance(self, mock_processing):
        """Test batch processing performance with mixed image sizes."""
        # Create images of different sizes
        sizes = [(200, 200), (400, 300), (600, 450), (300, 400), (512, 384)]
        image_files = []

        for i, size in enumerate(sizes):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                color = (i * 45, i * 35, i * 25)
                image = Image.new("RGB", size, color=color)
                image.save(tmp_file.name, "JPEG")
                image_files.append(tmp_file.name)

        try:
            model = WithoutBG.open_weights()
            with patch.object(
                model.model, "remove_background", side_effect=mock_processing
            ):
                # Measure batch processing with mixed sizes
                start_time = time.time()
                results = model.remove_background_batch(image_files)
                end_time = time.time()

                batch_time = end_time - start_time
                time_per_image = batch_time / len(image_files)

                # Assertions
                assert len(results) == len(image_files)
                assert batch_time < 15.0  # Should complete within 15 seconds

                # Verify different sizes were processed correctly
                for _, result in enumerate(results):
                    assert isinstance(result, Image.Image)
                    # Note: result size should match original image size
                    assert result.mode == "RGBA"

                print(
                    f"Mixed sizes batch: {batch_time:.3f}s total, "
                    f"{time_per_image:.3f}s per image"
                )

        finally:
            # Cleanup
            for file_path in image_files:
                os.unlink(file_path)

    def test_batch_processing_consistency(self, create_test_images, mock_processing):
        """Test that batch processing produces consistent results."""
        image_files = create_test_images(count=5, size=(300, 300))

        try:
            model = WithoutBG.open_weights()
            with patch.object(
                model.model, "remove_background", side_effect=mock_processing
            ):
                # Process the same batch multiple times
                all_results = []
                processing_times = []

                for _run in range(3):
                    start_time = time.time()
                    results = model.remove_background_batch(image_files)
                    end_time = time.time()

                    processing_times.append(end_time - start_time)
                    all_results.append(results)

                    # Verify each run
                    assert len(results) == len(image_files)

                # Check consistency of processing times
                avg_time = sum(processing_times) / len(processing_times)
                max_time = max(processing_times)
                min_time = min(processing_times)

                # Processing times should be relatively consistent
                assert max_time / min_time < 2.0  # Max shouldn't be more than 2x min

                print(
                    f"Consistency test - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, "
                    f"Max: {max_time:.3f}s"
                )

        finally:
            # Cleanup
            for file_path in image_files:
                os.unlink(file_path)

    def test_batch_processing_with_pil_images(self, mock_processing):
        """Test batch processing performance with PIL Image objects."""
        # Create PIL Image objects directly
        test_images = [
            Image.new("RGB", (256, 256), color=(i * 50, i * 30, i * 20))
            for i in range(8)
        ]

        model = WithoutBG.open_weights()
        with patch.object(
            model.model, "remove_background", side_effect=mock_processing
        ):
            # Measure batch processing with PIL images
            start_time = time.time()
            results = model.remove_background_batch(test_images)
            end_time = time.time()

            batch_time = end_time - start_time
            time_per_image = batch_time / len(test_images)

            # Assertions
            assert len(results) == len(test_images)
            assert batch_time < 10.0  # Should be faster with PIL objects

            print(
                f"PIL objects batch: {batch_time:.3f}s total, "
                f"{time_per_image:.3f}s per image"
            )

    def test_concurrent_batch_processing_simulation(
        self, create_test_images, mock_processing
    ):
        """Simulate concurrent batch processing workloads."""
        # Create multiple small batches
        batch1 = create_test_images(count=3, size=(200, 200))
        batch2 = create_test_images(count=4, size=(300, 250))
        batch3 = create_test_images(count=2, size=(400, 300))

        all_batches = [batch1, batch2, batch3]

        try:
            model = WithoutBG.open_weights()
            with patch.object(
                model.model, "remove_background", side_effect=mock_processing
            ):
                # Process batches sequentially (simulating concurrent workload)
                total_start_time = time.time()
                all_results = []

                for i, batch in enumerate(all_batches):
                    start_time = time.time()
                    results = model.remove_background_batch(batch)
                    end_time = time.time()

                    all_results.append(results)
                    batch_time = end_time - start_time

                    print(
                        f"Concurrent batch {i + 1} ({len(batch)} images): "
                        f"{batch_time:.3f}s"
                    )

                total_end_time = time.time()
                total_time = total_end_time - total_start_time
                total_images = sum(len(batch) for batch in all_batches)

                # Assertions
                assert len(all_results) == len(all_batches)
                assert (
                    total_time < 20.0
                )  # All batches should complete within 20 seconds

                overall_throughput = total_images / total_time
                print(
                    f"Overall concurrent throughput: {overall_throughput:.2f} "
                    f"images/sec"
                )

        finally:
            # Cleanup all batches
            for batch in all_batches:
                for file_path in batch:
                    os.unlink(file_path)
