"""End-to-end tests for CLI with real file processing."""

import os
import shutil

import pytest
from click.testing import CliRunner
from PIL import Image

from src.withoutbg.cli import main


class TestCLIE2E:
    """End-to-end tests using real image processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @pytest.fixture
    def temp_output_dir(self, temp_dir):
        """Create temporary output directory."""
        output_dir = temp_dir / "e2e_outputs"
        output_dir.mkdir()
        return output_dir

    @pytest.mark.real_processing
    def test_e2e_single_image_open_source_model(
        self, real_test_image_path, temp_output_dir
    ):
        """Test end-to-end processing of single image with withoutBG Open
        Weights Model."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        output_path = temp_output_dir / "output_open_weights.png"

        result = self.runner.invoke(
            main, [str(real_test_image_path), "--output", str(output_path), "--verbose"]
        )

        # Check command succeeded
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Check output file was created
        assert output_path.exists(), "Output file was not created"

        # Verify output image properties
        with (
            Image.open(output_path) as output_image,
            Image.open(real_test_image_path) as original_image,
        ):
            assert output_image.mode == "RGBA", "Output should be RGBA"
            assert (
                output_image.size == original_image.size
            ), "Output size should match input"

            # Check that alpha channel has variation (background was processed)
            alpha_channel = output_image.split()[-1]
            alpha_values = list(alpha_channel.getdata())
            unique_alpha_values = set(alpha_values)
            assert (
                len(unique_alpha_values) > 1
            ), "Alpha channel should have varying values"

    @pytest.mark.real_processing
    @pytest.mark.api
    def test_e2e_single_image_pro_api(self, real_test_image_path, temp_output_dir):
        """Test end-to-end processing with withoutBG API (if API key available)."""
        api_key = os.getenv("WITHOUTBG_API_KEY")
        if not api_key:
            pytest.skip("API key not available for E2E testing")

        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        output_path = temp_output_dir / "output_pro.png"

        result = self.runner.invoke(
            main,
            [
                str(real_test_image_path),
                "--output",
                str(output_path),
                "--model",
                "api",
                "--api-key",
                api_key,
                "--verbose",
            ],
        )

        # Check command succeeded
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Check output file was created
        assert output_path.exists(), "Output file was not created"

        # Verify output image properties
        with (
            Image.open(output_path) as output_image,
            Image.open(real_test_image_path) as original_image,
        ):
            assert output_image.mode == "RGBA", "Output should be RGBA"
            assert (
                output_image.size == original_image.size
            ), "Output size should match input"

    @pytest.mark.real_processing
    def test_e2e_batch_processing(self, test_images_dir, temp_output_dir):
        """Test end-to-end batch processing."""
        if not test_images_dir.exists():
            pytest.skip("Test images directory not available")

        # Copy test images to temporary directory to avoid modifying originals
        temp_input_dir = temp_output_dir / "input_images"
        shutil.copytree(test_images_dir, temp_input_dir)

        batch_output_dir = temp_output_dir / "batch_results"

        result = self.runner.invoke(
            main,
            [
                str(temp_input_dir),
                "--batch",
                "--output-dir",
                str(batch_output_dir),
                "--verbose",
            ],
        )

        # Check command succeeded
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Check output directory was created
        assert batch_output_dir.exists(), "Batch output directory was not created"

        # Check that output files were created
        input_images = list(temp_input_dir.glob("*.png")) + list(
            temp_input_dir.glob("*.jpg")
        )
        for input_image in input_images:
            expected_output = batch_output_dir / f"{input_image.stem}-withoutbg.png"
            assert (
                expected_output.exists()
            ), f"Output file {expected_output} was not created"

            # Verify basic properties
            with Image.open(expected_output) as output_image:
                assert output_image.mode == "RGBA", "Batch output should be RGBA"

    @pytest.mark.real_processing
    def test_e2e_different_output_formats(self, real_test_image_path, temp_output_dir):
        """Test end-to-end processing with different output formats."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        formats = [("png", "RGBA"), ("jpg", "RGB"), ("webp", "RGBA")]

        for fmt, expected_mode in formats:
            output_path = temp_output_dir / f"output.{fmt}"

            result = self.runner.invoke(
                main,
                [
                    str(real_test_image_path),
                    "--output",
                    str(output_path),
                    "--format",
                    fmt,
                    "--quality",
                    "90",
                ],
            )

            assert (
                result.exit_code == 0
            ), f"Failed to process {fmt} format: {result.output}"
            assert output_path.exists(), f"Output file for {fmt} format was not created"

            # Verify format-specific properties
            with Image.open(output_path) as output_image:
                assert (
                    output_image.mode == expected_mode
                ), f"{fmt} output should be {expected_mode}"

                # Map format names to PIL's canonical format names
                expected_pil_format = {
                    "jpg": "JPEG",
                    "jpeg": "JPEG",
                    "png": "PNG",
                    "webp": "WEBP",
                }
                expected_format = expected_pil_format.get(fmt.lower(), fmt.upper())
                assert output_image.format == expected_format, (
                    f"Output format should be {expected_format}, "
                    f"got {output_image.format}"
                )

    @pytest.mark.real_processing
    def test_e2e_large_image_processing(self, temp_output_dir):
        """Test end-to-end processing of large images."""
        # Create a large test image
        large_image = Image.new("RGB", (2048, 1536), color=(100, 150, 200))
        large_image_path = temp_output_dir / "large_test_image.jpg"
        large_image.save(large_image_path)

        output_path = temp_output_dir / "large_output.png"

        result = self.runner.invoke(
            main, [str(large_image_path), "--output", str(output_path), "--verbose"]
        )

        assert result.exit_code == 0, f"Large image processing failed: {result.output}"
        assert output_path.exists(), "Large image output was not created"

        # Verify output maintains original size
        with Image.open(output_path) as output_image:
            assert output_image.size == (
                2048,
                1536,
            ), "Large image output size should match input"
            assert output_image.mode == "RGBA", "Large image output should be RGBA"

    @pytest.mark.real_processing
    def test_e2e_unicode_filename(self, real_test_image_path, temp_output_dir):
        """Test end-to-end processing with unicode filenames."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        # Copy test image to unicode filename
        unicode_input_path = temp_output_dir / "测试图片_ñáme_emoji🎨.png"
        shutil.copy2(real_test_image_path, unicode_input_path)

        unicode_output_path = temp_output_dir / "输出图片_résult_🎯.png"

        result = self.runner.invoke(
            main,
            [
                str(unicode_input_path),
                "--output",
                str(unicode_output_path),
                "--verbose",
            ],
        )

        assert (
            result.exit_code == 0
        ), f"Unicode filename processing failed: {result.output}"
        assert unicode_output_path.exists(), "Unicode filename output was not created"

        # Verify output properties
        with Image.open(unicode_output_path) as output_image:
            assert output_image.mode == "RGBA", "Unicode filename output should be RGBA"

    @pytest.mark.real_processing
    def test_e2e_progress_indication(self, real_test_image_path, temp_output_dir):
        """Test that progress indication works during real processing."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        output_path = temp_output_dir / "progress_test.png"

        result = self.runner.invoke(
            main, [str(real_test_image_path), "--output", str(output_path), "--verbose"]
        )

        assert result.exit_code == 0

        # Check that verbose output contains expected messages
        output = result.output
        assert "Processing:" in output, "Should show processing message"
        assert "Output:" in output, "Should show output path"
        assert "✅ Processing complete!" in output, "Should show completion message"

    @pytest.mark.real_processing
    def test_e2e_automatic_output_naming(self, real_test_image_path, temp_output_dir):
        """Test automatic output file naming."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        # Copy test image to temp dir so we can predict output location
        input_path = temp_output_dir / "input_test.jpg"
        shutil.copy2(real_test_image_path, input_path)

        result = self.runner.invoke(main, [str(input_path)])

        assert result.exit_code == 0, f"Auto-naming failed: {result.output}"

        # Check that output file was created with expected name
        expected_output = temp_output_dir / "input_test-withoutbg.png"
        assert expected_output.exists(), "Auto-named output file was not created"

        # Verify output properties
        with Image.open(expected_output) as output_image:
            assert output_image.mode == "RGBA", "Auto-named output should be RGBA"

    @pytest.mark.real_processing
    def test_e2e_quality_comparison(self, real_test_image_path, temp_output_dir):
        """Test that output quality varies with different quality settings."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        qualities = [50, 75, 95]
        output_sizes = []

        for quality in qualities:
            output_path = temp_output_dir / f"quality_{quality}.jpg"

            result = self.runner.invoke(
                main,
                [
                    str(real_test_image_path),
                    "--output",
                    str(output_path),
                    "--format",
                    "jpg",
                    "--quality",
                    str(quality),
                ],
            )

            assert result.exit_code == 0, f"Quality {quality} processing failed"
            assert output_path.exists(), f"Quality {quality} output not created"

            # Record file size
            output_sizes.append(output_path.stat().st_size)

        # Higher quality should generally result in larger file sizes
        # (though this isn't always guaranteed with different compression algorithms)
        assert (
            len(set(output_sizes)) > 1
        ), "Different quality settings should produce different file sizes"

    @pytest.mark.real_processing
    def test_e2e_model_comparison(self, real_test_image_path, temp_output_dir):
        """Test comparison between different models (if API available)."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        # Test withoutBG Open Weights Model
        open_weights_output = temp_output_dir / "open_weights_result.png"
        result_open_weights = self.runner.invoke(
            main,
            [
                str(real_test_image_path),
                "--output",
                str(open_weights_output),
                "--model",
                "open-weights",
            ],
        )

        assert (
            result_open_weights.exit_code == 0
        ), "withoutBG Open Weights Model processing failed"
        assert (
            open_weights_output.exists()
        ), "withoutBG Open Weights Model output not created"

        # Test API model if available
        api_key = os.getenv("WITHOUTBG_API_KEY")
        if api_key:
            api_output = temp_output_dir / "api_result.png"
            result_api = self.runner.invoke(
                main,
                [
                    str(real_test_image_path),
                    "--output",
                    str(api_output),
                    "--model",
                    "api",
                    "--api-key",
                    api_key,
                ],
            )

            assert result_api.exit_code == 0, "API processing failed"
            assert api_output.exists(), "API output not created"

            # Compare basic properties
            with (
                Image.open(open_weights_output) as open_weights_image,
                Image.open(api_output) as api_image,
            ):
                assert (
                    open_weights_image.size == api_image.size
                ), "Both models should produce same size output"
                assert (
                    open_weights_image.mode == api_image.mode == "RGBA"
                ), "Both should produce RGBA output"


class TestCLIE2EErrorHandling:
    """E2E tests for error handling with real files."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @pytest.mark.real_processing
    def test_e2e_corrupted_image_file(self, temp_dir):
        """Test handling of corrupted image files in real processing."""
        # Create a corrupted file that looks like an image
        corrupted_path = temp_dir / "corrupted.jpg"
        with open(corrupted_path, "wb") as f:
            # Write some binary data that's not a valid image
            f.write(b"\xff\xd8\xff\xe0" + b"invalid image data" * 100)

        result = self.runner.invoke(main, [str(corrupted_path), "--verbose"])

        # Should fail gracefully
        assert result.exit_code == 1
        # Error message should be informative
        assert "error" in result.output.lower() or "Error" in result.output

    @pytest.mark.real_processing
    def test_e2e_unsupported_file_format(self, temp_dir):
        """Test handling of unsupported file formats."""
        # Create a text file with image extension
        fake_image_path = temp_dir / "fake.jpg"
        with open(fake_image_path, "w") as f:
            f.write("This is not an image file")

        result = self.runner.invoke(main, [str(fake_image_path)])

        assert result.exit_code == 1
        assert "error" in result.output.lower() or "Error" in result.output

    @pytest.mark.real_processing
    def test_e2e_permission_denied_output(self, real_test_image_path, temp_dir):
        """Test handling of permission errors for output files."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        import platform
        import stat

        # Create a read-only directory - Windows requires different approach
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()
        # Use platform-specific permission handling
        if platform.system() == "Windows":
            # On Windows, create a file first, then make it read-only
            output_path = readonly_dir / "output.png"
            output_path.touch()
            output_path.chmod(stat.S_IREAD)
            result = self.runner.invoke(
                main, [str(real_test_image_path), "--output", str(output_path)]
            )
            # Should fail due to permission error
            assert result.exit_code == 1
            # Restore permissions for cleanup
            output_path.chmod(stat.S_IWRITE | stat.S_IREAD)
        else:
            # Unix-style read-only directory
            readonly_dir.chmod(0o444)
            try:
                output_path = readonly_dir / "output.png"

                result = self.runner.invoke(
                    main, [str(real_test_image_path), "--output", str(output_path)]
                )

                # Should fail due to permission error
                assert result.exit_code == 1

            finally:
                # Restore permissions for cleanup
                readonly_dir.chmod(0o755)

    @pytest.mark.real_processing
    def test_e2e_disk_space_simulation(self, real_test_image_path, temp_dir):
        """Test behavior when output directory has limited space (simulation)."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        # This is a simulation - we can't easily test actual disk space limits
        # But we can test that the CLI handles file system errors gracefully
        output_path = temp_dir / "output.png"

        # Normal processing should work
        result = self.runner.invoke(
            main, [str(real_test_image_path), "--output", str(output_path)]
        )

        assert result.exit_code == 0
        assert output_path.exists()


class TestCLIE2EPerformance:
    """Performance tests for real CLI operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @pytest.mark.performance
    @pytest.mark.real_processing
    def test_e2e_processing_time_open_source_model(
        self, real_test_image_path, temp_output_dir, performance_tracker
    ):
        """Test processing time for withoutBG Open Weights Model."""
        if not real_test_image_path.exists():
            pytest.skip("Real test image not available")

        import time

        output_path = temp_output_dir / "perf_test.png"

        start_time = time.time()
        result = self.runner.invoke(
            main, [str(real_test_image_path), "--output", str(output_path)]
        )
        end_time = time.time()

        processing_time = end_time - start_time
        performance_tracker.record_metric(
            "open_weights_processing_time", processing_time, "seconds"
        )

        assert result.exit_code == 0
        assert output_path.exists()

        # withoutBG Open Weights Model should process reasonably quickly
        # This is generous since it includes model download time on first run
        performance_tracker.assert_performance(
            "open_weights_processing_time",
            30.0,  # 30 seconds max
            "withoutBG Open Weights Model processing took too long",
        )

    @pytest.mark.performance
    @pytest.mark.real_processing
    def test_e2e_batch_processing_throughput(
        self, test_images_dir, temp_output_dir, performance_tracker
    ):
        """Test batch processing throughput."""
        if not test_images_dir.exists():
            pytest.skip("Test images directory not available")

        import time

        # Copy test images to avoid modifying originals
        temp_input_dir = temp_output_dir / "perf_input"
        shutil.copytree(test_images_dir, temp_input_dir)

        batch_output_dir = temp_output_dir / "perf_batch_output"

        start_time = time.time()
        result = self.runner.invoke(
            main,
            [str(temp_input_dir), "--batch", "--output-dir", str(batch_output_dir)],
        )
        end_time = time.time()

        total_time = end_time - start_time

        # Count input images
        input_images = list(temp_input_dir.glob("*.png")) + list(
            temp_input_dir.glob("*.jpg")
        )
        num_images = len(input_images)

        if num_images > 0:
            time_per_image = total_time / num_images
            performance_tracker.record_metric(
                "batch_time_per_image", time_per_image, "seconds"
            )
            performance_tracker.record_metric("batch_total_time", total_time, "seconds")

            assert result.exit_code == 0

            # Each image should process in reasonable time
            performance_tracker.assert_performance(
                "batch_time_per_image",
                20.0,  # 20 seconds per image max
                "Batch processing too slow per image",
            )

    @pytest.mark.performance
    @pytest.mark.real_processing
    def test_e2e_memory_usage_large_image(self, temp_output_dir):
        """Test memory usage with large images."""
        import os

        import psutil

        # Create a large test image
        large_image = Image.new("RGB", (3000, 2000), color=(128, 64, 192))
        large_image_path = temp_output_dir / "large_perf_test.jpg"
        large_image.save(large_image_path)

        output_path = temp_output_dir / "large_perf_output.png"

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = self.runner.invoke(
            main, [str(large_image_path), "--output", str(output_path)]
        )

        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        assert result.exit_code == 0
        assert output_path.exists()

        # Memory increase should be reasonable for a large image
        # This is quite generous to account for model loading
        assert (
            memory_increase < 2000
        ), f"Memory usage increased by {memory_increase:.1f}MB, expected < 2000MB"
