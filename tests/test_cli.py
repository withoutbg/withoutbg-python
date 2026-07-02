"""Comprehensive tests for the CLI module."""

import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from PIL import Image

from src.withoutbg.cli import main
from src.withoutbg.exceptions import WithoutBGError


class TestCLIUnit:
    """Unit tests for CLI command parsing and validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_version_option(self):
        """Test --version option displays version correctly."""
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_help_option(self):
        """Test --help option displays help text."""
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Remove the background" in result.output
        assert "Examples:" in result.output

    def test_help_contains_all_options(self):
        """Test help text contains all available options."""
        result = self.runner.invoke(main, ["--help"])
        help_text = result.output

        expected_options = [
            "--output",
            "-o",
            "--api-key",
            "--use-api",
            "--model",
            "--batch",
            "--output-dir",
            "--format",
            "--quality",
            "--verbose",
            "-v",
        ]

        for option in expected_options:
            assert option in help_text

    def test_missing_input_file_error(self):
        """Test error when input file is missing."""
        result = self.runner.invoke(main, ["nonexistent.jpg"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_model_choice_validation(self):
        """Test model option accepts only valid choices."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            # Valid choice
            result = self.runner.invoke(
                main, [temp_file.name, "--model", "open-weights"]
            )
            # Should fail because of processing, not validation
            assert "Invalid value" not in result.output

            # Invalid choice
            result = self.runner.invoke(main, [temp_file.name, "--model", "invalid"])
            assert result.exit_code != 0
            assert "Invalid value" in result.output

    def test_format_choice_validation(self):
        """Test format option accepts only valid choices."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            # Valid choice
            result = self.runner.invoke(main, [temp_file.name, "--format", "png"])
            # Should fail because of processing, not validation
            assert "Invalid value" not in result.output

            # Invalid choice
            result = self.runner.invoke(main, [temp_file.name, "--format", "invalid"])
            assert result.exit_code != 0
            assert "Invalid value" in result.output

    def test_quality_validation(self):
        """Test quality option accepts integers."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            # Valid integer
            result = self.runner.invoke(main, [temp_file.name, "--quality", "80"])
            # Should fail because of processing, not validation
            assert "Invalid value" not in result.output

            # Invalid type
            result = self.runner.invoke(
                main, [temp_file.name, "--quality", "not_a_number"]
            )
            assert result.exit_code != 0
            assert "Invalid value" in result.output

    def test_verbose_flag_parsing(self):
        """Test verbose flag is parsed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            result = self.runner.invoke(main, [temp_file.name, "--verbose"])
            # Should parse without validation error
            assert "Invalid value" not in result.output

    def test_batch_flag_parsing(self):
        """Test batch flag is parsed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            result = self.runner.invoke(main, [temp_file.name, "--batch"])
            # Should parse without validation error
            assert "Invalid value" not in result.output

    def test_use_api_flag_parsing(self):
        """Test use-api flag is parsed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            result = self.runner.invoke(main, [temp_file.name, "--use-api"])
            # Should fail because no API key provided
            assert "API key required" in result.output

    def test_api_key_from_env_var(self):
        """Test API key can be provided via environment variable."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            env = {"WITHOUTBG_API_KEY": "test_key"}
            result = self.runner.invoke(main, [temp_file.name, "--use-api"], env=env)
            # Should not complain about missing API key
            assert "API key required" not in result.output

    def test_api_key_option_override(self):
        """Test --api-key option overrides environment variable."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            env = {"WITHOUTBG_API_KEY": "env_key"}
            result = self.runner.invoke(
                main, [temp_file.name, "--api-key", "option_key"], env=env
            )
            # Should not complain about missing API key
            assert "API key required" not in result.output


class TestCLIIntegration:
    """Integration tests with mocked backends."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @pytest.fixture
    def test_image_file(self, temp_dir):
        """Create a test image file."""
        image = Image.new("RGB", (256, 256), color=(255, 0, 0))
        image_path = temp_dir / "test_image.jpg"
        image.save(image_path)
        return image_path

    @pytest.fixture
    def test_images_directory(self, temp_dir):
        """Create directory with multiple test images."""
        images_dir = temp_dir / "images"
        images_dir.mkdir()

        # Create various test images
        formats = [
            ("image1.jpg", "RGB"),
            ("image2.png", "RGBA"),
            ("image3.webp", "RGB"),
        ]
        for filename, mode in formats:
            image = Image.new(mode, (200, 200), color=(255, 0, 0))
            image.save(images_dir / filename)

        return images_dir

    @patch("src.withoutbg.cli.WithoutBG")
    def test_single_image_processing_open_source_model(
        self, mock_withoutbg_class, test_image_file
    ):
        """Test processing single image with local withoutBG Open Weights Model."""
        # Mock successful processing
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = result_image

        result = self.runner.invoke(main, [str(test_image_file)])

        assert result.exit_code == 0
        mock_withoutbg_class.open_weights.assert_called_once()
        mock_instance.remove_background.assert_called_once()
        call_args = mock_instance.remove_background.call_args
        assert call_args[0][0] == test_image_file
        assert "progress_callback" in call_args[1]

    @patch("src.withoutbg.cli.WithoutBG")
    def test_single_image_processing_with_output_path(
        self, mock_withoutbg_class, test_image_file, temp_dir
    ):
        """Test processing with custom output path."""
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = result_image

        output_path = temp_dir / "custom_output.png"
        result = self.runner.invoke(
            main, [str(test_image_file), "--output", str(output_path)]
        )

        assert result.exit_code == 0
        assert output_path.exists()

    @patch("src.withoutbg.cli.WithoutBG")
    def test_single_image_processing_pro_api(
        self, mock_withoutbg_class, test_image_file
    ):
        """Test processing with withoutBG API model."""
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.api.return_value
        mock_instance.remove_background.return_value = result_image

        result = self.runner.invoke(
            main, [str(test_image_file), "--model", "api", "--api-key", "test_key"]
        )

        assert result.exit_code == 0
        mock_withoutbg_class.api.assert_called_once_with("test_key")
        mock_instance.remove_background.assert_called_once()
        call_args = mock_instance.remove_background.call_args
        assert call_args[0][0] == test_image_file
        assert "progress_callback" in call_args[1]

    @patch("src.withoutbg.cli.WithoutBG")
    def test_use_api_flag_forces_pro_model(self, mock_withoutbg_class, test_image_file):
        """Test that --use-api flag forces withoutBG API model."""
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.api.return_value
        mock_instance.remove_background.return_value = result_image

        result = self.runner.invoke(
            main, [str(test_image_file), "--use-api", "--api-key", "test_key"]
        )

        assert result.exit_code == 0
        mock_withoutbg_class.api.assert_called_once_with("test_key")
        mock_instance.remove_background.assert_called_once()
        call_args = mock_instance.remove_background.call_args
        assert call_args[0][0] == test_image_file
        assert "progress_callback" in call_args[1]

    @patch("src.withoutbg.cli.WithoutBG")
    def test_batch_processing_directory(
        self, mock_withoutbg_class, test_images_directory, temp_dir
    ):
        """Test batch processing of directory."""
        result_image = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = result_image

        output_dir = temp_dir / "output"
        result = self.runner.invoke(
            main,
            [str(test_images_directory), "--batch", "--output-dir", str(output_dir)],
        )

        assert result.exit_code == 0
        assert mock_instance.remove_background.call_count == 3  # 3 images in directory
        assert output_dir.exists()

    @patch("src.withoutbg.cli.WithoutBG")
    def test_batch_processing_creates_default_output_dir(
        self, mock_withoutbg_class, test_images_directory
    ):
        """Test batch processing creates default output directory."""
        result_image = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = result_image

        result = self.runner.invoke(main, [str(test_images_directory), "--batch"])

        assert result.exit_code == 0
        default_output_dir = test_images_directory / "withoutbg-results"
        assert default_output_dir.exists()

    @patch("src.withoutbg.cli.WithoutBG")
    def test_output_format_options(
        self, mock_withoutbg_class, test_image_file, temp_dir
    ):
        """Test different output format options."""
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = result_image

        formats = ["png", "jpg", "webp"]
        for fmt in formats:
            output_path = temp_dir / f"output.{fmt}"
            result = self.runner.invoke(
                main,
                [str(test_image_file), "--output", str(output_path), "--format", fmt],
            )

            assert (
                result.exit_code == 0
            ), f"Format {fmt} failed with output: {result.output}"
            assert output_path.exists()

            # Verify the file was saved with correct PIL format
            with Image.open(output_path) as saved_image:
                expected_pil_formats = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}
                expected_format = expected_pil_formats[fmt]
                assert (
                    saved_image.format == expected_format
                ), f"Expected {expected_format}, got {saved_image.format}"

    @patch("src.withoutbg.cli.WithoutBG")
    def test_jpeg_quality_setting(
        self, mock_withoutbg_class, test_image_file, temp_dir
    ):
        """Test JPEG quality setting."""
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = result_image

        output_path = temp_dir / "output.jpg"
        result = self.runner.invoke(
            main,
            [
                str(test_image_file),
                "--output",
                str(output_path),
                "--format",
                "jpg",
                "--quality",
                "80",
            ],
        )

        assert (
            result.exit_code == 0
        ), f"JPEG quality test failed with output: {result.output}"
        assert output_path.exists()

    @patch("src.withoutbg.cli.WithoutBG")
    def test_verbose_output(self, mock_withoutbg_class, test_image_file):
        """Test verbose output mode."""
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = result_image

        result = self.runner.invoke(main, [str(test_image_file), "--verbose"])

        assert result.exit_code == 0
        assert "Using withoutBG Open Weights Model for processing" in result.output
        assert "Processing:" in result.output
        assert "✅ Processing complete!" in result.output

    @patch("src.withoutbg.cli.WithoutBG")
    def test_verbose_output_with_api(self, mock_withoutbg_class, test_image_file):
        """Test verbose output with API processing."""
        result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        mock_instance = mock_withoutbg_class.api.return_value
        mock_instance.remove_background.return_value = result_image

        result = self.runner.invoke(
            main,
            [str(test_image_file), "--use-api", "--api-key", "test_key", "--verbose"],
        )

        assert result.exit_code == 0
        assert "Using withoutBG API for processing" in result.output

    def test_output_filename_generation(self, test_image_file, temp_dir):
        """Test automatic output filename generation."""
        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            result_image = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.return_value = result_image

            result = self.runner.invoke(main, [str(test_image_file)])

            assert result.exit_code == 0
            # Should create file with -withoutbg suffix
            expected_output = (
                test_image_file.parent / f"{test_image_file.stem}-withoutbg.png"
            )
            assert expected_output.exists()


class TestCLIErrorHandling:
    """Test error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_api_key_required_error(self, temp_dir):
        """Test error when API key is required but not provided."""
        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        result = self.runner.invoke(main, [str(image_path), "--use-api"])

        assert result.exit_code == 1
        assert "API key required" in result.output

    def test_withoutbg_error_handling(self, temp_dir):
        """Test handling of WithoutBGError exceptions."""
        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.side_effect = WithoutBGError("Test error")

            result = self.runner.invoke(main, [str(image_path)])

            assert result.exit_code == 1
            assert "Error: Test error" in result.output

    def test_keyboard_interrupt_handling(self, temp_dir):
        """Test handling of KeyboardInterrupt (Ctrl+C)."""
        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.side_effect = KeyboardInterrupt()

            result = self.runner.invoke(main, [str(image_path)])

            assert result.exit_code == 1
            assert "❌ Processing cancelled" in result.output

    def test_unexpected_error_handling(self, temp_dir):
        """Test handling of unexpected exceptions."""
        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.side_effect = ValueError("Unexpected error")

            result = self.runner.invoke(main, [str(image_path)])

            assert result.exit_code == 1
            assert "Unexpected error: Unexpected error" in result.output

    def test_batch_processing_no_images_error(self, temp_dir):
        """Test error when no images found in directory for batch processing."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = self.runner.invoke(main, [str(empty_dir), "--batch"])

        assert result.exit_code == 1
        assert "No image files found" in result.output

    @patch("src.withoutbg.cli.WithoutBG")
    def test_batch_processing_partial_failures(self, mock_withoutbg_class, temp_dir):
        """Test batch processing continues on individual failures."""
        # Create test images
        images_dir = temp_dir / "images"
        images_dir.mkdir()

        for i in range(3):
            image = Image.new("RGB", (100, 100))
            image.save(images_dir / f"image{i}.jpg")

        # Mock failure for second image
        def side_effect(path, **kwargs):
            if "image1" in str(path):
                raise ValueError("Processing failed")
            return Image.new("RGBA", (100, 100))

        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.side_effect = side_effect

        result = self.runner.invoke(main, [str(images_dir), "--batch", "--verbose"])

        assert result.exit_code == 0  # Should continue processing
        assert "❌ Failed to process" in result.output

    def test_permission_error_handling(self, temp_dir):
        """Test handling of permission errors when writing output."""
        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.return_value = Image.new("RGBA", (100, 100))

            # Mock save to raise permission error
            with patch.object(
                Image.Image, "save", side_effect=PermissionError("Permission denied")
            ):
                result = self.runner.invoke(main, [str(image_path)])

                assert result.exit_code == 1
                assert "Unexpected error" in result.output


class TestCLIEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_unicode_filename_handling(self, temp_dir):
        """Test handling of unicode filenames."""
        # Create image with unicode filename
        unicode_filename = "测试图片_ñáme.jpg"
        image_path = temp_dir / unicode_filename
        Image.new("RGB", (100, 100)).save(image_path)

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.return_value = Image.new("RGBA", (100, 100))

            result = self.runner.invoke(main, [str(image_path)])

            assert result.exit_code == 0

    @patch("src.withoutbg.cli.WithoutBG")
    def test_large_image_processing(self, mock_withoutbg_class, temp_dir):
        """Test processing of large images."""
        # Create large test image
        large_image = Image.new("RGB", (4000, 3000), color=(255, 0, 0))
        image_path = temp_dir / "large_image.jpg"
        large_image.save(image_path)

        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = Image.new("RGBA", (4000, 3000))

        result = self.runner.invoke(main, [str(image_path)])

        assert result.exit_code == 0

    def test_corrupted_image_handling(self, temp_dir):
        """Test handling of corrupted image files."""
        # Create a file that looks like an image but is corrupted
        corrupted_path = temp_dir / "corrupted.jpg"
        with open(corrupted_path, "wb") as f:
            f.write(b"This is not a valid image file")

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.side_effect = Exception(
                "Cannot identify image file"
            )

            result = self.runner.invoke(main, [str(corrupted_path)])

            assert result.exit_code == 1

    @patch("src.withoutbg.cli.WithoutBG")
    def test_various_image_formats(self, mock_withoutbg_class, temp_dir):
        """Test processing of various input image formats."""
        formats = [("test.jpg", "RGB"), ("test.png", "RGBA"), ("test.bmp", "RGB")]

        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = Image.new("RGBA", (100, 100))

        for filename, mode in formats:
            image = Image.new(mode, (100, 100))
            image_path = temp_dir / filename
            image.save(image_path)

            result = self.runner.invoke(main, [str(image_path)])
            assert result.exit_code == 0

    def test_single_file_with_batch_flag(self, temp_dir):
        """Test using --batch flag with single file."""
        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.return_value = Image.new("RGBA", (100, 100))

            result = self.runner.invoke(main, [str(image_path), "--batch"])

            assert result.exit_code == 0

    @patch("src.withoutbg.cli.WithoutBG")
    def test_rgba_to_jpg_conversion(self, mock_withoutbg_class, temp_dir):
        """Test conversion from RGBA to JPG format."""
        image_path = temp_dir / "test.png"
        Image.new("RGBA", (100, 100), color=(255, 0, 0, 128)).save(image_path)

        # Mock returns RGBA image
        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = Image.new(
            "RGBA", (100, 100), color=(255, 0, 0, 128)
        )

        output_path = temp_dir / "output.jpg"
        result = self.runner.invoke(
            main, [str(image_path), "--output", str(output_path), "--format", "jpg"]
        )

        assert (
            result.exit_code == 0
        ), f"RGBA to JPG conversion failed with output: {result.output}"
        assert output_path.exists()

        # Verify the saved image is RGB (not RGBA)
        with Image.open(output_path) as saved_image:
            assert saved_image.mode == "RGB"

    def test_directory_as_input_without_batch_flag(self, temp_dir):
        """Test providing directory as input without --batch flag."""
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        Image.new("RGB", (100, 100)).save(images_dir / "test.jpg")

        with patch("src.withoutbg.cli.WithoutBG") as mock_withoutbg_class:
            mock_instance = mock_withoutbg_class.open_weights.return_value
            mock_instance.remove_background.return_value = Image.new("RGBA", (100, 100))

            result = self.runner.invoke(main, [str(images_dir)])

            # Should process as batch automatically
            assert result.exit_code == 0

    def test_empty_api_key_environment_variable(self, temp_dir):
        """Test behavior with empty API key environment variable."""
        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        env = {"WITHOUTBG_API_KEY": ""}
        result = self.runner.invoke(main, [str(image_path), "--use-api"], env=env)

        assert result.exit_code == 1
        assert "API key required" in result.output


class TestCLITestUtilities:
    """Test helper functions and utilities for CLI testing."""

    def test_create_test_image_file(self, temp_dir):
        """Test utility function for creating test image files."""

        def create_test_image(path, size=(100, 100), color=(255, 0, 0), mode="RGB"):
            """Create a test image file."""
            image = Image.new(mode, size, color)
            image.save(path)
            return path

        image_path = create_test_image(temp_dir / "test.jpg")
        assert image_path.exists()

        # Verify image properties
        with Image.open(image_path) as image:
            assert image.size == (100, 100)
            assert image.mode == "RGB"

    def test_create_corrupted_file(self, temp_dir):
        """Test utility for creating corrupted image files."""

        def create_corrupted_image_file(path, content=b"corrupted"):
            """Create a corrupted image file for testing error handling."""
            with open(path, "wb") as f:
                f.write(content)
            return path

        corrupted_path = create_corrupted_image_file(temp_dir / "corrupted.jpg")
        assert corrupted_path.exists()

        # Verify it can't be opened as valid image
        with pytest.raises((OSError, IOError)):
            Image.open(corrupted_path)

    def test_verify_output_file_properties(self, temp_dir):
        """Test utility for verifying output file properties."""

        def verify_output_properties(
            output_path, expected_format=None, expected_mode=None, expected_size=None
        ):
            """Verify properties of output image file."""
            assert output_path.exists(), f"Output file {output_path} does not exist"

            with Image.open(output_path) as image:
                if expected_format:
                    assert image.format.lower() == expected_format.lower()
                if expected_mode:
                    assert image.mode == expected_mode
                if expected_size:
                    assert image.size == expected_size

                return image.size  # Return size instead of image object

        # Create test output file
        test_image = Image.new("RGBA", (200, 150), color=(255, 0, 0, 128))
        output_path = temp_dir / "output.png"
        test_image.save(output_path)

        # Test verification
        verified_size = verify_output_properties(
            output_path,
            expected_format="PNG",
            expected_mode="RGBA",
            expected_size=(200, 150),
        )
        assert verified_size == (200, 150)


class TestCLIPerformance:
    """Performance-related tests for CLI operations."""

    @pytest.mark.performance
    @patch("src.withoutbg.cli.WithoutBG")
    def test_batch_processing_performance(self, mock_withoutbg_class, temp_dir):
        """Test performance of batch processing multiple images."""
        import time

        # Create multiple test images
        images_dir = temp_dir / "performance_test"
        images_dir.mkdir()

        num_images = 10
        for i in range(num_images):
            image = Image.new("RGB", (512, 512))
            image.save(images_dir / f"image_{i:03d}.jpg")

        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = Image.new("RGBA", (512, 512))

        runner = CliRunner()
        start_time = time.time()

        result = runner.invoke(main, [str(images_dir), "--batch"])

        end_time = time.time()
        processing_time = end_time - start_time

        assert result.exit_code == 0
        assert mock_instance.remove_background.call_count == num_images

        # Performance assertion (should process 10 images in reasonable time)
        # This is quite lenient since we're mocking the actual processing
        assert (
            processing_time < 10.0
        ), f"Batch processing took {processing_time:.2f}s, expected < 10s"

    @pytest.mark.performance
    @patch("src.withoutbg.cli.WithoutBG")
    def test_cli_startup_performance(self, mock_withoutbg_class, temp_dir):
        """Test CLI startup and initialization performance."""
        import time

        image_path = temp_dir / "test.jpg"
        Image.new("RGB", (100, 100)).save(image_path)

        mock_instance = mock_withoutbg_class.open_weights.return_value
        mock_instance.remove_background.return_value = Image.new("RGBA", (100, 100))

        runner = CliRunner()
        start_time = time.time()

        result = runner.invoke(main, [str(image_path)])

        end_time = time.time()
        total_time = end_time - start_time

        assert result.exit_code == 0
        # CLI should start and process quickly with mocked backend
        assert total_time < 5.0, f"CLI execution took {total_time:.2f}s, expected < 5s"
