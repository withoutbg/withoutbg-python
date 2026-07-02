"""Custom exceptions for withoutbg package."""


class WithoutBGError(Exception):
    """Base exception for withoutbg package."""

    pass


class ModelNotFoundError(WithoutBGError):
    """Raised when a model file cannot be found or loaded."""

    pass


class APIError(WithoutBGError):
    """Raised when API requests fail."""

    pass


class InvalidImageError(WithoutBGError):
    """Raised when image input is invalid or corrupted."""

    pass


class ConfigurationError(WithoutBGError):
    """Raised when configuration is invalid."""

    pass
