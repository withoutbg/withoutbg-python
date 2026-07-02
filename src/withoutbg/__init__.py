"""withoutbg: Python SDK for local and cloud background removal.

Quick start::

    from withoutbg import WithoutBG

    # Local — free, private, works offline
    model = WithoutBG.open_weights()
    result = model.remove_background("photo.jpg")  # returns PIL Image (RGBA)
    result.save("output.png")

    # Cloud — better quality, no GPU needed
    model = WithoutBG.api(api_key="sk_...")
    result = model.remove_background("photo.jpg")
    result.save("output.png")

See https://withoutbg.com/documentation/integrations/python-sdk for full docs.
"""

from .__version__ import __version__
from .api import ProAPI, WithoutBGAPIClient
from .core import WithoutBG
from .exceptions import APIError, ConfigurationError, ModelNotFoundError, WithoutBGError
from .models import OpenSourceModel, OpenWeightsModel

__all__ = [
    "WithoutBG",
    "OpenWeightsModel",
    "WithoutBGAPIClient",
    "WithoutBGError",
    "ModelNotFoundError",
    "APIError",
    "ConfigurationError",
    "__version__",
    # Deprecated aliases — will be removed in a future release
    "OpenSourceModel",
    "ProAPI",
]
