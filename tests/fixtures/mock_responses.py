"""Mock API responses for testing."""

import base64
import io
import json

from PIL import Image


def get_mock_api_responses():
    """Get dictionary of mock API responses for testing."""

    # Create a sample result image for successful responses
    result_image = Image.new("RGBA", (256, 256), color=(100, 150, 200, 128))
    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    result_image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "success": {
            "status_code": 200,
            "json": {
                "image": result_image_b64,
                "processing_time": 1.23,
                "model_version": "pro-v2.1",
            },
        },
        "success_alpha_only": {
            "status_code": 200,
            "json": {
                "alpha_base64": base64.b64encode(b"alpha_channel_data").decode("utf-8"),
                "processing_time": 0.89,
                "model_version": "pro-v2.1",
            },
        },
        "unauthorized": {
            "status_code": 401,
            "json": {"error": "Invalid API key", "code": "INVALID_API_KEY"},
        },
        "insufficient_credits": {
            "status_code": 402,
            "json": {
                "error": "Insufficient credits",
                "code": "INSUFFICIENT_CREDITS",
                "credits_remaining": 0,
            },
        },
        "forbidden": {
            "status_code": 403,
            "json": {
                "error": "API key expired",
                "code": "API_KEY_EXPIRED",
                "expired_at": "2024-01-01T00:00:00Z",
            },
        },
        "rate_limited": {
            "status_code": 429,
            "json": {
                "error": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED",
                "retry_after": 60,
                "requests_per_minute": 7,
            },
        },
        "server_error": {
            "status_code": 500,
            "json": {
                "error": "Internal server error",
                "code": "INTERNAL_ERROR",
                "request_id": "req_123456789",
            },
        },
        "usage_info": {
            "status_code": 200,
            "json": {
                "credits_used": 150,
                "credits_total": 1000,
                "credits_remaining": 850,
                "plan": "professional",
                "billing_cycle": "monthly",
                "reset_date": "2024-02-01T00:00:00Z",
            },
        },
        "models_info": {
            "status_code": 200,
            "json": {
                "models": ["open_weights", "api", "premium"],
                "default": "api",
                "features": {
                    "open_weights": ["basic_removal"],
                    "api": ["advanced_removal", "edge_refinement"],
                    "premium": ["ultra_quality", "batch_processing"],
                },
            },
        },
        "invalid_format": {
            "status_code": 200,
            "json": {
                "status": "success",
                # Missing required 'image' field
            },
        },
        "malformed_json": {"status_code": 200, "text": "not valid json"},
        "empty_response": {"status_code": 200, "json": {}},
    }


def create_mock_response(response_type):
    """Create a mock response object for the given response type."""
    from unittest.mock import Mock

    responses = get_mock_api_responses()
    response_data = responses.get(response_type)

    if not response_data:
        raise ValueError(f"Unknown response type: {response_type}")

    mock_response = Mock()
    mock_response.status_code = response_data["status_code"]
    mock_response.ok = response_data["status_code"] < 400

    if "json" in response_data:
        mock_response.json.return_value = response_data["json"]
    elif "text" in response_data:
        mock_response.text = response_data["text"]
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

    # Add raise_for_status method
    if mock_response.status_code >= 400:
        mock_response.raise_for_status.side_effect = Exception(
            f"HTTP {response_data['status_code']}"
        )
    else:
        mock_response.raise_for_status.return_value = None

    return mock_response
