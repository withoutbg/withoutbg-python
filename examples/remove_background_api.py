"""Remove a background using the withoutBG cloud API.

Requires an API key from https://withoutbg.com.

Set your key in the environment (recommended — keeps it out of code):
    export WITHOUTBG_API_KEY=sk_your_key

Or pass it directly (only for local scripts):
    python examples/remove_background_api.py photo.jpg sk_your_key

Run:
    python examples/remove_background_api.py photo.jpg
"""

import sys
from pathlib import Path

from withoutbg import WithoutBG, APIError


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python remove_background_api.py <image> [api_key]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    api_key = (
        sys.argv[2] if len(sys.argv) > 2 else None
    )  # falls back to WITHOUTBG_API_KEY env var
    output_path = input_path.with_stem(input_path.stem + "-withoutbg").with_suffix(
        ".png"
    )

    try:
        model = WithoutBG.api(api_key=api_key)
        print(f"Processing {input_path} via cloud API...")
        result = model.remove_background(input_path)
        result.save(output_path)
        print(f"Saved to {output_path}")
    except APIError as e:
        print(f"API error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
