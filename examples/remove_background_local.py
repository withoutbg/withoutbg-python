"""Remove a background using the local open-weights model.

Run:
    python examples/remove_background_local.py photo.jpg

The model (~495MB) is downloaded from Hugging Face on first run and cached.
Subsequent runs load from cache and take 2–5 seconds per image.
"""

import sys
from pathlib import Path

from withoutbg import WithoutBG


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python remove_background_local.py <image>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.with_stem(input_path.stem + "-withoutbg").with_suffix(
        ".png"
    )

    print(f"Loading model (downloads ~495MB on first run)...")
    model = WithoutBG.open_weights()

    print(f"Processing {input_path}...")
    result = model.remove_background(input_path)

    result.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
