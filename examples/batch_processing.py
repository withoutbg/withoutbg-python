"""Batch background removal using the local open-weights model.

Processes all JPG/PNG/WEBP files in a directory. The model is loaded once
and reused for every image — do not recreate it per image or you will reload
~2GB of weights on every call.

Run:
    python examples/batch_processing.py ./input-photos/ ./output-photos/
"""

import sys
from pathlib import Path

from withoutbg import WithoutBG

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python batch_processing.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s). Loading model...")
    model = WithoutBG.open_weights()

    for i, image_path in enumerate(images, 1):
        output_path = (output_dir / image_path.name).with_suffix(".png")
        print(f"[{i}/{len(images)}] {image_path.name} → {output_path.name}")
        result = model.remove_background(image_path)
        result.save(output_path)

    print(f"Done. {len(images)} image(s) saved to {output_dir}/")


if __name__ == "__main__":
    main()
