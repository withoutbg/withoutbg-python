"""Synchronize the small shared runtime into sibling community distributions."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "src/withoutbg/gateway.py").read_bytes()
    targets = [
        repo.parent / "withoutbg-inference/model/withoutbg_openweights/gateway.py",
        repo.parent / "community/withoutbg-hf/space/gateway.py",
    ]
    for path in targets:
        if args.check:
            if path.read_bytes() != source:
                raise SystemExit(f"Shared gateway copy differs: {path}")
        else:
            path.write_bytes(source)
    print("Community gateway copies match")


if __name__ == "__main__":
    main()
