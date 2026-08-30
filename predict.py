#!/usr/bin/env python3
"""Score every image in a directory for likelihood of AIGC generation.

Usage:
    python predict.py ./images --checkpoint hybrid_checkpoint.pt --output predictions.json
"""

import argparse
import glob
import json
import os

from PIL import Image

from src.model import load_models, predict_image

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def find_images(directory: str):
    """Recursively collect supported image files in deterministic order."""
    paths = []
    for extension in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, "**", extension), recursive=True))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Score a directory of images for AIGC likelihood."
    )
    parser.add_argument(
        "image_dir",
        help="Directory of images to score (searched recursively).",
    )
    parser.add_argument(
        "--checkpoint",
        default="hybrid_checkpoint.pt",
        help="Path to the trained hybrid checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="predictions.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")

    print(f"Loading model from {args.checkpoint}...")
    models = load_models(args.checkpoint)

    image_paths = find_images(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    results = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                pred = predict_image(img, models)
            results.append({"image_path": path, "pred": round(pred, 4)})
        except Exception as exc:
            print(f"Skipping {path} (unreadable or inference error): {exc}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
