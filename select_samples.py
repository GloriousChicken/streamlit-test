#!/usr/bin/env python3
"""Scan xBD test labels and select diverse demo samples for the Streamlit app."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_LABELS = "REDACTED_PATH"
DEFAULT_IMAGES = "REDACTED_PATH"
DEFAULT_OUTPUT = "./samples/"


def count_buildings(label_path: Path) -> int:
    with open(label_path) as f:
        data = json.load(f)
    return len(data.get("features", {}).get("xy", []))


def get_metadata(label_path: Path) -> dict:
    with open(label_path) as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    return {
        "disaster": meta.get("disaster", "unknown"),
        "disaster_type": meta.get("disaster_type", "unknown"),
    }


def get_class_distribution(label_path: Path) -> dict:
    """Return {subtype: count} for all buildings in a post-disaster label."""
    with open(label_path) as f:
        data = json.load(f)
    dist = defaultdict(int)
    for feat in data.get("features", {}).get("xy", []):
        sub = feat.get("properties", {}).get("subtype", "no-damage")
        dist[sub] += 1
    return dict(dist)


def parse_sample_name(filename: str) -> str:
    """Extract sample base name: everything before _pre_disaster or _post_disaster."""
    for suffix in ("_post_disaster.json", "_pre_disaster.json"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def select_samples(labels_dir: Path, min_b: int, max_b: int, num_samples: int, min_classes: int = 3):
    post_files = sorted(labels_dir.glob("*_post_disaster.json"))
    if not post_files:
        print(f"No *_post_disaster.json files found in {labels_dir}")
        return []

    print(f"Scanning {len(post_files)} post-disaster label files...")

    # Collect candidates
    candidates = []
    for p in post_files:
        n = count_buildings(p)
        if min_b <= n <= max_b:
            meta = get_metadata(p)
            class_dist = get_class_distribution(p)
            candidates.append(
                {
                    "path": p,
                    "name": parse_sample_name(p.name),
                    "buildings": n,
                    "disaster": meta["disaster"],
                    "disaster_type": meta["disaster_type"],
                    "class_dist": class_dist,
                    "num_classes": len(class_dist),
                }
            )

    print(f"  {len(candidates)} candidates with {min_b}–{max_b} buildings")

    # Filter by class diversity
    candidates = [c for c in candidates if c["num_classes"] >= min_classes]
    print(f"  {len(candidates)} candidates with >= {min_classes} damage classes")

    # Group by event, pick closest to 50 buildings from each
    by_event = defaultdict(list)
    for c in candidates:
        by_event[c["disaster"]].append(c)

    best_per_event = {}
    for event, samples in by_event.items():
        best = min(samples, key=lambda s: (-s["num_classes"], abs(s["buildings"] - 50)))
        best_per_event[event] = best

    print(f"  {len(best_per_event)} distinct events with candidates")

    # Maximize disaster_type diversity: pick from as many distinct types as possible
    by_type = defaultdict(list)
    for event, sample in best_per_event.items():
        by_type[sample["disaster_type"]].append(sample)

    selected = []
    # Round-robin across disaster types
    type_iters = {t: iter(samples) for t, samples in by_type.items()}
    while len(selected) < num_samples and type_iters:
        exhausted = []
        for dtype, it in type_iters.items():
            if len(selected) >= num_samples:
                break
            try:
                sample = next(it)
                selected.append(sample)
            except StopIteration:
                exhausted.append(dtype)
        for dtype in exhausted:
            del type_iters[dtype]

    return selected


def format_class_dist(dist: dict) -> str:
    """Format class distribution as e.g. '4cls: ND=15 Mi=3 Ma=5 De=2'."""
    abbrev = {"no-damage": "ND", "minor-damage": "Mi", "major-damage": "Ma", "destroyed": "De"}
    parts = [f"{abbrev.get(k, k)}={v}" for k, v in sorted(dist.items())]
    return f"{len(dist)}cls: {' '.join(parts)}"


def print_results(selected):
    print(f"\nSelected {len(selected)} samples:\n")
    print(f"  {'Name':<45} {'Buildings':>9}  {'Event':<30} {'Type':<20} {'Classes'}")
    print(f"  {'─'*45} {'─'*9}  {'─'*30} {'─'*20} {'─'*30}")
    for s in selected:
        cls = format_class_dist(s["class_dist"])
        print(
            f"  {s['name']:<45} {s['buildings']:>9}  {s['disaster']:<30} {s['disaster_type']:<20} {cls}"
        )

    # Print SAMPLE_PAIRS snippet
    print("\n\n# SAMPLE_PAIRS snippet for app.py:\n")
    print("SAMPLE_PAIRS = [")
    for s in selected:
        print(f'    "{s["name"]}",')
    print("]")


def copy_files(selected, labels_dir: Path, images_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = [
        "_pre_disaster.json",
        "_post_disaster.json",
        "_pre_disaster.png",
        "_post_disaster.png",
    ]
    for s in selected:
        for suffix in suffixes:
            filename = s["name"] + suffix
            if suffix.endswith(".json"):
                src = labels_dir / filename
            else:
                src = images_dir / filename
            dst = output_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {filename}")
            else:
                print(f"  WARNING: {src} not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=Path(DEFAULT_LABELS))
    parser.add_argument("--images-dir", type=Path, default=Path(DEFAULT_IMAGES))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--min-buildings", type=int, default=20)
    parser.add_argument("--max-buildings", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument(
        "--min-classes", type=int, default=3,
        help="Minimum distinct damage classes per sample (1–4, default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print selections without copying"
    )
    args = parser.parse_args()

    selected = select_samples(
        args.labels_dir, args.min_buildings, args.max_buildings, args.num_samples,
        min_classes=args.min_classes,
    )
    if not selected:
        return

    print_results(selected)

    if args.dry_run:
        print("\n(Dry run — no files copied)")
    else:
        print(f"\nCopying files to {args.output_dir}/ ...")
        copy_files(selected, args.labels_dir, args.images_dir, args.output_dir)
        print("Done.")


if __name__ == "__main__":
    main()
