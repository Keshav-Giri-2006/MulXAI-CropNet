"""
Batch severity pseudo-label generation.

Walks a PlantVillage-style dataset directory and writes
datasets/severity_labels/severity_labels.csv with columns
(filename, disease, severity), per MEMBER2_SEVERITY_GUIDE.md and
DATASET_USAGE.md.

Reuses src.training.dataset_loader.get_class_names() and
load_dataset_paths() (Member 1, frozen, read-only) for directory discovery
and path enumeration, rather than reimplementing dataset traversal here.
This is the only Member 1 module this file imports from, and it is used
exactly as documented in MEMBER1_TO_MEMBER2_HANDOFF.md (auto-discovered
class names, no hardcoded class list).
"""

import argparse
import csv
from pathlib import Path
from typing import List

from src.preprocessing.severity_generator import compute_severity_for_file
from src.training.dataset_loader import get_class_names, load_dataset_paths

DEFAULT_OUTPUT_CSV = "datasets/severity_labels/severity_labels.csv"
CSV_FIELDNAMES = ["filename", "disease", "severity"]


def is_healthy_class(class_name: str) -> bool:
    """
    Determine whether a disease class name represents the healthy class.

    Matches case-insensitively on the substring "healthy" (e.g.
    "Tomato_healthy") rather than an exact string, so this keeps working
    if the exact PlantVillage folder naming convention changes without
    Member 2 needing to update this file.

    Args:
        class_name: A disease class name, as returned by get_class_names().

    Returns:
        True if this class represents "no disease" / healthy tissue.
    """
    return "healthy" in class_name.lower()


def generate_severity_labels(
    dataset_root: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
) -> str:
    """
    Generate severity_labels.csv for every image in a PlantVillage-style
    dataset directory.

    Args:
        dataset_root: Path to the dataset root (e.g. 'datasets/PlantVillage/').
        output_csv: Path to write the resulting CSV. Default:
            'datasets/severity_labels/severity_labels.csv'.

    Returns:
        The output_csv path, for convenience when chaining calls.
    """
    class_names = get_class_names(dataset_root)
    image_paths, labels = load_dataset_paths(dataset_root, class_names)

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Generating severity labels for {len(image_paths)} images...")

    rows: List[dict] = []
    skipped = 0

    for image_path, label_idx in zip(image_paths, labels):
        disease = class_names[label_idx]
        healthy = is_healthy_class(disease)

        try:
            severity = compute_severity_for_file(image_path, is_healthy=healthy)
        except (OSError, ValueError) as exc:
            # A single unreadable/corrupt image should not abort a batch
            # run over thousands of files; it is skipped and counted.
            print(f"  Skipping unreadable image {image_path}: {exc}")
            skipped += 1
            continue

        rows.append(
            {
                "filename": Path(image_path).name,
                "disease": disease,
                "severity": round(severity, 1),
            }
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Severity labels saved to {output_path} ({len(rows)} images, {skipped} skipped)")
    return str(output_path)


def main() -> None:
    """CLI entry point: python -m src.preprocessing.pseudo_labels [args]."""
    parser = argparse.ArgumentParser(
        description="Generate continuous (0-100) severity pseudo-labels "
        "for a PlantVillage-style dataset via HSV lesion masking."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/PlantVillage/",
        help="Path to the dataset root directory. Default: datasets/PlantVillage/",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to write severity_labels.csv. Default: {DEFAULT_OUTPUT_CSV}",
    )

    args = parser.parse_args()
    generate_severity_labels(args.dataset_root, args.output_csv)


if __name__ == "__main__":
    main()
