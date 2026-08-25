#!/usr/bin/env python3
"""Build a deterministic SLURM-array manifest from DCC experiment roots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TIFF_EXTENSIONS = {".tif", ".tiff"}


def collect(folder: Path, suffix: str) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    files: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TIFF_EXTENSIONS:
            continue
        if not path.stem.lower().endswith(suffix.lower()):
            continue
        pair_name = path.stem[: -len(suffix)]
        if pair_name in files:
            raise ValueError(f"Duplicate pair {pair_name!r} in {folder}")
        files[pair_name] = path.resolve()
    return files


def safe_fragment(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return safe.strip("._") or "unnamed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--group1-folder", default="group_1")
    parser.add_argument("--group2-folder", default="group_2")
    parser.add_argument("--group1-suffix", default="_group1")
    parser.add_argument("--group2-suffix", default="_group2")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.experiments.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"experiment", "root"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Experiments TSV must have tab-separated headers: experiment and root"
            )

        for source in reader:
            experiment = source["experiment"].strip()
            root = Path(source["root"].strip()).expanduser()
            if not experiment or not str(root):
                continue

            group1 = collect(root / args.group1_folder, args.group1_suffix)
            group2 = collect(root / args.group2_folder, args.group2_suffix)
            missing2 = sorted(set(group1) - set(group2))
            missing1 = sorted(set(group2) - set(group1))
            if missing1 or missing2:
                raise ValueError(
                    f"Unmatched files in {root}: "
                    f"missing group_1={missing1}; missing group_2={missing2}"
                )

            for pair_name in sorted(group1):
                output_dir = (
                    args.results_root
                    / safe_fragment(experiment)
                    / "per_image"
                    / safe_fragment(pair_name)
                )
                rows.append({
                    "experiment": experiment,
                    "pair_name": pair_name,
                    "group1_path": str(group1[pair_name]),
                    "group2_path": str(group2[pair_name]),
                    "output_dir": str(output_dir),
                })

    if not rows:
        raise ValueError("No matched TIFF pairs were found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment", "pair_name", "group1_path", "group2_path", "output_dir"
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"Manifest: {args.output}")
    print(f"Matched pairs: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
