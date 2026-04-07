from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple real-bootstrap roots into one combined root.")
    parser.add_argument("--input-roots", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def merge_roots(input_roots: list[Path], output_root: Path, overwrite: bool) -> dict[str, int]:
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"output root already exists: {output_root}")

    views_out = output_root / "views"
    scaffold_out = output_root / "scaffold"
    geometry_out = output_root / "geometry"
    views_out.mkdir(parents=True, exist_ok=True)
    scaffold_out.mkdir(parents=True, exist_ok=True)
    geometry_out.mkdir(parents=True, exist_ok=True)

    seen_samples: set[str] = set()
    seen_geometry: set[str] = set()
    copied_views = 0
    copied_scaffolds = 0
    copied_geometry = 0
    skipped_geometry = 0

    for root in input_roots:
        for view_path in sorted((root / "views").glob("*.json")):
            sample_id = view_path.stem
            if sample_id in seen_samples:
                raise ValueError(f"duplicate sample id across roots: {sample_id}")
            seen_samples.add(sample_id)
            shutil.copy2(view_path, views_out / view_path.name)
            copied_views += 1

            scaffold_path = root / "scaffold" / f"{sample_id}.pt"
            if not scaffold_path.exists():
                raise FileNotFoundError(f"missing scaffold for sample {sample_id}: {scaffold_path}")
            shutil.copy2(scaffold_path, scaffold_out / scaffold_path.name)
            copied_scaffolds += 1

        for geometry_path in sorted((root / "geometry").glob("*.pt")):
            uid = geometry_path.stem
            if uid in seen_geometry:
                skipped_geometry += 1
                continue
            seen_geometry.add(uid)
            shutil.copy2(geometry_path, geometry_out / geometry_path.name)
            copied_geometry += 1

    summary = {
        "num_input_roots": len(input_roots),
        "copied_views": copied_views,
        "copied_scaffolds": copied_scaffolds,
        "copied_geometry": copied_geometry,
        "skipped_duplicate_geometry": skipped_geometry,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    input_roots = [Path(path) for path in args.input_roots]
    output_root = Path(args.output_root)
    summary = merge_roots(input_roots=input_roots, output_root=output_root, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
