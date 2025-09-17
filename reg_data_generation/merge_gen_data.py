#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
import shutil
import sys
from typing import List

import pandas as pd

def natural_key(path: Path):
    # sort ..._1, ..._2, ..., ..._10 numerically
    m = re.search(r"(\d+)$", path.name)
    return (path.parent, int(m.group(1)) if m else path.name)

def parse_args():
    ap = argparse.ArgumentParser(description="Merge gen_data_* runs into one dataset")
    ap.add_argument("--sources", type=str, default="data/gen_data_20000_*",
                    help="Glob for source run folders (default: data/gen_data_20000_*)")
    ap.add_argument("--dest", type=str, default="data/gen_data_40000",
                    help="Destination folder for merged dataset")
    ap.add_argument("--results_name", type=str, default="results.csv",
                    help="Name of the results CSV in each source folder")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--move", action="store_true", help="Move files instead of copying")
    grp.add_argument("--copy", action="store_true", help="Copy files (default)")
    ap.add_argument("--molecules_subdir", type=str, default="molecules",
                    help="Subfolder name containing .bin molecules")
    ap.add_argument("--expect_per_run", type=int, default=None,
                    help="If set, assert each run has this many rows/files")
    return ap.parse_args()

def main(args):
    copy_mode = not args.move  # default is copy

    src_dirs: List[Path] = sorted(Path(".").glob(args.sources), key=natural_key)
    if not src_dirs:
        print(f"No sources matched pattern: {args.sources}", file=sys.stderr)
        sys.exit(1)

    dest_root = Path(args.dest)
    dest_mols = dest_root / args.molecules_subdir
    dest_mols.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    global_index = 0
    op_name = "Copying" if copy_mode else "Moving"

    print(f"Found {len(src_dirs)} source runs:")
    for p in src_dirs:
        print(f"  - {p}")

    for run_idx, src in enumerate(src_dirs, start=1):
        src_results = src / args.results_name
        src_mols = src / args.molecules_subdir

        if not src_results.is_file():
            print(f"[WARN] Missing results at {src_results}; skipping run.", file=sys.stderr)
            continue
        if not src_mols.is_dir():
            print(f"[WARN] Missing molecules dir at {src_mols}; skipping run.", file=sys.stderr)
            continue

        # Read results
        df = pd.read_csv(src_results)
        if "id_str" not in df.columns:
            print(f"[WARN] 'id_str' column missing in {src_results}; skipping run.", file=sys.stderr)
            continue

        # Determine per-run count
        # Prefer count from results; cross-check with file count
        row_count = len(df)
        bin_files = sorted(src_mols.glob("mol_*.bin"))
        file_count = len(bin_files)

        if args.expect_per_run is not None:
            if row_count != args.expect_per_run or file_count != args.expect_per_run:
                print(f"[ERROR] Expectation mismatch in {src.name}: rows={row_count}, files={file_count}, expected={args.expect_per_run}", file=sys.stderr)
                sys.exit(2)

        if row_count != file_count:
            print(f"[WARN] Row/file count mismatch in {src.name}: rows={row_count}, files={file_count}. Proceeding with min().", file=sys.stderr)

        n = min(row_count, file_count)

        # Build quick map from local index -> file path.
        # Assume local naming mol_000000.bin ..; fall back to sorted listing.
        def local_path(i: int) -> Path:
            candidate = src_mols / f"mol_{i:06d}.bin"
            return candidate if candidate.exists() else bin_files[i]

        # Track provenance (which run / index this came from)
        df = df.iloc[:n].copy()
        df["source_run"] = src.name
        df["source_row"] = range(n)

        # Assign new global id_str and copy/move files
        new_id_strs = []
        for i in range(n):
            src_path = local_path(i)
            new_name = f"mol_{global_index:06d}.bin"
            dst_path = dest_mols / new_name

            # Ensure we don't overwrite anything accidentally
            if dst_path.exists():
                print(f"[ERROR] Destination file already exists: {dst_path}", file=sys.stderr)
                sys.exit(3)

            print(f"{op_name} {src_path} -> {dst_path}")
            if copy_mode:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

            new_id_strs.append(new_name[:-4])  # without .bin

            global_index += 1

        # Update id_str to global names
        df.loc[df.index[:n], "id_str"] = new_id_strs
        all_dfs.append(df)

    if not all_dfs:
        print("No data merged; exiting.", file=sys.stderr)
        sys.exit(0)

    merged = pd.concat(all_dfs, axis=0, ignore_index=True)

    # Write merged results
    out_csv = dest_root / "results.csv"
    merged.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nWrote merged results: {out_csv}")
    print(f"Total molecules: {len(merged)}")
    print(f"Destination molecules dir: {dest_mols}")

if __name__ == "__main__":
    main(parse_args())
