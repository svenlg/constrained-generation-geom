#!/usr/bin/env python
"""
Analyze AL vs AM molecules across seeds and iterations.

Folder layout expected:

root/
    al_time_and_samples/
        0/
            samples/
                samples_0.bin
                samples_5.bin
                ...
            al_stats.csv
            config.yaml
            full_stats.csv
        1/
            ...
        ...
    am_time_and_samples/
        0/
            samples/
                samples_0.bin
                samples_5.bin
                ...
            ...
        ...

For each algorithm (AL/AM), seed (0..9) and samples_X.bin iteration,
we run the professor's validity pipeline and compute:

- total molecules
- valid molecules (passed RDKit pipeline)
- validity fraction
- mean QED (valid only)
- Lipinski rule-of-five:
    - mean number of violations (0..4)
    - fraction of molecules with zero violations ("Lipinski pass")
- mean logP (valid only)

Outputs:
- A big pandas DataFrame with all stats (with and without CIs)
- Markdown tables for validity, QED, Lipinski metrics, and logP
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


# --- deps ---------------------------------------------------------
try:
    from dgl.data.utils import load_graphs
except Exception as e:
    die(f"Failed to import DGL: {e}\nRun this in the env where DGL can load your bins.")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, QED, Crippen, Descriptors, rdMolDescriptors
except Exception as e:
    die(f"Failed to import RDKit: {e}")

# project bits
try:
    from flowmol.analysis.molecule_builder import SampledMolecule
    atom_type_map = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
except Exception as e:
    die(f"Import SampledMolecule/atom_type_map failed: {e}")


# --- mean + CI helper ---------------------------------------------
def mean_ci_scalar(samples, alpha: float = 0.05):
    """
    Compute mean and (1-alpha)% CI half-width for a 1D array of samples.
    Returns (mean, half_width).
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    n = samples.size
    if n == 0:
        return np.nan, np.nan

    mean = samples.mean()
    if n > 1:
        std = samples.std(ddof=1)
        # z-score for two-sided (1-alpha) CI, default 95%
        z = 1.96 if alpha == 0.05 else 1.96
        half_width = z * std / np.sqrt(n)
    else:
        half_width = 0.0
    return mean, half_width


# --- helpers from your original script ----------------------------
def numeric_key(name: str) -> int:
    m = re.search(r"\d+", name)
    return int(m.group()) if m else -1


def is_connected(mol):
    """Return True if molecule is a single connected fragment."""
    try:
        frags = Chem.GetMolFrags(mol, asMols=False)
        return len(frags) == 1
    except Exception:
        return False


def ensure_3d(mol):
    """Return a 3D mol. If no conformer or flat Z, embed + quick optimize. None on failure."""
    if mol is None:
        return None
    try:
        needs_3d = (mol.GetNumConformers() == 0)
        if not needs_3d:
            conf = mol.GetConformer()
            zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
            if (max(zs) - min(zs)) < 1e-4:
                needs_3d = True

        if needs_3d:
            mH = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            if AllChem.EmbedMolecule(mH, params) != 0:
                return None
            try:
                if AllChem.MMFFHasAllMoleculeParams(mH):
                    AllChem.MMFFOptimizeMolecule(mH, maxIters=200)
                else:
                    AllChem.UFFOptimizeMolecule(mH, maxIters=200)
            except Exception:
                pass
            mol = Chem.RemoveHs(mH)
        return mol
    except Exception:
        return None


def validate_mol(mol: Chem.Mol) -> Chem.Mol:
    """Validity check from your professor. Returns validated mol or None if invalid."""
    try:
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        Chem.AssignStereochemistryFrom3D(mol)

        for a in mol.GetAtoms():
            a.SetNoImplicit(True)
            if a.HasProp("_MolFileHCount"):
                a.ClearProp("_MolFileHCount")

        flags = Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        err = Chem.SanitizeMol(mol, sanitizeOps=flags, catchErrors=True)
        if err:
            return None
        else:
            mol.UpdatePropertyCache(strict=True)
            return mol
    except Exception:
        return None


def write_mol_safe(mol, out_path: Path, overwrite=False) -> bool:
    """Atomic write to .mol; skip if exists unless overwrite."""
    try:
        out_path = out_path.with_suffix(".mol")
        if out_path.exists() and not overwrite:
            return True
        tmp = out_path.with_suffix(".mol.tmp")
        Chem.MolToMolFile(mol, str(tmp))
        tmp.replace(out_path)
        return True
    except Exception as e:
        print(f"[WARN] Write failed {out_path.name}: {e}")
        return False


# --- Lipinski helpers ---------------------------------------------
def lipinski_violations(mol: Chem.Mol) -> int:
    """
    Compute number of Lipinski rule-of-five violations (0–4):
      - MW > 500
      - logP > 5
      - H-bond donors > 5
      - H-bond acceptors > 10
    """
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        return violations
    except Exception:
        # treat as worst case: all rules violated
        return 4


# --- main analysis ------------------------------------------------
def analyze(root: Path, outdir: str, cfo_folder: str, am_folder: str = None, overwrite_mol: bool = False):
    """
    Walk AL/AM folders, compute stats, optionally write .mol files.

    Returns:
        agg_base : DataFrame with global stats (no CI)
        agg_full : DataFrame with global stats + mean/CI across seeds
    """
    # (label, folder_name)
    algos = [("CFO", cfo_folder),
             ("AM", am_folder)]

    records = []  # per (algo, seed, iteration) aggregate

    for algo_label, algo_folder in algos:
        if algo_folder is None:
            continue
        algo_root = root / algo_folder
        if not algo_root.is_dir():
            print(f"[WARN] Algorithm folder not found: {algo_root}")
            continue

        # seeds: any numeric directories
        seed_dirs = sorted(
            [d for d in algo_root.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda p: int(p.name)
        )
        if not seed_dirs:
            print(f"[WARN] No seed dirs under {algo_root}")
            continue

        for seed_dir in seed_dirs:
            seed = int(seed_dir.name)
            samples_dir = seed_dir / "samples"
            if not samples_dir.is_dir():
                print(f"[WARN] No samples dir in {seed_dir}")
                continue

            sample_files = sorted(
                [p for p in samples_dir.iterdir() if p.suffix == ".bin"],
                key=lambda p: numeric_key(p.name),
            )
            if not sample_files:
                print(f"[WARN] No .bin files in {samples_dir}")
                continue

            print(f"[INFO] {algo_label} seed {seed}: {len(sample_files)} sample bins")

            for sfile in sample_files:
                try:
                    graphs, _ = load_graphs(str(sfile))
                except Exception as e:
                    print(f"[WARN] Could not load {sfile}: {e}")
                    continue

                iter_id = numeric_key(sfile.name)
                n_total = len(graphs)
                n_valid = 0
                qed_sum = 0.0
                lip_pass_sum = 0  # count of molecules with 0 violations
                lip_viol_sum = 0  # total #violations
                logp_sum = 0.0

                # if we want to write .mol files, set up folder: seed/outdir/iterXX
                if overwrite_mol or outdir:
                    iter_tag = f"{iter_id:02d}" if (0 <= iter_id < 100) else str(iter_id)
                    iter_dir = seed_dir / outdir / f"iter{iter_tag}"
                    if overwrite_mol or not iter_dir.exists():
                        iter_dir.mkdir(parents=True, exist_ok=True)
                else:
                    iter_dir = None

                for j, g in enumerate(graphs):
                    try:
                        sm = SampledMolecule(g, atom_type_map)
                        rd = getattr(sm, "rdkit_mol", None)
                        if rd is None:
                            continue

                        if not is_connected(rd):
                            continue

                        rd3d = ensure_3d(rd)
                        if rd3d is None:
                            continue

                        rd_valid = validate_mol(Chem.Mol(rd3d))
                        if rd_valid is None:
                            continue

                        # valid molecule
                        n_valid += 1

                        # QED
                        try:
                            q = float(QED.qed(rd_valid))
                        except Exception:
                            q = 0.0
                        qed_sum += q

                        # Lipinski
                        v = lipinski_violations(rd_valid)
                        lip_viol_sum += v
                        if v == 0:
                            lip_pass_sum += 1

                        # logP
                        try:
                            lp = Crippen.MolLogP(rd_valid)
                        except Exception:
                            lp = np.nan
                        logp_sum += lp

                        # Optional: write mol file
                        if iter_dir is not None:
                            out_name = f"{sfile.stem}_idx_{j:03d}.mol"
                            out_path = iter_dir / out_name
                            write_mol_safe(rd_valid, out_path, overwrite=overwrite_mol)

                    except Exception:
                        # Any failure for this graph: just skip, it's invalid
                        continue

                records.append(
                    dict(
                        algo=algo_label,
                        seed=seed,
                        iteration=iter_id,
                        total=n_total,
                        valid=n_valid,
                        qed_sum=qed_sum,
                        lip_pass_sum=lip_pass_sum,
                        lip_viol_sum=lip_viol_sum,
                        logp_sum=logp_sum,
                    )
                )

    if not records:
        die("No records collected - check your paths / data.")

    df = pd.DataFrame.from_records(records)

    # ---- seed-level derived metrics (per algo, seed, iteration) ----
    df["valid_fraction"] = df["valid"] / df["total"].replace(0, np.nan)
    df["qed_mean"] = df["qed_sum"] / df["valid"].replace(0, np.nan)
    df["lipinski_pass_fraction"] = df["lip_pass_sum"] / df["valid"].replace(0, np.nan)
    df["lipinski_violations_mean"] = df["lip_viol_sum"] / df["valid"].replace(0, np.nan)
    df["logp_mean"] = df["logp_sum"] / df["valid"].replace(0, np.nan)

    # ---- base aggregation (no CI, global over all seeds) -----------
    agg_base = (
        df.groupby(["algo", "iteration"], as_index=False)
          .agg(
              total=("total", "sum"),
              valid=("valid", "sum"),
              qed_sum=("qed_sum", "sum"),
              lip_pass_sum=("lip_pass_sum", "sum"),
              lip_viol_sum=("lip_viol_sum", "sum"),
              logp_sum=("logp_sum", "sum"),
          )
    )

    agg_base["valid_fraction"] = agg_base["valid"] / agg_base["total"].replace(0, np.nan)
    agg_base["qed_mean"] = agg_base["qed_sum"] / agg_base["valid"].replace(0, np.nan)
    agg_base["lipinski_pass_fraction"] = agg_base["lip_pass_sum"] / agg_base["valid"].replace(0, np.nan)
    agg_base["lipinski_violations_mean"] = agg_base["lip_viol_sum"] / agg_base["valid"].replace(0, np.nan)
    agg_base["logp_mean"] = agg_base["logp_sum"] / agg_base["valid"].replace(0, np.nan)

    # ---- CI aggregation (mean ± 95% CI across seeds) ---------------
    group = df.groupby(["algo", "iteration"])

    metric_cols = [
        "valid_fraction",
        "qed_mean",
        "lipinski_pass_fraction",
        "lipinski_violations_mean",
        "logp_mean",
    ]

    stats_list = []
    for m in metric_cols:
        tmp = group[m].apply(
            lambda x: pd.Series(
                mean_ci_scalar(x.values),
                index=[f"{m}_mean", f"{m}_ci"],
            )
        )
        stats_list.append(tmp)

    metrics_ci = pd.concat(stats_list, axis=1).reset_index()

    # merge base stats with CI stats
    agg_full = agg_base.merge(metrics_ci, on=["algo", "iteration"], how="left")


    index_cols = [
        "algo", "iteration", "total", "valid",
        "qed_sum", "lip_pass_sum", "lip_viol_sum", "logp_sum",
        "valid_fraction_x", "qed_mean_x",
        "lipinski_pass_fraction_x", "lipinski_violations_mean_x",
    ]

    value_cols = [
        "valid_fraction_y",
        "qed_mean_y",
        "lipinski_pass_fraction_y",
        "lipinski_violations_mean_y",
        "logp_mean_y",
    ]

    # pivot only the *_y columns over level_2
    flat = (
        agg_full
        .pivot_table(
            index=index_cols,
            columns="level_2",
            values=value_cols,
            aggfunc="first"
        )
    )

    # flatten MultiIndex columns: ('valid_fraction_y','valid_fraction_mean') -> 'valid_fraction_mean'
    flat.columns = [col_level2 for (_, col_level2) in flat.columns]

    # back to a normal DataFrame
    agg_full = flat.reset_index()
    # rename cols "_x" -> "all"
    agg_full = agg_full.rename(columns={
        "valid_fraction_x": "valid_fraction",
        "qed_mean_x": "qed_mean",
        "lipinski_pass_fraction_x": "lipinski_pass_fraction",
        "lipinski_violations_mean_x": "lipinski_violations_mean",
        "logp_mean_x": "logp_mean",
    })

    return agg_base, agg_full


# --- markdown printing --------------------------------------------
def print_markdown_tables_no_ci(agg: pd.DataFrame):
    """
    Print markdown tables WITHOUT confidence intervals.

    Rows: iteration
    Columns: CFO, AM
    """
    def metric_table(metric, pretty_name, fmt=".3f"):
        pivot = (
            agg.pivot(index="iteration", columns="algo", values=metric)
               .sort_index()
        )
        print(f"\n## {pretty_name} (no CI)\n")
        print(pivot.to_markdown(floatfmt=fmt))

    metric_table("valid_fraction", "Validity fraction (valid / total)")
    metric_table("qed_mean", "Mean QED (valid molecules)")
    metric_table("lipinski_pass_fraction", "Lipinski pass fraction (0 violations)")
    metric_table("lipinski_violations_mean", "Mean #Lipinski violations (0-4)")
    metric_table("logp_mean", "Mean logP (valid molecules)")


def print_markdown_tables_ci(agg: pd.DataFrame):
    """
    Print markdown tables WITH 95% confidence intervals.

    Each cell is "mean ± 95% CI" across seeds.
    Rows: iteration
    Columns: AL, AM
    """
    def metric_table(metric, pretty_name):
        mean_col = f"{metric}_mean"
        ci_col = f"{metric}_ci"

        if mean_col not in agg.columns or ci_col not in agg.columns:
            print(f"\n[WARN] Skipping {metric}: columns {mean_col} / {ci_col} not found.")
            return

        pivot_mean = (
            agg.pivot(index="iteration", columns="algo", values=mean_col)
               .sort_index()
        )
        pivot_ci = (
            agg.pivot(index="iteration", columns="algo", values=ci_col)
               .sort_index()
        )

        combined = pivot_mean.copy().astype(object)

        for idx in combined.index:
            for col in combined.columns:
                m = pivot_mean.loc[idx, col]
                c = pivot_ci.loc[idx, col]
                if pd.isna(m) or pd.isna(c):
                    combined.loc[idx, col] = ""
                else:
                    combined.loc[idx, col] = f"{m:.3f} ± {c:.3f}"

        print(f"\n## {pretty_name} (mean ± 95% CI across seeds)\n")
        print(combined.to_markdown())

    metric_table("valid_fraction", "Validity fraction (valid / total)")
    metric_table("qed_mean", "Mean QED (valid molecules)")
    metric_table("lipinski_pass_fraction", "Lipinski pass fraction (0 violations)")
    metric_table("lipinski_violations_mean", "Mean #Lipinski violations (0-4)")
    metric_table("logp_mean", "Mean logP (valid molecules)")


# --- main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Analyze AL vs AM molecules across seeds and iterations."
    )
    ap.add_argument(
        "--root",
        default="/Users/svlg/MasterThesis/v03_geom/aa_experiments/",
        help="Root folder.",
    )
    ap.add_argument(
        "--outdir",
        default="prepared",
        help="Subfolder under each seed where valid .mol files are written (default: prepared).",
    )
    ap.add_argument(
        "--cfo-folder",
        default="cfo_time_and_samples",
        help="Subfolder under each seed for CFO molecules (default: cfo_time_and_samples).",
    )
    ap.add_argument(
        "--am-folder",
        default=None,
        help="Subfolder under each seed for AM molecules (default: am_time_and_samples).",
    )
    ap.add_argument(
        "--overwrite-mol",
        action="store_true",
        help="Overwrite existing .mol files if they exist.",
    )
    ap.add_argument(
        "--csv",
        default="cfo_mols_stats",
        help="Base name (no extension) to save CSVs under the root folder.",
    )

    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()

    agg_base, agg_full = analyze(
        root=root,
        outdir=args.outdir,
        cfo_folder=args.cfo_folder,
        am_folder=args.am_folder,
        overwrite_mol=args.overwrite_mol,
    )

    print("\n==================== SUMMARY DATAFRAME (no CI) ====================\n")
    print(agg_base)

    print("\n==================== SUMMARY DATAFRAME (with CI) ==================\n")
    print(agg_full)

    if args.csv:
        out_base_path = Path(args.root) / f"{args.csv}_base.csv"
        out_full_path = Path(args.root) / f"{args.csv}_full.csv"
        agg_base.to_csv(out_base_path, index=False)
        agg_full.to_csv(out_full_path, index=False)
        print(f"\n[INFO] Saved aggregated stats to:\n  {out_base_path}\n  {out_full_path}")

    print("\n==================== MARKDOWN TABLES (no CI) =====================\n")
    print_markdown_tables_no_ci(agg_base)

    print("\n==================== MARKDOWN TABLES (with CI) ===================\n")
    print_markdown_tables_ci(agg_full)


if __name__ == "__main__":
    main()


"""
python mols_stats_analyze.py \
    --root /Users/svlg/MasterThesis/v03_geom/aa_experiments/ \
    --outdir prepared \
    --cfo-folder cfo_rl_25 \
    --am-folder am_rl_25 \
    --csv cfo_am_rl_25
"""