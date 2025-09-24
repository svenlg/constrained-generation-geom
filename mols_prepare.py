#!/usr/bin/env python
# prepare_mols_all.py
# DGL .bin -> RDKit (all graphs) -> ensure 3D -> write .mol to prepared/
# Filters out molecules that are not fully connected

import argparse, re, sys
from pathlib import Path

def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

# --- deps ---------------------------------------------------------
try:
    from dgl.data.utils import load_graphs
except Exception as e:
    die(f"Failed to import DGL: {e}\nRun this in the env where DGL can load your bins.")
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as e:
    die(f"Failed to import RDKit: {e}")
# Your project bits
try:
    from flowmol.analysis.molecule_builder import SampledMolecule
    atom_type_map = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
except Exception as e:
    die(f"Import SampledMolecule/atom_type_map failed: {e}")

# --- helpers ------------------------------------------------------
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

def write_mol_safe(mol, out_path: Path, overwrite=False) -> bool:
    """Atomic write to .mol; skip if exists unless overwrite."""
    try:
        out_path = out_path.with_suffix(".mol")
        if out_path.exists() and not overwrite:
            print(f"[SKIP] Exists: {out_path.name}")
            return True
        tmp = out_path.with_suffix(".mol.tmp")
        Chem.MolToMolFile(mol, str(tmp))
        tmp.replace(out_path)
        print(f"[OK]   Wrote: {out_path.name}")
        return True
    except Exception as e:
        print(f"[WARN] Write failed {out_path.name}: {e}")
        return False

# --- main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare 3D .mol files from all molecules in DGL .bin samples (connected only).")
    ap.add_argument("--root", default="/Users/svlg/MasterThesis/v03_geom/aa_experiments/",
                    help="Root folder that contains experiment dirs (default: your path)")
    ap.add_argument("--experiment", default="0923_2305_al_dipole_energy",
                    help="Experiment name (default matches your example)")
    ap.add_argument("--seed", type=int, default=1, help="Seed suffix in folder name (default: 1)")
    ap.add_argument("--outdir", default="prepared", help="Output subfolder name (default: prepared)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .mol files")
    args = ap.parse_args()

    base = Path(args.root).expanduser().resolve() / f"{args.experiment}_{args.seed}"
    samples_dir = base / "samples"
    prepared_dir = base / args.outdir
    if not samples_dir.is_dir():
        die(f"Samples folder not found: {samples_dir}")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        [p for p in samples_dir.iterdir() if p.suffix == ".bin"],
        key=lambda p: numeric_key(p.name)
    )
    if not sample_files:
        die(f"No .bin files found in {samples_dir}")

    total = success = failed = skipped = 0
    for sfile in sample_files:
        try:
            graphs, _ = load_graphs(str(sfile))
        except Exception as e:
            print(f"[WARN] Could not load {sfile.name}: {e}")
            continue

        n = len(graphs)
        iter_id = numeric_key(sfile.name)
        base_stub = f"iter_{iter_id:06d}" if iter_id >= 0 else sfile.stem
        print(f"[INFO] {sfile.name}: {n} graphs -> processing all")

        for j in range(n):
            total += 1
            try:
                sm = SampledMolecule(graphs[j], atom_type_map)
                rd = getattr(sm, "rdkit_mol", None)
                if rd is None:
                    print(f"[WARN] idx {j:03d}: rdkit_mol is None")
                    failed += 1
                    continue
                if not is_connected(rd):
                    print(f"[SKIP] idx {j:03d}: molecule is disconnected")
                    skipped += 1
                    continue
                rd3d = ensure_3d(rd)
                if rd3d is None:
                    print(f"[WARN] idx {j:03d}: 3D generation failed")
                    failed += 1
                    continue
                out_name = f"{base_stub}_idx_{j:03d}.mol"
                out_path = prepared_dir / out_name
                if write_mol_safe(rd3d, out_path, overwrite=args.overwrite):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[WARN] idx {j:03d}: conversion failed: {e}")
                failed += 1

    print(f"[DONE] Tried {total} | wrote {success} | failed {failed} | skipped (disconnected) {skipped} | output: {prepared_dir}")

if __name__ == "__main__":
    main()
