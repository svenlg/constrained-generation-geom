#!/usr/bin/env python
"""
Evaluate saved DGL samples with AugmentedReward + pred_vs_real.

Outputs:
  <...>/<experiment>_<seed>/eval/eval_long.csv
  <...>/<experiment>_<seed>/eval/eval_wide.parquet

The long CSV has one row per (iteration file, sample idx):
  iter, idx, pred_reward, pred_constraint, true_reward, true_constraint
The wide Parquet has rows = iteration; columns = MultiIndex [('pred_reward', j), ...].
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
import numpy as np

# --- your project imports ---
from omegaconf import OmegaConf
from dgl.data.utils import load_graphs
import dgl

from regessor import RCModel                       # your RC model
from true_rc import pred_vs_real                   # your "true" evaluator
from environment import AugmentedReward            # your AugmentedReward wrapper
from flowmol.analysis.molecule_builder import SampledMolecule

# atom mapping used by SampledMolecule
ATOM_TYPE_MAP = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


# ---------------- utilities ----------------
def die(msg: str, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def numeric_key(name: str) -> int:
    m = re.search(r"\d+", name)
    return int(m.group()) if m else -1

def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_regressor(config: OmegaConf, device: torch.device):
    import os.path as osp
    K_x, K_a, K_c, K_e = 3, 10, 6, 5
    model_path = osp.join("pretrained_models", config.fn, config.model_type, config.date, "best_model.pt")
    state = torch.load(model_path, map_location=device)
    if config.model_type == "gnn":
        model_config = {
            "property": state["config"]["property"],
            "node_feats": K_a + K_c + K_x,
            "edge_feats": K_e,
            "hidden_dim": state["config"]["hidden_dim"],
            "depth": state["config"]["depth"],
        }
        model = RCModel(property=model_config["property"], config=config, model_config=model_config)
    elif config.model_type == "egnn":
        model_config = {
            "property": state["config"]["property"],
            "num_atom_types": K_a,
            "num_charge_classes": K_c,
            "num_bond_types": K_e,
            "hidden_dim": state["config"]["hidden_dim"],
            "depth": state["config"]["depth"],
        }
        model = RCModel(property=model_config["property"], config=config, model_config=model_config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    model_config["filter_config"] = OmegaConf.to_container(config.filter_config)
    model.gnn.load_state_dict(state["model_state"])
    model.to(device).eval()
    return model, OmegaConf.create(model_config)


# --------------- main evaluation ---------------
def eval_saved_samples(
    root: str,
    experiment: str,
    seed: int,
    config_path: str = "configs/augmented_lagrangian.yaml",
    outdir: str = "eval",
):
    device = best_device()

    base = Path(root).expanduser().resolve() / f"{experiment}_{seed}"
    samples_dir = base / "samples"
    out_dir = base / outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not samples_dir.is_dir():
        die(f"Samples folder not found: {samples_dir}")

    # ---- load config + models ----
    cfg = OmegaConf.load(config_path)

    reward_model, _ = load_regressor(cfg.reward, device=device)
    constraint_model, _ = load_regressor(cfg.constraint, device=device)

    augmented_reward = AugmentedReward(
        reward_fn=reward_model,
        constraint_fn=constraint_model,
        alpha=cfg.reward_lambda,
        bound=cfg.constraint.bound,
        config=cfg.augmented_reward,
        device=device,
    )
    augmented_reward.set_lambda_rho(lambda_=0.0, rho_=cfg.augmented_lagrangian.rho_init)

    sample_files = sorted([p for p in samples_dir.iterdir() if p.suffix == ".bin"], key=lambda p: numeric_key(p.name))
    if not sample_files:
        die(f"No .bin files in {samples_dir}")

    rows = []
    per_iter_pred_reward: Dict[int, Dict[int, float]] = {}
    per_iter_pred_constraint: Dict[int, Dict[int, float]] = {}
    per_iter_true_reward: Dict[int, Dict[int, float]] = {}
    per_iter_true_constraint: Dict[int, Dict[int, float]] = {}

    for sfile in sample_files:
        graphs, _ = load_graphs(str(sfile))
        n = len(graphs)
        iter_id = numeric_key(sfile.name)
        print(f"[INFO] {sfile.name}: {n} graphs")

        # Batch DGL graphs and move to device for AugmentedReward
        dgl_batch = dgl.batch(graphs).to(device)

        # ---- PREDICTED: via AugmentedReward ----
        _ = augmented_reward(dgl_batch)  # <--- REQUIRED CALL
        pred_rc = augmented_reward.get_reward_constraint()  # <--- {'reward': tensor[N], 'constraint': tensor[N]}

        pred_reward_all = pred_rc["reward"].detach().cpu().numpy().reshape(-1)
        pred_constraint_all = pred_rc["constraint"].detach().cpu().numpy().reshape(-1)

        # ---- Build RDKit mols (and mask invalid/disconnected if requested) ----
        rd_kept = []
        keep_idx = []
        for j, g in enumerate(graphs):
            try:
                sm = SampledMolecule(g, ATOM_TYPE_MAP)
                rd = getattr(sm, "rdkit_mol", None)
                if rd is None:
                    continue
                rd_kept.append(rd)
                keep_idx.append(j)
            except Exception:
                continue

        # subset predictions to those we kept
        if len(keep_idx) == 0:
            print(f"[WARN] iter {iter_id}: no valid molecules after filtering")
            continue

        keep_idx_np = np.array(keep_idx, dtype=int)
        pred_rc_masked = {
            "reward": torch.as_tensor(pred_reward_all[keep_idx_np]),
            "constraint": torch.as_tensor(pred_constraint_all[keep_idx_np]),
        }

        # ---- TRUE values vs predictions ----
        log_pred_vs_real, true_reward, true_constraint = pred_vs_real(
            rd_kept, pred_rc_masked,
            reward=cfg.reward.fn, constraint=cfg.constraint.fn
        )

        # ---- collect rows ----
        per_iter_pred_reward.setdefault(iter_id, {})
        per_iter_pred_constraint.setdefault(iter_id, {})
        per_iter_true_reward.setdefault(iter_id, {})
        per_iter_true_constraint.setdefault(iter_id, {})

        # map masked arrays back to original indices
        for k, j in enumerate(keep_idx):
            pr = float(pred_rc_masked["reward"][k].item()) if pred_rc_masked["reward"][k] is not None else np.nan
            pc = float(pred_rc_masked["constraint"][k].item()) if pred_rc_masked["constraint"][k] is not None else np.nan
            tr = float(true_reward[k]) if true_reward is not None else np.nan
            tc = float(true_constraint[k]) if true_constraint is not None else np.nan

            rows.append({
                "iter": iter_id, "idx": j,
                "pred_reward": pr, "pred_constraint": pc,
                "true_reward": tr, "true_constraint": tc,
            })
            per_iter_pred_reward[iter_id][j] = pr
            per_iter_pred_constraint[iter_id][j] = pc
            per_iter_true_reward[iter_id][j] = tr
            per_iter_true_constraint[iter_id][j] = tc

        # optional: print quick summary
        print(f"  mean pred reward: {np.nanmean([per_iter_pred_reward[iter_id][j] for j in keep_idx]):.4f} "
              f"| mean true reward: {np.nanmean([per_iter_true_reward[iter_id][j] for j in keep_idx]):.4f}")

    # ---- Write long CSV ----
    df_long = pd.DataFrame(rows).sort_values(["iter", "idx"]).reset_index(drop=True)
    long_path = out_dir / "eval_long.csv"
    df_long.to_csv(long_path, index=False)
    print(f"[SAVE] {long_path}")

    # ---- Build wide Parquet (rows=iter; columns=MultiIndex: ('pred_reward', j), ...) ----
    iters = sorted(df_long["iter"].unique())
    max_idx = int(df_long["idx"].max()) if not df_long.empty else -1

    cols = []
    for kind in ["pred_reward", "pred_constraint", "true_reward", "true_constraint"]:
        for j in range(max_idx + 1):
            cols.append((kind, j))
    wide = pd.DataFrame(index=iters, columns=pd.MultiIndex.from_tuples(cols, names=["kind", "idx"]), dtype=float)
    wide.index.name = "iter"

    for it in iters:
        for j in range(max_idx + 1):
            if j in per_iter_pred_reward.get(it, {}):
                wide.loc[it, ("pred_reward", j)] = per_iter_pred_reward[it][j]
            if j in per_iter_pred_constraint.get(it, {}):
                wide.loc[it, ("pred_constraint", j)] = per_iter_pred_constraint[it][j]
            if j in per_iter_true_reward.get(it, {}):
                wide.loc[it, ("true_reward", j)] = per_iter_true_reward[it][j]
            if j in per_iter_true_constraint.get(it, {}):
                wide.loc[it, ("true_constraint", j)] = per_iter_true_constraint[it][j]

    wide_path = out_dir / "eval_wide.parquet"
    wide.to_parquet(wide_path)
    print(f"[SAVE] {wide_path}")

    return df_long, wide


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate saved samples with AugmentedReward and pred_vs_real")
    ap.add_argument("--root", default="/Users/svlg/MasterThesis/v03_geom/aa_experiments/",
                    help="Root folder that contains experiment dirs (default: your path)")
    ap.add_argument("--experiment", default="0923_2305_al_dipole_energy",
                    help="Experiment name (default matches your example)")
    ap.add_argument("--seed", type=int, default=1, help="Seed suffix in folder name (default: 1)")
    ap.add_argument("--config", default="configs/augmented_lagrangian.yaml", help="Config YAML path")
    ap.add_argument("--outdir", default="eval", help="Subfolder in experiment dir to write outputs")
    args = ap.parse_args()

    eval_saved_samples(
        root=args.root,
        experiment=args.experiment,
        seed=args.seed,
        config_path=args.config,
        outdir=args.outdir,
    )
