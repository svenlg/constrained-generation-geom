import time
import pandas as pd
from true_rc.posebuster_scorer import posebusters_score
from true_rc.xtb_calc import compute_xtb
from true_rc.sascore import get_sacore
from typing import List
import dgl
from true_rc.interatomic_distances import bond_distance

import logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

def get_rc_properties(
        rd_mols: List,
        dgl_mols: dgl.DGLGraph,
        reward: str = "dipole",
        constraint: str = "score",
        verbose: bool = False,
        return_dict: bool = False,
        ) -> List[dict]:
    
    ##### Possible Constraint Calculations
    if constraint == "score":
        # Get PoseBusters Feedback
        tmp_time = time.time()
        df_scores = posebusters_score(rd_mols, disconnected=True)
        if verbose:
            print(f"PoseBusters time: {time.time() - tmp_time:.2f} seconds", flush=True)
    elif constraint == "sascore":
        # Get SAScore Feedback
        tmp_time = time.time()
        df_scores = get_sacore(rd_mols)
        if verbose:
            print(f"SAScore time: {time.time() - tmp_time:.2f} seconds", flush=True)
    else:
        df_scores = pd.DataFrame({"not_used": [0.0]*len(rd_mols)})

    # Interatomic Distance Calculations
    tmp_time = time.time()
    tmp_interatomic_distances = bond_distance(dgl_mols).tolist()
    interatomic_distances = [{"interatomic_distances": tmp} for tmp in tmp_interatomic_distances]
    if verbose:
        print(f"Interatomic Distance time: {time.time() - tmp_time:.2f} seconds", flush=True)
    df_interatomic_distance = pd.DataFrame.from_records(interatomic_distances)

    # XTB-Calulations
    tmp_time = time.time()
    properties = []
    for tmp_mol in rd_mols:
        rtn_dict = compute_xtb(tmp_mol, "rdkit", verbose)
        properties.append(rtn_dict)
    if verbose:
        print(f"XTB time: {time.time() - tmp_time:.2f} seconds", flush=True)
    df_xtb = pd.DataFrame.from_records(properties)
    
    df = pd.concat([df_scores, df_interatomic_distance, df_xtb], axis=1)
    return df if not return_dict else df.to_dict(orient="records")


def pred_vs_real(rd_mols: List, dgl_mols: dgl.DGLGraph, pred_dict: dict, reward: str, constraint: str) -> pd.DataFrame:
    # Compare the DataFrame with the reference DataFrame

    rmse = lambda x, y: ((x - y) ** 2).mean() ** 0.5
    mse = lambda x, y: ((x - y) ** 2).mean()
    mae = lambda x, y: (abs(x - y)).mean()    
    df_true = get_rc_properties(rd_mols, dgl_mols, reward=reward, constraint=constraint)

    true_reward = df_true[reward].to_numpy()
    pred_reward = pred_dict["reward"].flatten()
    reward_dct = {
        "reward/rmse": rmse(true_reward, pred_reward),
        "reward/mse": mse(true_reward, pred_reward),
        "reward/mae": mae(true_reward, pred_reward),
        "true/reward": true_reward.mean(),
        "true/reward_std": true_reward.std(),
    }

    true_constraint = df_true[constraint].to_numpy()
    pred_constraint = pred_dict["constraint"].flatten()

    constraint_dict = {
        "constraint/rmse": rmse(true_constraint, pred_constraint),
        "constraint/mse": mse(true_constraint, pred_constraint),
        "constraint/mae": mae(true_constraint, pred_constraint),
        "true/constraint": true_constraint.mean(),
        "true/constraint_std": true_constraint.std(),
    }
    
    return_dict = {**reward_dct, **constraint_dict}

    df_true_avg = df_true.mean().to_dict()
    for key, value in df_true_avg.items():
        return_dict[f"feat/{key}"] = value

    return return_dict, true_reward, true_constraint