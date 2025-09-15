import time
import pandas as pd
from true_rc.posebuster_scorer import posebusters_score
from true_rc.xtb_calc import compute_xtb
from typing import List

import logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

def get_rc_properties(rd_mols: List, verbose: bool = False, return_dict: bool = False) -> List[dict]:
    # Get PoseBusters Feedback
    tmp_time = time.time()
    df_scores = posebusters_score(rd_mols, disconnected=True)
    if verbose:
        print(f"PoseBusters time: {time.time() - tmp_time:.2f} seconds", flush=True)

    # XTB-Calulations
    tmp_time = time.time()
    properties = []
    for tmp_mol in rd_mols:
        rtn_dict = compute_xtb(tmp_mol, "rdkit", verbose)
        properties.append(rtn_dict)
    if verbose:
        print(f"XTB time: {time.time() - tmp_time:.2f} seconds", flush=True)

    df_props = pd.DataFrame.from_records(properties)
    df = pd.concat([df_scores, df_props], axis=1)

    return df if not return_dict else df.to_dict(orient="records")

def pred_vs_real(rd_mols: List, pred_dict: dict, reward: str, constraint: str) -> pd.DataFrame:
    # Compare the DataFrame with the reference DataFrame
    df_true = get_rc_properties(rd_mols)
    true_reward = df_true[reward].to_numpy()
    true_constraint = df_true[constraint].to_numpy()

    pred_reward = pred_dict["reward"].flatten()
    pred_constraint = pred_dict["constraint"].flatten()
    
    rmse = lambda x, y: ((x - y) ** 2).mean() ** 0.5
    mse = lambda x, y: ((x - y) ** 2).mean()
    mae = lambda x, y: (abs(x - y)).mean()
    r2 = lambda x, y: 1 - ((x - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    return_dict = {
        "reward/rmse": rmse(true_reward, pred_reward),
        "reward/mse": mse(true_reward, pred_reward),
        "reward/mae": mae(true_reward, pred_reward),
        "reward/r2": r2(true_reward, pred_reward),
        "constraint/rmse": rmse(true_constraint, pred_constraint),
        "constraint/mse": mse(true_constraint, pred_constraint),
        "constraint/mae": mae(true_constraint, pred_constraint),
        "constraint/r2": r2(true_constraint, pred_constraint),
        "true/reward": true_reward.mean(),
        "true/reward_std": true_reward.std(),
        "true/constraint": true_constraint.mean(),
        "true/constraint_std": true_constraint.std(),
    }

    return return_dict, true_reward, true_constraint