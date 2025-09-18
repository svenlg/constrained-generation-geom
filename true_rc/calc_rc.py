import time
import pandas as pd
from true_rc.posebuster_scorer import posebusters_score
from true_rc.xtb_calc import compute_xtb
from typing import List

import logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

def get_rc_properties(
        rd_mols: List,
        reward: str = "dipole",
        constraint: str = "score",
        verbose: bool = False,
        return_dict: bool = False,
        ) -> List[dict]:
    
    if constraint == "score" :
        # Get PoseBusters Feedbac
        tmp_time = time.time()
        df_scores = posebusters_score(rd_mols, disconnected=True)
        if verbose:
            print(f"PoseBusters time: {time.time() - tmp_time:.2f} seconds", flush=True)
    else:
        df_scores = pd.DataFrame({"score": [0.0]*len(rd_mols)})

    if reward == "dipole":
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

    rmse = lambda x, y: ((x - y) ** 2).mean() ** 0.5
    mse = lambda x, y: ((x - y) ** 2).mean()
    mae = lambda x, y: (abs(x - y)).mean()
    r2 = lambda x, y: 1 - ((x - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    
    df_true = get_rc_properties(rd_mols, reward=reward, constraint=constraint)

    true_reward = df_true[reward].to_numpy()
    pred_reward = pred_dict["reward"].flatten()
    reward_dct = {
        "reward/rmse": rmse(true_reward, pred_reward),
        "reward/mse": mse(true_reward, pred_reward),
        "reward/mae": mae(true_reward, pred_reward),
        "reward/r2": r2(true_reward, pred_reward),
        "true/reward": true_reward.mean(),
        "true/reward_std": true_reward.std(),
    }

    if constraint == "score":
        true_constraint = df_true[constraint].to_numpy()
        pred_constraint = pred_dict["constraint"].flatten()

        constraint_dict = {
            "constraint/rmse": rmse(true_constraint, pred_constraint),
            "constraint/mse": mse(true_constraint, pred_constraint),
            "constraint/mae": mae(true_constraint, pred_constraint),
            "constraint/r2": r2(true_constraint, pred_constraint),
            "true/constraint": true_constraint.mean(),
            "true/constraint_std": true_constraint.std(),
        }
    else:
        true_constraint = None
        constraint_dict = {}

    return_dict = {**reward_dct, **constraint_dict}

    return return_dict, true_reward, true_constraint