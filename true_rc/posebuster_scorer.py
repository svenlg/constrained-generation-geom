import numpy as np
import pandas as pd
from posebusters import PoseBusters
from typing import List

import logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

ALL_COLS = ['mol_pred_loaded', 
            'sanitization', 'inchi_convertible', 'all_atoms_connected', 
            'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness', 
            'non-aromatic_ring_non-flatness', 'double_bond_flatness',
            'internal_energy']

COLS_TO_KEEP = ['sanitization', 'inchi_convertible', 'all_atoms_connected', 'internal_energy']

COLS_TO_FIX = ['internal_energy']

POSEBUSTERS_WEIGHTS = {
    "sanitization": 0.75,
    "inchi_convertible": 0.03,
    "all_atoms_connected": 0.75,
    "internal_energy": 0.05,
}

def posebusters_score(rd_mols: List, disconnected:bool=False) -> pd.Series:
    """
    Returns a pandas Series aligned to df.index.
    """
    buster = PoseBusters(config="mol")
    df_scores = buster.bust(rd_mols)
    df_scores = df_scores.reset_index(drop=True)
    df_scores = df_scores[COLS_TO_KEEP]
    df_scores[COLS_TO_FIX] = df_scores[COLS_TO_FIX].notna()

    df = df_scores.copy()

    # Weights
    if disconnected:
        POSEBUSTERS_WEIGHTS["all_atoms_connected"] = 1.0
    
    W = np.array([v for k,v in POSEBUSTERS_WEIGHTS.items() if k in df.columns])
    if W.shape[0] != len(df.columns):
        raise ValueError(f"weights length ({W.shape[0]}) must match number of used columns ({len(df.columns)}).")

    # Weighted failure score
    fails = ~df
    score = (fails * W).sum(axis=1).astype(float)

    # Cut at 1.0
    score = score.clip(0.0, 1.0)

    df_scores['score'] = score.to_numpy()

    return df_scores

