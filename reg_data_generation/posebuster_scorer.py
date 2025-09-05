import numpy as np
import pandas as pd
import rdkit
from posebusters import PoseBusters
from typing import List

ALL_COLS = ['mol_pred_loaded', 
            'sanitization', 'inchi_convertible', 'all_atoms_connected', 
            'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness', 
            'non-aromatic_ring_non-flatness', 'double_bond_flatness',
            'internal_energy']

COLS_TO_KEEP = ['sanitization', 'inchi_convertible', 'all_atoms_connected', 'internal_energy']

COLS_TO_FIX = ['internal_energy']

POSEBUSTERS_COLS_IN_ORDER = [
    'sanitization',        # 0.75
    'inchi_convertible',   # 0.03
    'all_atoms_connected', # 0.75
    'internal_energy',     # 0.05
]

POSEBUSTERS_WEIGHTS = {
    "sanitization": 0.75,
    "inchi_convertible": 0.03,
    "all_atoms_connected": 0.75,
    "internal_energy": 0.05,
}

def posebusters_score(rd_mols: list) -> pd.Series:
    """
    Returns a pandas Series aligned to df.index.
    """
    buster = PoseBusters(config="mol")
    df_scores = buster.bust(rd_mols)
    df_scores = df_scores.reset_index(drop=True)
    df_scores = df_scores[COLS_TO_KEEP]

    df = df.copy()
    # Use only columns present; preserve order
    df = df[POSEBUSTERS_COLS_IN_ORDER]
    if df.empty:
        raise ValueError("None of the expected PoseBusters columns are present.")

    df[COLS_TO_FIX] = df[COLS_TO_FIX].notna()

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

