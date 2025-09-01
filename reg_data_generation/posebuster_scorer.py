import numpy as np
import pandas as pd


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

def posebusters_score(df: pd.DataFrame) -> pd.Series:
    """
    Returns a pandas Series aligned to df.index.
    """
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

    return score

