import numpy as np
import pandas as pd


POSEBUSTERS_COLS_IN_ORDER = [
    "mol_pred_loaded",        # 0.05
    "sanitization",           # 0.50
    "all_atoms_connected",    # 0.30
    "bond_lengths",           # 0.10
    "bond_angles",            # 0.01
    "internal_steric_clash",  # 0.01
    "aromatic_ring_flatness", # 0.01
    "double_bond_flatness",   # 0.01
    "internal_energy",        # 0.01
]

POSEBUSTERS_WEIGHTS = {
    "mol_pred_loaded": 0.05,
    "sanitization": 0.50,
    "all_atoms_connected": 0.30,
    "bond_lengths": 0.10,
    "bond_angles": 0.01,
    "internal_steric_clash": 0.01,
    "aromatic_ring_flatness": 0.01,
    "double_bond_flatness": 0.01,
    "internal_energy": 0.01,
}

def _coerce_bool(x):
    """Robust truthiness: treat 1/True/'true' as True; 0/False/'false'/NaN as False."""
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true","t","yes","y","1","pass","passed"}:
            return True
        if s in {"false","f","no","n","0","fail","failed"}:
            return False
    # fallback
    return bool(x)

def _default_weights(cols):
    """
    Heavier weight for earlier checks (given order).
    If exactly the standard PoseBusters columns, use a hand-tuned vector.
    Otherwise, make a simple linear decay and normalize.
    """
    if list(cols) == POSEBUSTERS_COLS_IN_ORDER:
        w = np.array([POSEBUSTERS_WEIGHTS[col] for col in POSEBUSTERS_COLS_IN_ORDER], dtype=float)
        w /= w.sum()
        return w
    # generic: linear decay n..1, earlier = larger
    n = len(cols)
    w = np.arange(n, 0, -1, dtype=float)
    w /= w.sum()
    return w

def posebusters_score(df: pd.DataFrame,
                      weights: np.ndarray = None,
                      auto_one_on_disconnect: bool = True) -> pd.Series:
    """
    Compute a 0-1 badness score per row.
      - 0 = all tests pass; 1 = none pass.
      - Optional hard rule: if all_atoms_connected == False → 1.0.
      - Otherwise: weighted sum of failed checks (earlier columns weigh more).
    Returns a pandas Series aligned to df.index.
    """
    # Use only columns present; preserve order
    cols = [c for c in POSEBUSTERS_COLS_IN_ORDER if c in df.columns]
    if not cols:
        raise ValueError("None of the expected PoseBusters columns are present.")

    W = _default_weights(cols) if weights is None else np.asarray(weights, dtype=float)
    if W.shape[0] != len(cols):
        raise ValueError(f"weights length ({W.shape[0]}) must match number of used columns ({len(cols)}).")
    W = W / W.sum()  # ensure “all fail” sums to 1.0

    # Coerce to boolean pass/fail
    passes = df[cols].map(_coerce_bool)

    # Hard rule: not sanitized = 1.0
    if auto_one_on_disconnect and "all_atoms_connected" in cols:
        rule = ~passes["sanitization"]
    else:
        rule = pd.Series(False, index=df.index)

    # Weighted failure score
    fails = ~passes
    score = (fails * W).sum(axis=1).astype(float)

    # Apply hard rule
    score = score.where(~rule, 1.0)

    # clip for safety
    return score.clip(0.0, 1.0)

