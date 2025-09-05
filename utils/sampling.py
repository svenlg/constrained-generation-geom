from typing import Optional
from omegaconf import OmegaConf
import torch
import flowmol
import copy
import dgl


MAX_ALLOWED_ATOMS = 75  # upper bound for molecule size (can be upto 182 for GEOM)
MIN_ALLOWED_ATOMS = 30  # lower bound for molecule size (can be as low as 3 for GEOM)


def sampling(
    config: OmegaConf,
    model: flowmol.FlowMol,
    device: torch.device,
    min_num_atoms: Optional[int] = None,
    max_num_atoms: Optional[int] = None,
    n_atoms: Optional[int] = None,
):
    """
    Sampling helper with two modes:

    1) Fixed-size sampling (when `n_atoms` is set):
       -> uses `model.sample(...)` to draw molecules with exactly `n_atoms`.

    2) Random-size sampling (default):
       -> uses `model.sample_random_sizes(...)`.
       -> If either `min_num_atoms` or `max_num_atoms` is provided, both must be
          provided and pass validity checks, then they’re forwarded as bounds.

    Validity checks when min/max are set:
      - min_num_atoms < max_num_atoms
      - max_num_atoms > 0
      - min_num_atoms < 183
    """
    model.to(device)

    n_atoms_provided = n_atoms is not None

    # --- Mode 1: fixed-size sampling if n_atoms is specified ---
    if n_atoms_provided:
        if not isinstance(n_atoms, int) or (n_atoms <= MIN_ALLOWED_ATOMS-1) or (n_atoms >= MAX_ALLOWED_ATOMS+1):
            raise ValueError(f"n_atoms must be a positive int between {MIN_ALLOWED_ATOMS} and {MAX_ALLOWED_ATOMS}, got {n_atoms!r}")
        
        new_molecules = model.sample(
            n_atoms=torch.tensor([n_atoms] * config.num_samples),
            n_timesteps=config.num_integration_steps + 1,
            device=device,
        )

    # --- Mode 2: random-size sampling (optionally bounded) ---
    else:
        # Fill missing bound with maximum allowed
        if min_num_atoms is None:
            min_num_atoms = MIN_ALLOWED_ATOMS
        if max_num_atoms is None:
            max_num_atoms = MAX_ALLOWED_ATOMS

        # Type & value checks
        if not (isinstance(min_num_atoms, int) and isinstance(max_num_atoms, int)):
            raise ValueError("`min_num_atoms` and `max_num_atoms` must be ints.")
        if (min_num_atoms >= max_num_atoms) or (min_num_atoms < MIN_ALLOWED_ATOMS) or (max_num_atoms > MAX_ALLOWED_ATOMS):
            raise ValueError(
                f"Invalid size bounds: min_num_atoms={min_num_atoms}, max_num_atoms={max_num_atoms}. "
                f"Must satisfy: min_num_atoms < max_num_atoms, max_num_atoms >= {MIN_ALLOWED_ATOMS}, min_num_atoms <= {MAX_ALLOWED_ATOMS}."
            )

        new_molecules = model.sample_random_sizes(
            n_molecules=config.num_samples,
            n_timesteps=config.num_integration_steps + 1,
            device=device,
            min_num_atoms=min_num_atoms,
            max_num_atoms=max_num_atoms,
        )

    dgl_mols, rd_mols = [], []
    for mol in new_molecules:
        dgl_mols.append(copy.deepcopy(mol.g))
        rd_mols.append(copy.deepcopy(mol.rdkit_mol))
    
    dgl_mols = dgl.batch(dgl_mols).to(device)
    return dgl_mols, rd_mols
