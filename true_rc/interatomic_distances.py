import dgl
import torch


def extract_moldata_from_graph(g: dgl.DGLGraph):

    # extract node-level features
    positions = g.ndata['x_t']

    # get bond types and atom indices for every edge, convert types from simplex to integer
    bond_types = g.edata['e_t'].argmax(dim=1)
    bond_types[bond_types == 5] = 0 # set masked bonds to 0
    bond_src_idxs, bond_dst_idxs = g.edges()

    # get just the upper triangle of the adjacency matrix
    upper_edge_mask = g.edata['ue_mask']
    bond_types = bond_types[upper_edge_mask]
    bond_src_idxs = bond_src_idxs[upper_edge_mask]
    bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

    # get only non-zero bond types
    bond_mask = bond_types != 0
    bond_src_idxs = bond_src_idxs[bond_mask]
    bond_dst_idxs = bond_dst_idxs[bond_mask]

    return positions, bond_src_idxs, bond_dst_idxs


def bond_distance(g: dgl.DGLGraph) -> torch.Tensor:

    tmp = [graph for graph in dgl.unbatch(g)]
    rtn_tensor= []
    for tmp_g in tmp:
        positions, bond_src_idxs, bond_dst_idxs = extract_moldata_from_graph(tmp_g)

        bond_vecs = positions[bond_src_idxs] - positions[bond_dst_idxs]
        bond_dists = torch.linalg.norm(bond_vecs, dim=-1)

        rtn_tensor.append(bond_dists.mean().unsqueeze(0))

    return torch.cat(rtn_tensor)



def dist_constraint(g: dgl.DGLGraph) -> torch.Tensor:

    positions = g.ndata['x_t']
    low_tri_inds = torch.tril_indices(positions.shape[0], positions.shape[0], offset=-1)
    conf_dists = torch.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    conf_dists = conf_dists[low_tri_inds[0], low_tri_inds[1]]

    return conf_dists.mean()



################################

import torch
import dgl
from typing import List, Tuple
from collections import defaultdict, deque


# Example atom type map (indices must match g.ndata["a_1"] one-hot)
ATOM_TYPE_MAP: List[str] = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
IDX_H = ATOM_TYPE_MAP.index("H")


def extract_moldata_from_graph(
    g: dgl.DGLGraph,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract:
      positions:    (N, 3)  float tensor
      atom_type_idx:(N,)    long tensor, indices into ATOM_TYPE_MAP
      bond_types:   (E,)    long tensor in {1..4} (SINGLE, DOUBLE, TRIPLE, AROMATIC)
      bond_src_idxs:(E,)    long tensor, source node indices of real bonds (upper triangle)
      bond_dst_idxs:(E,)    long tensor, dest node indices of real bonds (upper triangle)
    """
    # Node positions
    positions = g.ndata["x_1"]                    # (N, 3)

    # Atom types as indices
    atom_type_idx = g.ndata["a_1"].argmax(dim=1)  # (N,)

    # Edge-level bond type prediction (one-hot over 6 classes)
    bond_types = g.edata["e_1"].argmax(dim=1)     # (E_all,)
    bond_types[bond_types == 5] = 0               # masked -> 0

    bond_src_idxs, bond_dst_idxs = g.edges()      # (E_all,)

    # Only take edges in the upper triangle
    upper_edge_mask = g.edata["ue_mask"]          # (E_all,)
    bond_types = bond_types[upper_edge_mask]
    bond_src_idxs = bond_src_idxs[upper_edge_mask]
    bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

    # Only keep real bonds (types 1..4)
    bond_mask = bond_types != 0
    bond_types = bond_types[bond_mask]            # (E,)
    bond_src_idxs = bond_src_idxs[bond_mask]      # (E,)
    bond_dst_idxs = bond_dst_idxs[bond_mask]      # (E,)

    return positions, atom_type_idx, bond_types, bond_src_idxs, bond_dst_idxs


def ideal_bond_lengths(
    bond_types: torch.Tensor,
    atom_type_idx: torch.Tensor,
    bond_src_idxs: torch.Tensor,
    bond_dst_idxs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ideal bond length per bond based on:
      - bond type (single/double/triple/aromatic)
      - whether the bond involves hydrogen.

    bond_types:   (E,) with values in {1:SINGLE, 2:DOUBLE, 3:TRIPLE, 4:AROMATIC}
    atom_type_idx:(N,)
    bond_src_idxs:(E,)
    bond_dst_idxs:(E,)

    Returns:
      ideal_len: (E,) tensor.
    """
    device = bond_types.device
    IDX_H = ATOM_TYPE_MAP.index("H")

    is_H = (atom_type_idx == IDX_H)                # (N,)
    has_H = is_H[bond_src_idxs] | is_H[bond_dst_idxs]  # (E,)

    # Reasonable example values in Å (tune to your liking)
    # Indexing with bond_types directly.
    ideal_len_heavy = torch.tensor(
        [
            1.5,   # 0: unused / masked (fallback)
            1.54,  # 1: SINGLE     (C–C single)
            1.34,  # 2: DOUBLE     (C=C)
            1.20,  # 3: TRIPLE     (C≡C)
            1.40,  # 4: AROMATIC   (C_aro–C_aro)
            1.5,   # 5: masked (not used)
        ],
        device=device,
    )

    ideal_len_H = torch.tensor(
        [
            1.0,   # 0: unused
            1.09,  # 1: SINGLE     (C–H)
            1.05,  # 2: DOUBLE     (rare)
            1.00,  # 3: TRIPLE     (rare)
            1.08,  # 4: AROMATIC   (C_aro–H)
            1.0,   # 5: masked
        ],
        device=device,
    )

    len_heavy = ideal_len_heavy[bond_types]  # (E,)
    len_H     = ideal_len_H[bond_types]      # (E,)

    ideal_len = torch.where(has_H, len_H, len_heavy)  # (E,)

    return ideal_len


def get_components(
    num_nodes: int,
    bond_src_idxs: torch.Tensor,
    bond_dst_idxs: torch.Tensor,
):
    """
    Connected components based *only* on real bonds.
    Returns:
      components: list of components, each a list of Python ints (node indices).
    """
    from collections import defaultdict, deque

    adj = defaultdict(list)
    src_list = bond_src_idxs.tolist()
    dst_list = bond_dst_idxs.tolist()

    for s, d in zip(src_list, dst_list):
        adj[s].append(d)
        adj[d].append(s)

    visited = [False] * num_nodes
    components = []

    for start in range(num_nodes):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        comp = [start]
        while q:
            v = q.popleft()
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
                    comp.append(nb)
        components.append(comp)

    return components


def connectivity_matrix_and_loss(
    g: dgl.DGLGraph,
    iso_penalty_value: float = 3.0,
    w_matrix: float = 1.0,
    w_comp_geom: float = 0.00,
    target_com_sep: float = 4.0,
    return_M: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a connectivity-aware matrix M and a scalar loss.

    M is an (N, N) matrix:
      - For bonded pairs (i, j):
            M_ij = ||x_i - x_j|| - ideal_bond_length(i, j)
        where ideal length depends on atom types and bond order.

      - For non-bonded pairs:
            M_ij = 0

      - For isolated atoms (no bonds at all), diagonals:
            M_ii = iso_penalty_value  (e.g. 3)

    The scalar loss combines:
      - matrix_term    = mean(|M|)        (differentiable w.r.t. positions)
      - comp_geom_term = penalty on large separation between components' COMs
                         (differentiable w.r.t. positions)
      - comp_count_term = penalty on having more than one connected component
                          (NOT differentiable w.r.t. positions, but still useful)

    Returns:
      M:    (N, N) tensor
      loss: scalar tensor
    """
    positions, atom_type_idx, bond_types, bond_src_idxs, bond_dst_idxs = \
        extract_moldata_from_graph(g)

    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    # --- pairwise distance matrix (conf_dists) ---
    # (N, N), symmetric, 0 on diagonal, differentiable w.r.t. positions
    conf_dists = torch.cdist(positions, positions)

    # --- build bond mask & ideal length matrix ---
    bond_mask_mat = torch.zeros((N, N), device=device, dtype=dtype)
    ideal_len_mat = torch.zeros((N, N), device=device, dtype=dtype)

    if bond_types.numel() > 0:
        ideal_len = ideal_bond_lengths(
            bond_types,
            atom_type_idx,
            bond_src_idxs,
            bond_dst_idxs,
        )  # (E,)

        # fill symmetric entries
        bond_mask_mat[bond_src_idxs, bond_dst_idxs] = 1.0
        bond_mask_mat[bond_dst_idxs, bond_src_idxs] = 1.0

        ideal_len_mat[bond_src_idxs, bond_dst_idxs] = ideal_len
        ideal_len_mat[bond_dst_idxs, bond_src_idxs] = ideal_len

    # --- bonded deviation matrix ---
    bonded_dev = (conf_dists - ideal_len_mat) * bond_mask_mat  # (N, N)

    # --- isolated atoms: diagonal penalty = iso_penalty_value ---
    degrees = bond_mask_mat.sum(dim=1)          # (N,)
    iso_mask = (degrees == 0).float()           # (N,)
    iso_diag = torch.diag(iso_mask * iso_penalty_value)  # (N, N)

    # final matrix
    M = bonded_dev + iso_diag  # (N, N)

    # --- matrix-based loss term (differentiable w.r.t. positions) ---
    matrix_term = M.abs().mean()

    # --- component geometry term (differentiable w.r.t. positions) ---
    components = get_components(N, bond_src_idxs, bond_dst_idxs)
    if len(components) > 1:
        coms = []
        for comp in components:
            idx = torch.tensor(comp, device=device, dtype=torch.long)
            coms.append(positions[idx].mean(dim=0))
        coms = torch.stack(coms, dim=0)            # (C, 3)

        com_dists = torch.cdist(coms, coms)        # (C, C)
        i, j = torch.tril_indices(coms.size(0), coms.size(0), offset=-1, device=device)
        pair_dists = com_dists[i, j]               # (C_pairs,)

        comp_geom_term = ((pair_dists - target_com_sep).clamp_min(0.0) ** 2).mean()
    else:
        comp_geom_term = torch.zeros((), device=device, dtype=dtype)

    # --- combine ---
    loss = w_matrix * matrix_term + w_comp_geom * comp_geom_term

    if loss >= 0.1:
        loss = torch.tensor(0.1, device=device, dtype=dtype, requires_grad=True)
    
    return M, loss if return_M else loss