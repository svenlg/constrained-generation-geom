import torch

def dist_constraint(positions, d_min: float = 0.9):
    low_tri_inds = torch.tril_indices(positions.shape[0], positions.shape[0], offset=-1)
    conf_dists = torch.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    conf_dists = conf_dists[low_tri_inds[0], low_tri_inds[1]]

    return torch.sigmoid(conf_dists - d_min)