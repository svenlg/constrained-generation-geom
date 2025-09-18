import dgl
import torch

def inter_atomic_distances(positions):
    low_tri_inds = torch.tril_indices(positions.shape[0], positions.shape[0], offset=-1)
    conf_dists = torch.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    conf_dists = conf_dists[low_tri_inds[0], low_tri_inds[1]]
    return conf_dists

def geometry_constraint(g: dgl.DGLGraph):
    min_dists = []
    for mol in dgl.unbatch(g):
        conf_dists = inter_atomic_distances(mol.ndata['x_t'])
        min_dist = torch.min(conf_dists)
        min_dists.append(min_dist)
    min_dists = torch.stack(min_dists)
    return min_dists


def dist_constraint(positions, d_min: float = 0.9):
    conf_dists = inter_atomic_distances(positions)
    return torch.sigmoid(conf_dists - d_min)