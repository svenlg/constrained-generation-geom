import dgl
import torch
from functools import partial
from omegaconf import OmegaConf



def inter_atomic_distances(positions):
    low_tri_inds = torch.tril_indices(positions.shape[0], positions.shape[0], offset=-1)
    conf_dists = torch.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    conf_dists = conf_dists[low_tri_inds[0], low_tri_inds[1]]
    return conf_dists

# def geometry_constraint(g: dgl.DGLGraph):
#     dists = []
#     for mol in dgl.unbatch(g):
#         conf_dists = inter_atomic_distances(mol.ndata['x_t'])
#         min_dist = torch.min(conf_dists)
#         dists.append(min_dist)
#     dists = torch.stack(dists)
#     return dists

def geometry_constraint(g: dgl.DGLGraph, reduction: str = "min", bound: float = 0.9):
    dists = []
    for mol in dgl.unbatch(g):
        conf_dists = inter_atomic_distances(mol.ndata['x_t'])
        if reduction == "min":
            dist = torch.min(conf_dists)
        elif reduction == "mean":
            dist = torch.mean(conf_dists)
        elif reduction == "relu":
            dist = torch.relu(bound - conf_dists).sum()
        elif reduction == "relu_mean":
            dist = torch.relu(bound - conf_dists)
            dist_mask = dist > 0
            if dist_mask.sum() > 0:
                dist = torch.mean(dist[dist_mask])
            else:
                dist = torch.tensor(0.0, device=dist.device, requires_grad=True)
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        dists.append(dist)
    dists = torch.stack(dists)
    return dists

def wrapped_geometry_constraint(config: OmegaConf):
    reduction = config.get("reduction", "min")
    bound = config.get("bound", 0.9)
    return partial(geometry_constraint, reduction=reduction, bound=bound)

def dist_constraint(positions, d_min: float = 0.9):
    conf_dists = inter_atomic_distances(positions)
    return torch.sigmoid(conf_dists - d_min)