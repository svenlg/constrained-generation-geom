import dgl
import torch


def extract_moldata_from_graph(g: dgl.DGLGraph):

    # extract node-level features
    # positions = g.ndata['x_1']
    positions = g.ndata['x_t']

    # get bond types and atom indices for every edge, convert types from simplex to integer
    # bond_types = g.edata['e_1'].argmax(dim=1)
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



def dist_constraint(positions: dgl.DGLGraph) -> torch.Tensor:

    low_tri_inds = torch.tril_indices(positions.shape[0], positions.shape[0], offset=-1)
    conf_dists = torch.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    conf_dists = conf_dists[low_tri_inds[0], low_tri_inds[1]]

    return conf_dists.mean()