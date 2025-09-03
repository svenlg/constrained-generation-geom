import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import dgl


class GNN(nn.Module):
    def __init__(self, property: str, node_feats: int, edge_feats: int, hidden_dim=256, depth=6):
        super().__init__()

        self.property = property
        self.in_conv = dgl.nn.GraphConv(node_feats, hidden_dim)
        self.edge_linear = nn.Linear(edge_feats, hidden_dim)

        blocks = []
        for _ in range(depth):
            blocks.append(ResBlock(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Linear(hidden_dim, 1)

        if self.property == "dipole" or self.property == "dipole_zero":
            self.output = nn.Softplus() # dipole is non-negative
            self.tau = 1.0
        elif self.property == "score": # score is between 0 and 1
            self.output = nn.Sigmoid()
            self.tau = 2.0
        else:
            self.output = nn.Identity()
            self.tau = 1.0

        # nn.init.zeros_(self.head.weight)
        # nn.init.zeros_(self.head.bias)

    def forward(self, g: dgl.DGLGraph) -> Tensor:
        with g.local_scope():
            h = torch.cat([
                g.ndata["a_t"],
                g.ndata["c_t"],
                g.ndata["x_t"],
            ], dim=-1)
            h = self.in_conv(g, h)

            # Add edge feature to initial representation
            g.edata["e"] = self.edge_linear(g.edata["e_t"])

            g.update_all(
                dgl.function.copy_e("e", "m"),    # message: copy each edge's "e" to mailbox "m"
                dgl.function.mean("m", "e_mean")  # reduce: mean over mailbox → store in ndata['e_mean']
            )
            h += g.ndata["e_mean"]

            for block in self.blocks:
                h = block(g, h)

            g.ndata["h"] = h
            h = dgl.readout_nodes(g, "h", op="mean")
            return self.output(self.head(h) / self.tau)


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)

        self.conv2 = dgl.nn.GraphConv(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, g: dgl.DGLGraph, x: Tensor) -> Tensor:
        h = self.conv1(g, x)
        h = self.norm1(h)
        h = F.silu(h)

        h = self.conv2(g, h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)

