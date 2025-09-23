import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

# ---------- Utilities ----------
class CatProbEmbedding(nn.Module):
    """
    Embeds categorical distributions (probs) over K classes into R^D.
    """
    def __init__(self, num_classes: int, emb_dim: int, tau: float = 1.0, hard: bool = True):
        super().__init__()
        self.linear = nn.Linear(num_classes, emb_dim, bias=False)  # unused, for compatibility
        self.tau = tau
        self.hard = hard

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        # gumbel_softmax expects logits; use log-probs for stability
        logits = torch.log(probs.clamp(min=eps))
        onehot = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)  # (N, K)
        return self.linear(onehot)


class GaussianRBF(nn.Module):
    """Simple fixed RBF expansion of distances^2 for scalar invariants."""
    def __init__(self, num_k: int = 16, cutoff: float = 10.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_k)
        widths = torch.full((num_k,), (cutoff / num_k))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, dist_sq: torch.Tensor) -> torch.Tensor:
        # dist_sq: (..., 1)
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        x = (dist - self.centers) / (self.widths + 1e-12)
        return torch.exp(-0.5 * x**2)  # (..., K)


# ---------- EGNN (SO(3)/E(n) equivariant) block ----------
class EGNNBlock(nn.Module):
    """
    EGNN-style block:
      m_ij = phi_m(h_i, h_j, e_ij, ||r_ij||^2)
      x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
      h_i' = LN( h_i + phi_h([h_i, sum_j psi_m(h_i, h_j, e_ij, ||r||^2)]) )
    """
    def __init__(self, hidden_dim: int, rbf_k: int = 16):
        super().__init__()
        self.rbf = GaussianRBF(num_k=rbf_k, cutoff=10.0)

        # scalar edge message
        self.phi_m = nn.Sequential(
            nn.Linear(2 * hidden_dim + rbf_k + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # vector field scalar (reduces H -> 1 gate)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # feature message to be summed then used in update
        self.psi_m = nn.Sequential(
            nn.Linear(2 * hidden_dim + rbf_k + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.upd_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_h = nn.LayerNorm(hidden_dim)

    def edge_udf(self, edges):
        h_i = edges.dst["h"]
        h_j = edges.src["h"]
        x_i = edges.dst["x"]                      # (E, 3)
        x_j = edges.src["x"]
        e_ij = edges.data["e_emb"]                # (E, H)

        rij = x_i - x_j                           # (E, 3)
        dist_sq = (rij * rij).sum(-1, keepdim=True)  # (E, 1)
        rbf = self.rbf(dist_sq)                   # (E, K)

        cat_ij = torch.cat([h_i, h_j, rbf, e_ij], dim=-1)  # (E, 2H+K+H)
        m_ij = self.phi_m(cat_ij)                 # (E, H)
        vf_gate = self.phi_x(m_ij)                # (E, 1)
        dx_ij = rij * vf_gate                     # (E, 3)

        pm_ij = self.psi_m(cat_ij)                # (E, H) feature message
        return {"dx": dx_ij, "pm": pm_ij}

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, x: torch.Tensor):
        with g.local_scope():
            g.ndata["h"] = h
            g.ndata["x"] = x
            g.apply_edges(self.edge_udf)
            # aggregate dx
            g.update_all(fn.copy_e("dx", "m_dx"), fn.sum("m_dx", "dx_sum"))
            # aggregate pm
            g.update_all(fn.copy_e("pm", "m_pm"), fn.sum("m_pm", "pm_sum"))
            dx_sum = g.ndata["dx_sum"]            # (N, 3)
            pm_sum = g.ndata["pm_sum"]            # (N, H)

            x_next = x + dx_sum
            h_next = self.norm_h(h + self.upd_h(torch.cat([h, pm_sum], dim=-1)))
            return h_next, x_next


# ---------- Full Model ----------
class EGNN(nn.Module):
    """
    A single model that can run in:
      - edge-aware (non-equivariant) mode, or
      - SO(3)/E(n)-equivariant mode (EGNN blocks).

    Inputs (stored in DGLGraph):
      g.ndata['a_t'] : (N, Ka) atom-type probs
      g.ndata['c_t'] : (N, Kc) charge-class probs  (if you have a scalar charge instead, map it beforehand)
      g.ndata['x_t'] : (N, 3)  coordinates
      g.edata['e_t'] : (E, Ke) bond-type probs
    """
    def __init__(
        self,
        property: str,
        num_atom_types: int,
        num_charge_classes: int,
        num_bond_types: int,
        hidden_dim: int = 256,
        depth: int = 6,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
        rbf_k: int = 16,
    ):
        super().__init__()
        self.property = property

        # ---- Embeddings for categorical distributions ----
        d_each = hidden_dim // 3
        self.atom_emb = CatProbEmbedding(num_atom_types, d_each, gumbel_tau, gumbel_hard)
        self.charge_emb = CatProbEmbedding(num_charge_classes, d_each, gumbel_tau, gumbel_hard)
        self.bond_emb = CatProbEmbedding(num_bond_types, d_each, gumbel_tau, gumbel_hard)

        # Fuse node pieces to hidden_dim
        self.fuse_nodes = nn.Linear(3 * d_each, hidden_dim)

        # Edge embedding lift to hidden_dim for use in blocks
        self.edge_lift = nn.Linear(d_each, hidden_dim)

        # Stacks
        self.blocks = nn.ModuleList([EGNNBlock(hidden_dim, rbf_k=rbf_k) for _ in range(depth)])

        # Head
        self.head = nn.Linear(hidden_dim, 1)
        if self.property == "dipole":
            self.output = nn.Softplus()
            self.tau = 1.0
        elif self.property == "score":
            self.output = nn.Sigmoid()
            self.tau = 2.0
        else:
            self.output = nn.Identity()
            self.tau = 1.0

    def encode_nodes(self, g):
        A = g.ndata["a_t"]   # probs
        C = g.ndata["c_t"]   # probs
        X = g.ndata["x_t"]   # (N, 3)

        a_h = self.atom_emb(A)
        c_h = self.charge_emb(C)
        x_h = torch.zeros_like(a_h)  # placeholder to keep dims consistent

        h0 = torch.cat([a_h, c_h, x_h], dim=-1)
        h0 = self.fuse_nodes(h0)
        return h0

    def encode_edges(self, g):
        E = g.edata["e_t"]     # probs
        e_h = self.bond_emb(E) # (E, d_each)
        return self.edge_lift(e_h)  # (E, H)

    def forward(self, g: dgl.DGLGraph):
        with g.local_scope():
            # Node/edge encodings
            h = self.encode_nodes(g)               # (N, H)
            g.edata["e_emb"] = self.encode_edges(g)  # (E, H)

            x = g.ndata["x_t"]
            for blk in self.blocks:
                h, x = blk(g, h, x)
            g.ndata["h"] = h

            # Graph readout
            g.ndata["h_final"] = g.ndata["h"]
            hg = dgl.readout_nodes(g, "h_final", op="mean")
            out = self.head(hg) / self.tau
            return self.output(out)

