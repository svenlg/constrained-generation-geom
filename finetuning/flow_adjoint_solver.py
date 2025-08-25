import torch
import flowmol
from flowmol.analysis.molecule_builder import SampledMolecule

class LeanAdjointSolverFlow:
    """Solver as per adjoint matching paper."""

    def __init__(self, gen_model: flowmol.FlowMol, grad_reward_fn, device=None):
        self.model = gen_model
        self.interpolant_scheduler = gen_model.interpolant_scheduler
        self.grad_reward_fn = grad_reward_fn
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def step(self, adj, g_t, t, alpha, alpha_dot, dt, upper_edge_mask):
        adj_t = adj.detach() # detach to avoid gradients

        with torch.enable_grad():
            # turn on autograd for the graph
            g_t.ndata["x_t"] = g_t.ndata["x_t"].detach().requires_grad_(True)
            node_batch_idx = torch.zeros(g_t.num_nodes(), dtype=torch.long)

            # predict the destination of the trajectory given the current time-point
            dst_dict = self.model.vector_field(
                g_t, 
                t=torch.full((g_t.batch_size,), t, device=g_t.device),
                node_batch_idx=node_batch_idx,
                upper_edge_mask=upper_edge_mask,
                apply_softmax=True,
                remove_com=True
            )
            # take integration step for positions
            x_1 = dst_dict['x']
            x_t = g_t.ndata['x_t']

            v_pred = self.model.vector_field.vector_field(x_t, x_1, alpha[0], alpha_dot[0])

            eps_pred = 2 * v_pred - alpha_dot[0]/(alpha[0]+dt) * x_t
            g_term = (adj_t * eps_pred).sum()
            v = torch.autograd.grad(g_term, x_t, retain_graph=False)[0]
        assert v.shape == x_t.shape
        adj_tmh = adj_t + dt * v
        # if adj_tmh.isnan().any():
        #     print("Loss is NaN, skipping step") 
        return adj_tmh.detach(), v_pred.detach()

    def solve(self, graph_trajectories, ts):
        """Solving loop."""
        T = ts.shape[0]
        assert T == len(graph_trajectories)
        # ts: tensor of shape (num_ts,) (0, dt, 2dt, ..., 1-dt, 1)
        dt = ts[1] - ts[0]
        ts = ts.flip(0) # flip to go from 1 to 0

        alpha_s = self.interpolant_scheduler.alpha_t(ts)
        alpha_dot_s = self.interpolant_scheduler.alpha_t_prime(ts)

        # graph_trajectories: is a list of dgl graphs (graph_trajectory[0] =^= t=0 and graph_trajectory[T-1] =^= t=1
        graph_trajectories = graph_trajectories[::-1] # flip to go from 1 to 0      
        g1 = graph_trajectories[0] # graphs at t=0  
        minus_adj = self.grad_reward_fn(g1) # returns a list of tensors (representing the adjoint)
        adj = - torch.cat(minus_adj, dim=0) # flip the sign of the adjoint and concatenate (shape (nodes(node1 + node2 + ...), 3))

        row_mask = ~torch.isnan(adj).all(dim=1)
        # if adj.isnan().any():
        #     print("Loss is NaN, skipping step")
        # assert torch.all(torch.isfinite(adj))
        # assert x1.shape == adj.shape

        trajs_adj = []
        traj_v_pred = []
        upper_edge_mask = g1.edata['ue_mask'] # (edges, 1)
        # trajs_pos = []

        for i in range(1, T):
            t = ts[i]
            g_t = graph_trajectories[i]
            alpha = alpha_s[i]
            alpha_dot = alpha_dot_s[i]
            adj, v_pred = self.step(adj=adj, g_t=g_t, t=t, alpha=alpha, alpha_dot=alpha_dot, dt=dt, upper_edge_mask=upper_edge_mask)
        #     trajs_pos.append(g_t.ndata['x_t'].detach())
            trajs_adj.append(adj.detach())
            traj_v_pred.append(v_pred.detach())

            tmp_row_mask = ~torch.isnan(adj).all(dim=1)
            row_mask = torch.logical_and(row_mask, tmp_row_mask)
            # if adj.isnan().any():
            #     print("Loss is NaN, skipping step") 
        
        res = {
                't': ts[1:], # (T,)
                'alpha': alpha_s[1:, 0], # (T,)
                'alpha_dot': alpha_dot_s[1:, 0], # (T,)
                'traj_graph': graph_trajectories[1:], # list of dgl graphs (T, )
            #     'traj_x': torch.stack(trajs_pos), # (T, nodes, 3)'
                'traj_adj': torch.stack(trajs_adj), # (T, nodes, 3)
                'traj_v_pred': torch.stack(traj_v_pred), # (T, nodes, 3)
                'row_mask': row_mask, # (nodes,)
            }

        # assert res['traj_adj'].shape == res['traj_x'].shape
        assert res['traj_adj'].shape == res['traj_v_pred'].shape
        assert res['traj_adj'].shape[0] == res['t'].shape[0]
        assert len(res['traj_graph']) == res['t'].shape[0]
        assert res['alpha'].shape[0] == res['t'].shape[0] and res['alpha_dot'].shape[0] == res['t'].shape[0]
        assert res['t'].shape[0] == ts.shape[0] - 1
        return res
