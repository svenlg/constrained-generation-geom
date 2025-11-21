import dgl
import torch
import flowmol


def step(model, adj, g_t, t, alpha, alpha_dot, dt, upper_edge_mask, calc_adj=True):
    
    with torch.enable_grad():
        # turn on autograd for the graph
        g_t.ndata["x_t"] = g_t.ndata["x_t"].detach().requires_grad_(True)
        g_t.ndata["a_t"] = g_t.ndata["a_t"].detach().requires_grad_(True)
        g_t.ndata["c_t"] = g_t.ndata["c_t"].detach().requires_grad_(True)
        g_t.edata["e_t"] = g_t.edata["e_t"].detach().requires_grad_(True)

        node_batch_idx = torch.zeros(g_t.num_nodes(), dtype=torch.long)

        # predict the destination of the trajectory given the current time-point
        dst_dict = model.vector_field(
            g_t, 
            t=torch.full((g_t.batch_size,), t, device=g_t.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # take integration step for all features
        v = {}
        v_pred = {}
        
        for feat in ['x', 'a', 'c']:
            x_1 = dst_dict[feat]
            x_t = g_t.ndata[f'{feat}_t']
            if calc_adj:
                adj_t_feat = adj[feat].detach()

            v_pred[feat] = model.vector_field.vector_field(x_t, x_1, alpha[0], alpha_dot[0])

            if calc_adj:
                eps_pred = 2 * v_pred[feat] - alpha_dot[0]/(alpha[0]+dt) * x_t
                g_term = (adj_t_feat * eps_pred).sum()
                v[feat] = torch.autograd.grad(g_term, x_t, retain_graph=True)[0]

        for feat in ['e']:
            x_1 = dst_dict[feat]
            x_t = g_t.edata[f'{feat}_t'][upper_edge_mask]

            if calc_adj:
                adj_t_feat = adj[feat].detach()

            v_pred[feat] = model.vector_field.vector_field(x_t, x_1, alpha[0], alpha_dot[0])
            if calc_adj:
                eps_pred = 2 * v_pred[feat] - alpha_dot[0]/(alpha[0]+dt) * x_t
                g_term = (adj_t_feat * eps_pred).sum()
                v[feat] = torch.autograd.grad(g_term, x_t, retain_graph=False)[0]

    adj_tmh = {}
    if calc_adj:
        for feat in ['x', 'a', 'c', 'e']:
            adj_tmh[feat] = adj[feat].detach() + dt * v[feat].detach()
            adj_tmh[feat].detach()
            v_pred[feat].detach()

    return v_pred, adj_tmh

class LeanAdjointSolverFlow:
    """Solver as per adjoint matching paper."""

    def __init__(self, gen_model: flowmol.FlowMol, grad_reward_fn, device=None):
        self.model = gen_model
        self.interpolant_scheduler = gen_model.interpolant_scheduler
        self.grad_reward_fn = grad_reward_fn
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def solve(self, graph_trajectories, ts):
        """Solving loop."""
        # NOTE: T is the number of steps after the cutoff time
        T = ts.shape[0]
        assert T == len(graph_trajectories)
        # ts: tensor with times bigger than cutoff time (1, 1-dt, 1-2*dt, ..., cutoff_time)
        dt = ts[0] - ts[1]

        alpha_s = self.interpolant_scheduler.alpha_t(ts)
        alpha_dot_s = self.interpolant_scheduler.alpha_t_prime(ts)

        # graph_trajectories: is a list of dgl graphs (graph_trajectory[0] =^= t=1 and graph_trajectory[T-1] =^= t=cutoff_time)
        g1 = graph_trajectories[0] # graphs at t=0
        minus_adj = self.grad_reward_fn(g1)
        if isinstance(minus_adj, torch.Tensor): # returns batched graphs where each feature is the adjoint or tensor
            adj = - torch.cat(minus_adj, dim=0)

        if isinstance(minus_adj, dgl.DGLGraph): 
            adj = {}
            adj['x'] = - minus_adj.ndata['x_t'].clone().detach()
            adj['a'] = - minus_adj.ndata['a_t'].clone().detach()
            adj['c'] = - minus_adj.ndata['c_t'].clone().detach()
            adj['e'] = - minus_adj.edata['e_t'].clone().detach()
            adj['ue_mask'] = minus_adj.edata['ue_mask'].detach().clone()

        trajs_adj = []
        traj_v_pred = []
        upper_edge_mask = g1.edata['ue_mask'] # (edges, 1)
        adj['e'] = adj['e'][upper_edge_mask]
        # trajs_pos = []

        for i in range(1, T):
            t = ts[i]
            g_t = graph_trajectories[i]
            alpha = alpha_s[i]
            alpha_dot = alpha_dot_s[i]
            adj, v_pred = step(
                model = self.model,
                adj = adj, 
                g_t = g_t, 
                t = t, 
                alpha = alpha, 
                alpha_dot = alpha_dot, 
                dt = dt, 
                upper_edge_mask = upper_edge_mask,
                calc_adj=True
            )
            trajs_adj.append(adj)
            traj_v_pred.append(v_pred)
        
        res = {
            't': ts[1:], # (T,)
            'alpha': alpha_s[1:], # (T, 4)
            'alpha_dot': alpha_dot_s[1:], # (T, 4)
            'traj_graph': graph_trajectories[1:], # list of dgl graphs (T,)
            'traj_adj': trajs_adj, # list of dicts with {x, a, c, e} each (T,)
            'traj_v_pred': traj_v_pred, # list of dicts with {x, a, c, e} each (T,)
        }

        assert len(res['traj_adj']) == len(res['traj_v_pred'])
        assert len(res['traj_adj']) == res['t'].shape[0]
        assert len(res['traj_graph']) == res['t'].shape[0]
        assert res['alpha'].shape[0] == res['t'].shape[0] and res['alpha_dot'].shape[0] == res['t'].shape[0]
        assert res['t'].shape[0] == ts.shape[0] - 1
        return res
