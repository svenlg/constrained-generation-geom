import numpy as np
import copy
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import flowmol
from torch.utils.data import Dataset, ConcatDataset
from finetuning.flow_adjoint_solver import LeanAdjointSolverFlow


class AMDataset(Dataset):
    def __init__(self, solver_info):
        solver_info = self.detach_all(solver_info)
        self.t = solver_info['t'] # (T,)
        self.sigma_t = solver_info['sigma_t'] # (T,)
        self.alpha = solver_info['alpha'] # (T,)
        self.alpha_dot = solver_info['alpha_dot']# (T,)
        self.traj_g = solver_info['traj_graph'] # list of dgl graphs (T,)
        self.traj_adj = solver_info['traj_adj'] # (T, nodes, 3)
        self.traj_v_base = solver_info['traj_v_pred'] # (T, nodes, 3)
        self.row_mask = solver_info['row_mask'] # (nodes,)

        self.T = self.t.size(0) # T = number of time steps
        self.bs = 1 # len(self.traj_g[0].batch_num_nodes())
        
    def __len__(self):
        return self.bs

    def __getitem__(self, idx):
        return {
            't': self.t,
            'sigma_t': self.sigma_t,
            'alpha': self.alpha,
            'alpha_dot': self.alpha_dot,
            'traj_graph': self.traj_g,
            # 'traj_x': self.traj_x,
            'traj_adj': self.traj_adj,
            'traj_v_base': self.traj_v_base,
            'row_mask': self.row_mask,
        }
    
    def detach_all(self, solver_info):
        for key, value in solver_info.items():
            if isinstance(value, torch.Tensor):
                solver_info[key] = value.detach()
            elif isinstance(value, list):
                for g in value:
                    for k in g.ndata.keys():
                        if isinstance(g.ndata[k], torch.Tensor):
                            g.ndata[k] = g.ndata[k].detach()
                    for k in g.edata.keys():
                        if isinstance(g.edata[k], torch.Tensor):
                            g.edata[k] = g.edata[k].detach()
        return solver_info


def create_timestep_subset(total_steps, final_percent=0.25, sample_percent=0.25):
    """
    Create a subset of time-steps for efficient computation. (See paper Appendix G2)
    
    Args:
        total_steps (int): Total number of time-steps in the process
        final_percent (float): Percentage of final steps to always include
        sample_percent (float): Percentage of additional steps to sample
    
    Returns:
        np.ndarray: Sorted array of selected timestep indices
    """
    # Calculate the number of steps for each section
    final_steps_count = int(total_steps * final_percent)
    sample_steps_count = int(total_steps * sample_percent)
    
    # Always take the first final_percent steps (assuming highest index is 0)
    final_samples = np.arange(final_steps_count)
    
    # Sample additional steps without replacement from the remaining steps
    # Exclude the steps already in final_samples
    remaining_steps = np.setdiff1d(
        np.arange(total_steps), 
        final_samples
    )
    
    # Sample additional steps
    additional_samples = np.random.choice(
        remaining_steps, 
        size=sample_steps_count, 
        replace=False
    )
    
    # Combine and sort the samples
    combined_samples = np.sort(np.concatenate([final_samples, additional_samples]))
    
    return combined_samples


def adj_matching_loss(v_base, v_fine, adj, sigma):
    """Adjoint matching loss for FM"""
    diff = v_fine - v_base
    term_diff = (2 / sigma[:,None,None]) * diff
    term_adj = sigma[:,None,None] * adj
    term_difference = term_diff - term_adj
    term_difference = torch.sum(torch.square(term_difference), dim=[1, 2])
    loss = torch.mean(term_difference)
    return loss 


class AdjointMatchingFinetuningTrainerFlowMol:
    def __init__(self,
            config: OmegaConf,
            model: flowmol.FlowMol,
            base_model: flowmol.FlowMol,
            grad_reward_fn: callable,
            device: torch.device = None,
            verbose: bool = False,
        ):
        # Config
        self.config = config
        self.sampling_config = config.sampling
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.max_nodes = config.get("max_nodes", 210)

        # Clip
        self.clip_grad_norm = config.get("clip_grad_norm", 1e5)
        self.clip_loss = config.get("clip_loss", 0.5)

        # Models
        self.fine_model = model
        self.base_model = base_model
        self.fine_model.to(self.device)
        self.base_model.to(self.device)

        # Reward (Gradient of the reward function(al))
        self.grad_reward_fn = grad_reward_fn

        # Setup optimizer
        self.configure_optimizers()

    def configure_optimizers(self):
        if hasattr(self, 'optimizer'):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)

    def get_model(self):
        return self.fine_model

    def sample_trajectories(self):
        self.fine_model.eval()
        _, graph_trajectories = self.fine_model.sample_random_sizes(
            n_molecules = self.config.batch_size,
            n_timesteps = self.sampling_config.num_integration_steps + 1,     
            sampler_type = self.sampling_config.sampler_type,
            device = self.device,
            keep_intermediate_graphs = True,
        ) 
        ts = torch.linspace(0.0, 1.0, self.sampling_config.num_integration_steps + 1).to(self.device)
        sigmas = self.fine_model.vector_field.sigmas
        sigmas = torch.stack(sigmas, dim=0).to(self.device)
        return graph_trajectories, ts, sigmas

    def generate_dataset(self):
        """Sample dataset for training based on adjoint ODE and sampled trajectories."""
        datasets = []

        # run in eval mode
        self.fine_model.eval()
        self.base_model.eval()

        solver = LeanAdjointSolverFlow(self.base_model, self.grad_reward_fn)

        iterations = self.sampling_config.num_samples // self.config.batch_size
        for i in range(iterations):
            with torch.no_grad():
                while True:
                    graph_trajectories, ts, sigmas = self.sample_trajectories()
                    if graph_trajectories[0].num_nodes() <= self.max_nodes:
                        break  
                    print(f"Rerolling: got {graph_trajectories[0].num_nodes()} nodes (> {self.max_nodes})")

            # graph_trajectories is a list of the intermediate graphs
            solver_info = solver.solve(graph_trajectories=graph_trajectories, ts=ts)
            # add sigma_t to solver_info
            solver_info['sigma_t'] = sigmas
            
            if (~solver_info['row_mask']).all():
                print("Row mask is all True, skipping sample")
                continue
            dataset = AMDataset(solver_info=solver_info)
            datasets.append(dataset)

        if len(datasets) == 0:
            return None
        dataset = ConcatDataset(datasets)
        return dataset

    def train_step(self, sample):
        """Training step."""

        ts = sample['t'].to(self.device)
        sigmas = sample['sigma_t'].to(self.device)
        alpha = sample['alpha'].to(self.device)
        alpha_dot = sample['alpha_dot'].to(self.device)
        traj_g = [g.to(self.device) for g in sample['traj_graph']]
        traj_adj = sample['traj_adj'].to(self.device)
        # traj_x = sample['traj_x'].to(self.device)
        traj_v_base = sample['traj_v_base'].to(self.device)
        row_mask = sample['row_mask'].to(self.device)

        # Get index for time steps to calculate adjoint matching loss
        idxs = create_timestep_subset(ts.shape[0])

        v_base = []
        v_fine = []
        adj = []
        sigma = []
        
        for idx in idxs:
            t = ts[idx]
            sigma_t = sigmas[idx]
            adj_t = traj_adj[idx]
            v_base_t = traj_v_base[idx]
            g_base_t = traj_g[idx]
            alpha_t = alpha[idx]
            alpha_dot_t = alpha_dot[idx]

            node_batch_idx = torch.zeros(g_base_t.num_nodes(), dtype=torch.long)

            # predict the destination of the trajectory given the current time-point
            dst_dict = self.fine_model.vector_field(
                g_base_t, 
                t=torch.full((g_base_t.batch_size,), t, device=g_base_t.device),
                node_batch_idx=node_batch_idx,
                upper_edge_mask=g_base_t.edata['ue_mask'],
                apply_softmax=True,
                remove_com=True
            )
            # take integration step for positions
            x_1 = dst_dict['x']
            x_t = g_base_t.ndata['x_t']

            v_fine_t = self.fine_model.vector_field.vector_field(x_t, x_1, alpha_t, alpha_dot_t)
            
            v_base_t = v_base_t[row_mask]
            v_fine_t = v_fine_t[row_mask]
            adj_t = adj_t[row_mask]
            
            v_base.append(v_base_t)
            v_fine.append(v_fine_t)
            adj.append(adj_t)
            sigma.append(sigma_t)
        
        # stack the tensors
        v_base = torch.stack(v_base, dim=0)
        v_fine = torch.stack(v_fine, dim=0)
        sigma = torch.stack(sigma, dim=0)
        adj = torch.stack(adj, dim=0)
        
        loss = adj_matching_loss(
            v_base=v_base,
            v_fine=v_fine,
            adj=adj,
            sigma=sigma,
        )

        if loss.isnan().any():
            return torch.tensor(float("inf"), device=self.device)
        
        # step optimizer
        self.optimizer.zero_grad()

        # self.fine_model.zero_grad()
        loss.backward(retain_graph=False)

        # loss clapping
        if self.clip_loss > 0.0:
            loss = torch.clamp(loss, min=0.0, max=self.clip_loss)

        if self.verbose and self.config.clip_grad_norm > 0.0:
            total_norm = 0
            for p in self.fine_model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Before Clipping Norm: {total_norm:.6f}")

        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)

        self.optimizer.step()

        return loss

    def finetune(self, dataset, steps=None):
        """Finetuning the model."""

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        c = 0
        total_loss = 0

        self.fine_model.to(self.device)
        self.fine_model.train()
        
        self.optimizer.zero_grad()

        # iterate over the dataset
        if steps is not None:
            idxs = np.random.permutation(dataset.__len__())[:steps]
        else:
            idxs = np.random.permutation(dataset.__len__())
        
        for idx in idxs:
            # print('outer', idx)
            sample = dataset[idx]
            loss = self.train_step(sample).item()
            total_loss = total_loss + loss
            c+=1 
            # print('loss', loss)

        del dataset
        return total_loss / c
