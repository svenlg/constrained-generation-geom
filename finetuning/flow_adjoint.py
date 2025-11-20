import numpy as np
import copy
from tqdm import tqdm
from typing import Optional
from omegaconf import OmegaConf

import dgl
import torch
import flowmol
from torch.utils.data import Dataset, ConcatDataset
from finetuning.flow_adjoint_solver import LeanAdjointSolverFlow, step


MAX_ALLOWED_ATOMS = 75  # upper bound for molecule size (can be upto 182 for GEOM)
MIN_ALLOWED_ATOMS = 30  # lower bound for molecule size (can be as low as 3 for GEOM)


#### UTILITIES ####
def check_and_get_atom_numbers(config: OmegaConf):
    max_nodes = config.get("max_nodes", 210)
    if config.sampling.n_atoms is not None:
        if not isinstance(config.sampling.n_atoms, int):
            raise ValueError(f"n_atoms must be a positive int, got {config.sampling.n_atoms}")
        if config.sampling.n_atoms < MIN_ALLOWED_ATOMS or config.sampling.n_atoms > MAX_ALLOWED_ATOMS:
            raise ValueError(f"n_atoms must be between {MIN_ALLOWED_ATOMS} and {MAX_ALLOWED_ATOMS}, got {config.sampling.n_atoms}")
        if config.sampling.n_atoms * config.batch_size > max_nodes:
            raise ValueError(f"n_atoms * batch_size = {config.sampling.n_atoms * config.batch_size} > max_nodes ({max_nodes}). Please decrease n_atoms or increase max_nodes.")
        n_atoms = config.sampling.n_atoms
    else:
        n_atoms = None

    if config.sampling.min_num_atoms is not None:
        if not isinstance(config.sampling.min_num_atoms, int):
            raise ValueError(f"min_num_atoms must be a positive int, got {config.sampling.min_num_atoms}")
        if config.sampling.min_num_atoms < MIN_ALLOWED_ATOMS or config.sampling.min_num_atoms > MAX_ALLOWED_ATOMS:
            raise ValueError(f"min_num_atoms must be between {MIN_ALLOWED_ATOMS} and {MAX_ALLOWED_ATOMS}, got {config.sampling.min_num_atoms}")
        if config.sampling.min_num_atoms * config.batch_size > max_nodes:
            raise ValueError(f"min_num_atoms * batch_size = {config.sampling.min_num_atoms * config.batch_size} > max_nodes ({max_nodes}). Please decrease min_num_atoms or increase max_nodes.")
        min_num_atoms = config.sampling.min_num_atoms if n_atoms is None else None
    else:
        min_num_atoms = MIN_ALLOWED_ATOMS

    if config.sampling.max_num_atoms is not None:
        if not isinstance(config.sampling.max_num_atoms, int):
            raise ValueError(f"max_num_atoms must be a positive int, got {config.sampling.max_num_atoms}")
        if config.sampling.max_num_atoms < MIN_ALLOWED_ATOMS or config.sampling.max_num_atoms > MAX_ALLOWED_ATOMS:
            raise ValueError(f"max_num_atoms must be between {MIN_ALLOWED_ATOMS} and {MAX_ALLOWED_ATOMS}, got {config.sampling.max_num_atoms}")
        max_num_atoms = config.sampling.max_num_atoms if n_atoms is None else None
    else:
        max_num_atoms = MAX_ALLOWED_ATOMS

    return max_nodes, n_atoms, min_num_atoms, max_num_atoms


def create_timestep_subset(
        total_steps, 
        final_percent: float = 0.25, 
        sample_percent: float = 0.25, 
        samples_for_sumapproximation: int = None,
    ) -> np.ndarray:
    """
    Create a subset of time-steps for efficient computation. (See AM-Paper Appendix G2)
    
    Args:
        total_steps (int): Total number of time-steps in the process
        final_percent (float): Percentage of final steps to always include
        sample_percent (float): Percentage of additional steps to sample
        samples_for_sumapproximation (int, optional): Maximum number of steps to include in the final subset. If None, includes all selected steps.
    
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
    combined_samples = np.sort(np.concatenate([final_samples, additional_samples]))

    # Take at most samples_for_sumapproximation samples
    if samples_for_sumapproximation is not None and samples_for_sumapproximation < combined_samples.shape[0]:
        combined_samples = np.random.choice(
            combined_samples, 
            size=samples_for_sumapproximation, 
            replace=False
        )

    # Sort the idx before returning
    return np.sort(combined_samples)


#### DATASET ####
class AMDataset(Dataset):
    def __init__(self, solver_info):
        # NOTE: T is the number of steps after the cutoff time
        solver_info = self.detach_all(solver_info)
        self.t = solver_info['t'] # (T,)
        self.sigma_t = solver_info['sigma_t'] # (T,)
        self.alpha = solver_info['alpha'] # (T,)
        self.alpha_dot = solver_info['alpha_dot']# (T,)
        self.traj_g = solver_info['traj_graph'] # list of dgl graphs (T,)
        self.traj_adj = solver_info['traj_adj'] # list of dicts with {x, a, c, e} each (T,)
        self.traj_v_base = solver_info['traj_v_pred'] # list of dicts with {x, a, c, e} each (T,)

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
            'traj_adj': self.traj_adj,
            'traj_v_base': self.traj_v_base,
        }
    
    def detach_all(self, solver_info):
        for key, value in solver_info.items():
            if isinstance(value, torch.Tensor):
                solver_info[key] = value.detach()
            elif isinstance(value, list):
                if isinstance(value[0], dgl.DGLGraph):
                    for g in value:
                        for k in g.ndata.keys():
                            if isinstance(g.ndata[k], torch.Tensor):
                                g.ndata[k] = g.ndata[k].detach()
                        for k in g.edata.keys():
                            if isinstance(g.edata[k], torch.Tensor):
                                g.edata[k] = g.edata[k].detach()
                if isinstance(value[0], dict):
                    for dict_i in range(len(value)):
                        for k, v in value[dict_i].items():
                            if isinstance(v, torch.Tensor):
                                value[dict_i][k] = v.detach()
        return solver_info


#### LOSS ####
def adj_matching_loss(v_base, v_fine, adj, sigma, LCT):
    """Adjoint matching loss for FM"""
    eps = 1e-12
    diff = v_fine - v_base
    sig = sigma.view(-1, 1, 1)  
    term_diff = (2.0 / (sig + eps)) * diff
    term_adj = sig * adj
    term = term_diff + term_adj
    per_t = (term ** 2).sum(dim=[1, 2])
    clipped = torch.clamp(per_t, max=LCT) if LCT > 0.0 else per_t
    loss = clipped.sum()
    return loss

loss_weights = {'a': 0.4, 'c': 1.0, 'e': 2.0, 'x': 3.0}
def adj_matching_loss_list_of_dicts(v_base, v_fine, adj, sigma, LCT, features=['x', 'a', 'c', 'e']):
    """Adjoint matching loss for FM"""
    eps = 1e-12
    loss = 0.0
    for i, feat in enumerate(features):
        diff = v_fine[feat] - v_base[feat]                  # [T, ..., ...]
        sig = sigma[:, i].view(-1, 1, 1)                    # [T,1,1]
        term_diff = (2.0 / (sig + eps)) * diff              # [T, ..., ...]
        term_adj = sig * adj[feat]                          # [T, ..., ...]
        term = term_diff + term_adj                         # [T, ..., ...]
        per_t = (term ** 2).sum(dim=[1, 2])                 # [T]
        clipped = torch.clamp(per_t, max=LCT) if LCT > 0.0 else per_t  # [T]
        loss = loss + clipped.sum() * loss_weights[feat]
    return loss


#### SAMPLING ####
def sampling(
    config: OmegaConf,
    batch_size: int,
    model: flowmol.FlowMol,
    device: torch.device,
):
    model.to(device)
    n_atoms_provided = config.n_atoms is not None

    # --- Mode 1: fixed-size sampling if n_atoms is specified ---
    if n_atoms_provided:
        _, graph_trajectories = model.sample(
            sampler_type = config.sampler_type,
            n_atoms=torch.tensor([config.n_atoms] * batch_size, device=device),
            n_timesteps=config.num_integration_steps + 1,
            device=device,
            keep_intermediate_graphs = True,
        )

    # --- Mode 2: random-size sampling (optionally bounded) ---
    else:
        _, graph_trajectories = model.sample_random_sizes(
            sampler_type = config.sampler_type,
            n_molecules=batch_size,
            n_timesteps=config.num_integration_steps + 1,
            device=device,
            min_num_atoms=config.min_num_atoms,
            max_num_atoms=config.max_num_atoms,
            keep_intermediate_graphs = True,
        )

    return graph_trajectories


#### TRAINER ####
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

        # Nodes limit
        (
            self.max_nodes,
            self.sampling_config.n_atoms,
            self.sampling_config.min_num_atoms,
            self.sampling_config.max_num_atoms,
        ) = check_and_get_atom_numbers(config)

        # Reward_lambda and LCT and clip_grad_norm
        reward_lambda = config.get("reward_lambda", 1.0)
        lct = config.get("lct", None)
        self.LCT = lct * reward_lambda**2 if lct > 0.0 else 0.0
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)

        # Models
        self.fine_model = model
        self.base_model = base_model
        self.fine_model.to(self.device)
        self.base_model.to(self.device)

        # Reward (Gradient of the reward function(al))
        self.grad_reward_fn = grad_reward_fn
        self.features = config.get("features", ['x', 'a', 'c', 'e'])
        assert type(self.features) == list, f"features must be a list"

        # Engineering tricks:
        self.cutoff_time = config.get("cutoff_time", 0.5) 
        samples_for_sumapproximation = config.get("samples_for_sumapproximation", 10)
        self.samples_for_sumapproximation = min(samples_for_sumapproximation, self.sampling_config.num_integration_steps + 1)
        self.final_percent = config.get("final_percent", 0.25)
        self.sample_percent = config.get("sample_percent", 0.25)

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
            
        graph_trajectories = sampling(
            config = self.sampling_config,
            batch_size = self.config.batch_size,
            model = self.fine_model,
            device = self.device,
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
                    if self.verbose:
                        print(f"Sampled trajectory with {graph_trajectories[0].num_nodes()} nodes and {graph_trajectories[0].num_edges()} edges.")
                    if graph_trajectories[0].num_nodes() <= self.max_nodes:
                        break 
                    if self.verbose:
                        print(f"Rerolling: got {graph_trajectories[0].num_nodes()} nodes (> {self.max_nodes})")
            
            # ts: tensor of shape (num_ts,) (0, dt, 2dt, ..., 1-dt, 1)
            # graph_trajectories: is a list of dgl graphs (graph_trajectory[0] =^= t=0 and graph_trajectory[T-1] =^= t=1
            # flip to go from 1 to 0
            ts = ts.flip(0)
            sigmas = sigmas.flip(0)
            graph_trajectories = graph_trajectories[::-1]

            # Cutoff time for efficiency
            if self.cutoff_time > 0.0:
                cutoff_idx = int((1 - self.cutoff_time) * (ts.shape[0] - 1))
                ts = ts[:cutoff_idx + 1]
                graph_trajectories = graph_trajectories[:cutoff_idx + 1]
                sigmas = sigmas[:cutoff_idx] # sigmas has one less entry than ts

            # graph_trajectories is a list of the intermediate graphs
            solver_info = solver.solve(graph_trajectories=graph_trajectories, ts=ts)
            # add sigma_t to solver_info
            solver_info['sigma_t'] = sigmas
            
            dataset = AMDataset(solver_info=solver_info)
            datasets.append(dataset)

        if len(datasets) == 0:
            return None
        dataset = ConcatDataset(datasets)
        return dataset

    def push_to_device(self, sample):
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.to(self.device)
            elif isinstance(value, list):
                if isinstance(value[0], dgl.DGLGraph):
                    for i in range(len(value)):
                        value[i] = value[i].to(self.device)
                if isinstance(value[0], dict):
                    for dict_i in range(len(value)):
                        for k, v in value[dict_i].items():
                            if isinstance(v, torch.Tensor):
                                value[dict_i][k] = v.to(self.device)
        return sample

    def train_step(self, sample):
        """Training step."""

        sample = self.push_to_device(sample)
        ts = sample['t']
        sigmas = sample['sigma_t']
        alpha = sample['alpha']
        alpha_dot = sample['alpha_dot']
        traj_g = sample['traj_graph']
        traj_adj = sample['traj_adj']
        traj_v_base = sample['traj_v_base']

        # Get index for time steps to calculate adjoint matching loss
        idxs = create_timestep_subset(ts.shape[0], self.final_percent, self.sample_percent, self.samples_for_sumapproximation)

        v_base = []
        v_fine = []
        adj = []
        sigma = []

        dt = ts[0] - ts[1]
        
        for idx in idxs:
            t = ts[idx]
            sigma_t = sigmas[idx]
            adj_t = traj_adj[idx]
            v_base_t = traj_v_base[idx]
            g_base_t = traj_g[idx]
            alpha_t = alpha[idx]
            alpha_dot_t = alpha_dot[idx]

            v_fine_t, _ = step(
                model = self.fine_model,
                adj = None,
                g_t = g_base_t, 
                t = t, 
                alpha = alpha_t, 
                alpha_dot = alpha_dot_t, 
                dt = dt,
                upper_edge_mask = g_base_t.edata['ue_mask'],
                calc_adj=False
            )
            
            v_base.append(v_base_t)
            v_fine.append(v_fine_t)
            adj.append(adj_t)
            sigma.append(sigma_t)
        
        assert len(v_base) == len(v_fine) == len(adj) == len(sigma)

        # stack each list of dicts to a dict of feature tensors
        # v_base = {feat: torch.stack([v_base[i][feat] for i in range(len(v_base))], dim=0) for feat in ['x', 'a', 'c', 'e']}
        # v_fine = {feat: torch.stack([v_fine[i][feat] for i in range(len(v_fine))], dim=0) for feat in ['x', 'a', 'c', 'e']}
        # adj = {feat: torch.stack([adj[i][feat] for i in range(len(adj))], dim=0) for feat in ['x', 'a', 'c', 'e']}
        v_base = {feat: torch.stack([v_base[i][feat] for i in range(len(v_base))], dim=0) for feat in self.features}
        v_fine = {feat: torch.stack([v_fine[i][feat] for i in range(len(v_fine))], dim=0) for feat in self.features}
        adj = {feat: torch.stack([adj[i][feat] for i in range(len(adj))], dim=0) for feat in self.features}
        sigma = torch.stack(sigma, dim=0)

        loss = adj_matching_loss_list_of_dicts(
            v_base=v_base,
            v_fine=v_fine,
            adj=adj,
            sigma=sigma,
            LCT=self.LCT,
            features=self.features
        )

        if loss.isnan().any():
            return torch.tensor(float("inf"), device=self.device)
        
        # step optimizer
        self.optimizer.zero_grad()

        # self.fine_model.zero_grad()
        loss.backward(retain_graph=False)

        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)

        self.optimizer.step()

        return loss

    def finetune(self, dataset, steps=None):
        """Finetuning the model."""
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
            sample = dataset[idx]
            loss = self.train_step(sample).item()
            total_loss = total_loss + loss
            c+=1 

        del dataset
        return total_loss / c

