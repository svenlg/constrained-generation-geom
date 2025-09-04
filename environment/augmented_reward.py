import copy
import torch
import dgl
from typing import Union

class AugmentedReward:
    def __init__(
            self, 
            reward_fn: callable, # if torch module make sure to call .to(self.device)
            constraint_fn: callable, # if torch module make sure to call .to(self.device)
            alpha: float,
            bound: float,
            device: torch.device = None,
        ):

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if isinstance(reward_fn, torch.nn.Module):
            self.reward_fn = reward_fn.to(self.device)
            self.reward_fn.eval()
        if isinstance(constraint_fn, torch.nn.Module):
            self.constraint_fn = constraint_fn.to(self.device)
            self.constraint_fn.eval()

        self.alpha = copy.deepcopy(float(alpha))
        self.bound = copy.deepcopy(float(bound))

        # Initialize lambda and rho
        self.lambda_ = 0.0
        self.rho_ = 1.0

    def set_lambda_rho(self, lambda_: float, rho_: float):
        self.lambda_ = copy.deepcopy(float(lambda_))
        self.rho_ = copy.deepcopy(float(rho_))

    def get_reward_constraint(self):
        cur_reward = self.tmp_reward.clone().detach().mean().cpu().item()
        cur_constraint_violations = torch.sum(self.tmp_constraint > self.bound).detach().cpu().item() / len(self.tmp_constraint)
        cur_constraint = self.tmp_constraint.clone().detach().mean().cpu().item()
        return cur_reward, cur_constraint, cur_constraint_violations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # We want to maximize reward and minimize constraint
        # Note lambda < 0, rho > 0
        self.tmp_reward = self.reward_fn(x)
        reward = self.tmp_reward.mean()
        self.tmp_constraint = self.constraint_fn(x)
        constraint = self.tmp_constraint.mean()
        tmp_lambda = torch.ones_like(constraint, device=constraint.device) * self.lambda_
        tmp_rho = torch.ones_like(constraint, device=constraint.device) * self.rho_
        tmp_zero = torch.zeros_like(constraint, device=constraint.device)
        self.tmp_total = (
                reward - tmp_rho/2 * torch.max(tmp_zero, constraint - tmp_lambda/tmp_rho)**2 
            ).mean()
        return self.alpha * self.tmp_total

    def grad_augmented_reward_fn(self, x: Union[torch.Tensor, dgl.DGLGraph]) -> torch.Tensor:

        if isinstance(self.reward_fn, torch.nn.Module):
            self.reward_fn.eval()
        if isinstance(self.constraint_fn, torch.nn.Module):
            self.constraint_fn.eval()

        with torch.enable_grad():
            x = x.to(self.device)
            if isinstance(x, dgl.DGLGraph):
                x.ndata['x_t'] = x.ndata['x_t'].clone().detach().requires_grad_(True)
                x.ndata['a_t'] = x.ndata['a_t'].clone().detach().requires_grad_(True)
                x.ndata['c_t'] = x.ndata['c_t'].clone().detach().requires_grad_(True)
                x.edata['e_t'] = x.edata['e_t'].clone().detach().requires_grad_(True)
            if isinstance(x, torch.Tensor):
                x = x.clone().detach().requires_grad_(True)

            tmp_augmented_reward = self(x)

            tmp_augmented_reward.backward()

            if isinstance(x, dgl.DGLGraph):
                grad = dgl.graph((x.edges()[0], x.edges()[1]), num_nodes=x.num_nodes(), device=x.device)
                grad.set_batch_num_nodes(x.batch_num_nodes())
                grad.set_batch_num_edges(x.batch_num_edges())

                grad.ndata['x_t'] = x.ndata['x_t'].grad.clone().detach().requires_grad_(False)
                grad.ndata['a_t'] = x.ndata['a_t'].grad.clone().detach().requires_grad_(False)
                grad.ndata['c_t'] = x.ndata['c_t'].grad.clone().detach().requires_grad_(False)

                grad.edata['e_t'] = x.edata['e_t'].grad.clone().detach().requires_grad_(False)
                grad.edata['ue_mask'] = x.edata['ue_mask'].detach().clone()

            if isinstance(x, torch.Tensor):
                grad = x.grad.clone().detach().requires_grad_(False)

        return grad

    def get_statistics(self) -> dict:
        total_reward = self.tmp_total.clone().detach().cpu().item()
        reward = self.tmp_reward.clone().detach().mean().cpu().item()
        constraint = self.tmp_constraint.clone().detach()
        constraint_violations = torch.sum(constraint > self.bound).detach().cpu().item() / len(constraint)
        constraint = torch.mean(constraint).detach().cpu().item()
        return {
            f"reward": reward,
            f"constraint": constraint,
            f"total_reward": total_reward,
            f"constraint_violations": constraint_violations,
        }
    