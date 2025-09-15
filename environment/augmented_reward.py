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

        # For logging
        self.last_grad_norm_full = None
        self.last_grad_norm_reward = None
        self.last_grad_norm_constraint = None

    def _stack_and_norm(self, tensors) -> float:
        # tensors: iterable of tensors or None
        flats = [t.reshape(-1) for t in tensors if t is not None]
        if not flats:
            return 0.0
        v = torch.cat(flats)
        # keep on device, but return a CPU float
        return torch.linalg.vector_norm(v).detach().cpu().item()

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
        self.tmp_constraint = self.constraint_fn(x)
        tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
        tmp_rho = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
        tmp_zero = torch.zeros_like(self.tmp_constraint, device=self.tmp_constraint.device)
        tmp_bound = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.bound
        self.tmp_total = (
                self.tmp_reward - tmp_rho/2 * torch.max(tmp_zero, self.tmp_constraint - tmp_bound - tmp_lambda/tmp_rho)**2
            ).mean()
        return self.alpha * self.tmp_total

    # def __call__(self, x: torch.Tensor) -> torch.Tensor:
    #     # We want to maximize reward and minimize constraint
    #     # Note lambda < 0, rho > 0
    #     self.tmp_reward = self.reward_fn(x)
    #     reward = self.tmp_reward.mean()
    #     self.tmp_constraint = self.constraint_fn(x)
    #     constraint = self.tmp_constraint.mean()
    #     tmp_lambda = torch.ones_like(constraint, device=constraint.device) * self.lambda_
    #     tmp_rho = torch.ones_like(constraint, device=constraint.device) * self.rho_
    #     tmp_zero = torch.zeros_like(constraint, device=constraint.device)
    #     tmp_bound = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.bound
    #     self.tmp_total = (
    #             reward - tmp_rho/2 * torch.max(tmp_zero, constraint - tmp_bound - tmp_lambda/tmp_rho)**2
    #         ).mean()
    #     return self.alpha * self.tmp_total

    # def grad_augmented_reward_fn(self, x: Union[torch.Tensor, dgl.DGLGraph]) -> torch.Tensor:

    #     if isinstance(self.reward_fn, torch.nn.Module):
    #         self.reward_fn.eval()
    #     if isinstance(self.constraint_fn, torch.nn.Module):
    #         self.constraint_fn.eval()

    #     with torch.enable_grad():
    #         x = x.to(self.device)
    #         if isinstance(x, dgl.DGLGraph):
    #             x.ndata['x_t'] = x.ndata['x_t'].clone().detach().requires_grad_(True)
    #             x.ndata['a_t'] = x.ndata['a_t'].clone().detach().requires_grad_(True)
    #             x.ndata['c_t'] = x.ndata['c_t'].clone().detach().requires_grad_(True)
    #             x.edata['e_t'] = x.edata['e_t'].clone().detach().requires_grad_(True)
    #         if isinstance(x, torch.Tensor):
    #             x = x.clone().detach().requires_grad_(True)

    #         tmp_augmented_reward = self(x)

    #         tmp_augmented_reward.backward()

    #         if isinstance(x, dgl.DGLGraph):
    #             grad = dgl.graph((x.edges()[0], x.edges()[1]), num_nodes=x.num_nodes(), device=x.device)
    #             grad.set_batch_num_nodes(x.batch_num_nodes())
    #             grad.set_batch_num_edges(x.batch_num_edges())

    #             grad.ndata['x_t'] = x.ndata['x_t'].grad.clone().detach().requires_grad_(False)
    #             grad.ndata['a_t'] = x.ndata['a_t'].grad.clone().detach().requires_grad_(False)
    #             grad.ndata['c_t'] = x.ndata['c_t'].grad.clone().detach().requires_grad_(False)

    #             grad.edata['e_t'] = x.edata['e_t'].grad.clone().detach().requires_grad_(False)
    #             grad.edata['ue_mask'] = x.edata['ue_mask'].detach().clone()

    #         if isinstance(x, torch.Tensor):
    #             grad = x.grad.clone().detach().requires_grad_(False)

    #     return grad

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

            # forward: builds self.tmp_reward, self.tmp_constraint, self.tmp_total
            tmp_augmented_reward = self(x)

            # ------- NEW: set up leaf inputs for autograd.grad -------
            if isinstance(x, dgl.DGLGraph):
                inputs = [x.ndata['x_t'], x.ndata['a_t'], x.ndata['c_t'], x.edata['e_t']]
            else:
                inputs = [x]

            # ------- NEW: gradient norms for reward and constraint (separately) -------
            # use means so scalars are well-defined for autograd
            reward_mean = self.tmp_reward.mean()
            constraint_mean = self.tmp_constraint.mean()

            reward_grads = torch.autograd.grad(
                reward_mean, inputs, retain_graph=True, allow_unused=True
            )
            constraint_grads = torch.autograd.grad(
                constraint_mean, inputs, retain_graph=True, allow_unused=True
            )

            self.last_grad_norm_reward = self._stack_and_norm(reward_grads)
            self.last_grad_norm_constraint = self._stack_and_norm(constraint_grads)

            # backprop for the full augmented objective (unchanged logic)
            tmp_augmented_reward.backward()  # no retain_graph needed since we collected the others first

            # build/return grad as before + compute full norm
            if isinstance(x, dgl.DGLGraph):
                grad = dgl.graph((x.edges()[0], x.edges()[1]), num_nodes=x.num_nodes(), device=x.device)
                grad.set_batch_num_nodes(x.batch_num_nodes())
                grad.set_batch_num_edges(x.batch_num_edges())

                grad.ndata['x_t'] = x.ndata['x_t'].grad.clone().detach().requires_grad_(False)
                grad.ndata['a_t'] = x.ndata['a_t'].grad.clone().detach().requires_grad_(False)
                grad.ndata['c_t'] = x.ndata['c_t'].grad.clone().detach().requires_grad_(False)

                grad.edata['e_t'] = x.edata['e_t'].grad.clone().detach().requires_grad_(False)
                grad.edata['ue_mask'] = x.edata['ue_mask'].detach().clone()

                # ------- NEW: full grad norm over all parts -------
                full_grads = [x.ndata['x_t'].grad, x.ndata['a_t'].grad, x.ndata['c_t'].grad, x.edata['e_t'].grad]
            else:
                grad = x.grad.clone().detach().requires_grad_(False)
                # ------- NEW: full grad norm for tensor -------
                full_grads = [x.grad]

            # take out the alpha scaling for logging
            for tmp in full_grads:
                if tmp is not None:
                    tmp /= self.alpha

            self.last_grad_norm_full = self._stack_and_norm(full_grads)

        return grad

    def get_statistics(self) -> dict:
        total_reward = self.tmp_total.clone().detach().cpu().item()
        reward = self.tmp_reward.clone().detach().mean().cpu().item()
        constraint = self.tmp_constraint.clone().detach()
        constraint_violations = torch.sum(constraint > self.bound).detach().cpu().item() / len(constraint)
        constraint = torch.mean(constraint).detach().cpu().item()
        return {
            "reward": reward,
            "constraint": constraint,
            "total_reward": total_reward,
            "constraint_violations": constraint_violations,
            "grad_norm_full": self.last_grad_norm_full,
            "grad_norm_reward": self.last_grad_norm_reward,
            "grad_norm_constraint": self.last_grad_norm_constraint,
        }

    def get_reward_constraint(self) -> dict:
        reward = self.tmp_reward.clone().detach().cpu().numpy()
        constraint = self.tmp_constraint.clone().detach().cpu().numpy()
        return {
            f"reward": reward,
            f"constraint": constraint,
        }
    
    def get_last_grad_norms(self) -> dict:
        # return a copy so callers can't mutate internal state
        return {
            "full": self.last_grad_norms["full"],
            "reward": self.last_grad_norms["reward"],
            "constraint": self.last_grad_norms["constraint"],
        }