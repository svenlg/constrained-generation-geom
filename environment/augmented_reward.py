import copy
import torch
import dgl
from typing import Union
from omegaconf import OmegaConf


class AugmentedReward:
    def __init__(
            self, 
            reward_fn: callable, # if torch module make sure to call .to(self.device)
            constraint_fn: callable, # if torch module make sure to call .to(self.device)
            alpha: float,
            bound: float,
            device: torch.device = None,
            config: OmegaConf = None,
            baseline: bool = False,
        ):

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if isinstance(reward_fn, torch.nn.Module):
            self.reward_fn = reward_fn.to(self.device)
            self.reward_fn.eval()
        else:
            self.reward_fn = reward_fn
        if isinstance(constraint_fn, torch.nn.Module):
            self.constraint_fn = constraint_fn.to(self.device)
        else:
            self.constraint_fn = constraint_fn
        
        self.alpha = copy.deepcopy(float(alpha))
        self.bound = copy.deepcopy(float(bound))

        self.filter_bound = self.constraint_fn.filter_function(torch.tensor(self.bound).to(self.device)).item()

        # Initialize lambda and rho
        self.lambda_ = 0.0
        self.rho_ = 1.0

        # For fixed value cauclation (baseline)
        self.baseline = baseline
        if self.baseline:
            self.lambda_ = config.get("base_lambda", 1.0)
            self.rho = 0.0
        
        # For logging
        self.last_grad_norm_full = None
        self.last_grad_norm_reward = None
        self.last_grad_norm_constraint = None
        self.last_grad_norm_penalty = None

    def set_lambda_rho(self, lambda_: float, rho_: float):
        if not self.baseline:
            self.lambda_ = copy.deepcopy(float(lambda_))
            self.rho_ = copy.deepcopy(float(rho_))

    def _ensure_eval_mode(self):
        if isinstance(self.reward_fn, torch.nn.Module):
            self.reward_fn.eval()
        if isinstance(self.constraint_fn, torch.nn.Module):
            self.constraint_fn.eval()

    def _stack_and_norm(self, tensors) -> float:
        # tensors: iterable of tensors or None
        flats = [t.reshape(-1) for t in tensors if t is not None]
        if not flats:
            return 0.0
        v = torch.cat(flats)
        # keep on device, but return a CPU float
        return torch.linalg.vector_norm(v).detach().cpu().item()
    
    def _penalty_mean(self) -> torch.Tensor:
        """mean((rho/2)*relu(g)^2), with g in units matching constraint mode."""
        tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
        tmp_rho = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
        tmp_bound = torch.ones_like(self.tmp_constraint) * self.filter_bound
        if self.rho_ > 0.0 and self.baseline:
            g = self.tmp_constraint - tmp_bound - tmp_lambda / tmp_rho
            relu_g = torch.clamp(g, min=0.0)
            tmp_ret = ((tmp_rho / 2.0) * (relu_g ** 2)).mean()
        elif self.baseline:
            g = self.tmp_constraint - tmp_bound
            tmp_ret = tmp_lambda * g
        else:
            g = torch.tensor(0.0, device=self.tmp_constraint.device)
            relu_g = torch.clamp(g, min=0.0)
            tmp_ret = ((tmp_rho / 2.0) * (relu_g ** 2)).mean()
        return tmp_ret

    def _make_leaf_inputs(self, x: Union[torch.Tensor, dgl.DGLGraph]):
        """Clone x onto device and return (x_leaf, inputs_list) where inputs_list are the leaf tensors."""
        x = x.to(self.device)
        if isinstance(x, dgl.DGLGraph):
            x.ndata['x_t'] = x.ndata['x_t'].clone().detach().requires_grad_(True)
            x.ndata['a_t'] = x.ndata['a_t'].clone().detach().requires_grad_(True)
            x.ndata['c_t'] = x.ndata['c_t'].clone().detach().requires_grad_(True)
            x.edata['e_t'] = x.edata['e_t'].clone().detach().requires_grad_(True)
            inputs = [x.ndata['x_t'], x.ndata['a_t'], x.ndata['c_t'], x.edata['e_t']]
        else:
            x = x.clone().detach().requires_grad_(True)
            inputs = [x]
        return x, inputs

    def _autograd_norm(self, scalar: torch.Tensor, inputs) -> float:
        """Grad norm of 'scalar' w.r.t. 'inputs'."""
        grads = torch.autograd.grad(scalar, inputs, retain_graph=True, allow_unused=True)
        return self._stack_and_norm(grads)

    def _build_grad_and_collect_full_grads(self, x):
        """After backward() on the full objective, package grads and list of full grad tensors."""
        if isinstance(x, dgl.DGLGraph):
            grad = dgl.graph((x.edges()[0], x.edges()[1]), num_nodes=x.num_nodes(), device=x.device)
            grad.set_batch_num_nodes(x.batch_num_nodes())
            grad.set_batch_num_edges(x.batch_num_edges())
            grad.ndata['x_t'] = x.ndata['x_t'].grad.clone().detach().requires_grad_(False)
            grad.ndata['a_t'] = x.ndata['a_t'].grad.clone().detach().requires_grad_(False)
            grad.ndata['c_t'] = x.ndata['c_t'].grad.clone().detach().requires_grad_(False)
            grad.edata['e_t'] = x.edata['e_t'].grad.clone().detach().requires_grad_(False)
            grad.edata['ue_mask'] = x.edata['ue_mask'].detach().clone()
            full_grads = [x.ndata['x_t'].grad, x.ndata['a_t'].grad, x.ndata['c_t'].grad, x.edata['e_t'].grad]
        else:
            grad = x.grad.clone().detach().requires_grad_(False)
            full_grads = [x.grad]
        return grad, full_grads

    def _unscale_full_grads_inplace(self, grads_list):
        """Remove the alpha scaling applied to the full objective before logging."""
        for g in grads_list:
            if g is not None:
                g /= self.alpha

    def __call__(self, x: Union[torch.Tensor, dgl.DGLGraph]) -> torch.Tensor:
        # Maximize reward, minimize constraint penalty
        self.tmp_reward, self.gnn_reward = self.reward_fn(x, return_gnn_output=True) # tmp_reward shape (batch,), gnn_reward is scalar
        self.tmp_constraint, self.gnn_constraint = self.constraint_fn(x, return_gnn_output=True) # tmp_constraint shape (batch,), gnn_constraint is scalar

        reward = self.tmp_reward.mean()
        constraint = self.tmp_constraint.mean()

        # The normal case (r - rho/2 * ReLU (g))
        if self.rho_ > 0.0 and not self.baseline:
            g = constraint - self.bound - self.lambda_ / self.rho_
            self.tmp_total = ( reward - (self.rho_ / 2.0) * torch.clamp(g, min=0.0) ** 2 ).mean()
        elif self.baseline:
            g = (constraint - self.bound)
            self.tmp_total = reward - self.lambda_ * g
        else:
            g = torch.tensor(0.0, device=self.tmp_constraint.device, requires_grad=True)
            self.tmp_total = ( reward - (self.rho_ / 2.0) * torch.clamp(g, min=0.0) ** 2 ).mean()
        
        return self.alpha * self.tmp_total

    def grad_augmented_reward_fn(self, x: Union[torch.Tensor, dgl.DGLGraph]) -> torch.Tensor:
        self._ensure_eval_mode()
        with torch.enable_grad():
            x, inputs = self._make_leaf_inputs(x)

            # forward: fills self.tmp_reward, self.tmp_constraint, self.tmp_total
            tmp_augmented_reward = self(x)

            # per-term grad norms
            self.last_grad_norm_reward = self._autograd_norm(self.tmp_reward.mean(), inputs)
            self.last_grad_norm_constraint = self._autograd_norm(self.tmp_constraint.mean(), inputs)
            if self.rho_ > 0.0:
                self.last_grad_norm_penalty = self._autograd_norm(self._penalty_mean(), inputs)

            # true objective gradient via backward ---
            tmp_augmented_reward.backward()

            grad, full_grads = self._build_grad_and_collect_full_grads(x)
            self._unscale_full_grads_inplace(full_grads) # divide by alpha for logging only
            self.last_grad_norm_full = self._stack_and_norm(full_grads)

        return grad

    def get_statistics(self) -> dict:
        total_reward = self.tmp_total.clone().detach().cpu().item()
        reward = self.tmp_reward.clone().detach().mean().cpu().item()
        constraint = self.tmp_constraint.clone().detach().mean().cpu().item()
        violations = (self.tmp_constraint >= self.filter_bound+1e-6).float().mean().cpu().item()
        if self.rho > 0.0 and not self.baseline:
            penalty = self.rho_ / 2.0 * max(constraint - self.filter_bound, ) ** 2
        else:
            penalty = self.lambda_ * (constraint - self.filter_bound)
        pred_reward = self.gnn_reward.detach().cpu().item()
        pred_constraint = self.gnn_constraint.detach().cpu().item()
        predict_penalty = self.rho_ / 2.0 * max(pred_constraint - self.filter_bound, 0.0) ** 2
        ret_dict = {
            "reward": float(reward),
            "constraint": float(constraint),
            "total_reward": float(total_reward),
            "constraint_violations": float(violations),
            "penalty": float(penalty),
            "pred/reward": float(pred_reward),
            "pred/constraint": float(pred_constraint),
            "pred/penalty": float(predict_penalty),
        }
        if self.last_grad_norm_full is not None:
            ret_dict['grad_norm/full'] = float(self.last_grad_norm_full)
        if self.last_grad_norm_reward is not None:
            ret_dict['grad_norm/reward'] = float(self.last_grad_norm_reward)
        if self.last_grad_norm_constraint is not None:
            ret_dict['grad_norm/constraint'] = float(self.last_grad_norm_constraint)
        if self.last_grad_norm_penalty is not None:
            ret_dict['grad_norm/penalty'] = float(self.last_grad_norm_penalty)
        if self.last_grad_norm_full is not None and self.last_grad_norm_penalty is not None:
            ret_dict['grad_norm/percentage_penalty'] = float(self.last_grad_norm_penalty / (self.last_grad_norm_full + 1e-10))
        return ret_dict

    def get_reward_constraint(self) -> dict:
        # TODO: check if this returns the correct values
        pred_reward = self.gnn_reward.clone().detach().cpu().numpy()
        pred_constraint = self.gnn_constraint.clone().detach().cpu().numpy()
        return {
            "reward": pred_reward,
            "constraint": pred_constraint,
        }

    def get_individual_reward_constraint(self) -> dict:
        reward = self.tmp_reward.clone().detach().cpu().numpy()
        constraint = self.tmp_constraint.clone().detach().cpu().numpy()
        return {
            "reward": reward,
            "constraint": constraint,
        }


