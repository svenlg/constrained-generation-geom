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

        # constraint is logits
        self.constraint_logits = config.get("constraint_logits", False)

        # Normalization parameters
        self.normalize = bool(config.get("normalize", False))
        self.normalize_target = float(config.get("target", 1.0))
        self.normalize_eps = float(config.get("eps", 1e-10))

        # For logging
        self.last_grad_norm_full = None
        self.last_grad_norm_reward = None
        self.last_grad_norm_constraint = None
        self.last_grad_norm_penalty = None

    def set_lambda_rho(self, lambda_: float, rho_: float):
        self.lambda_ = copy.deepcopy(float(lambda_))
        self.rho_ = copy.deepcopy(float(rho_))

    def _ensure_eval_mode(self):
        if isinstance(self.reward_fn, torch.nn.Module):
            self.reward_fn.eval()
        if isinstance(self.constraint_fn, torch.nn.Module):
            self.constraint_fn.eval()

    def _bound_tensor_like(self, template: torch.Tensor) -> torch.Tensor:
        if self.constraint_logits:
            # safe logit even if bound is 0/1 by clamping slightly
            b = torch.as_tensor(self.bound, device=template.device, dtype=template.dtype)
            eps = 1e-10
            b = torch.clamp(b, eps, 1.0 - eps)
            b_logit = torch.log(b) - torch.log1p(-b)  # logit(b)
            return torch.ones_like(template) * b_logit
        else:
            return torch.ones_like(template) * self.bound

    def _stack_and_norm(self, tensors) -> float:
        # tensors: iterable of tensors or None
        flats = [t.reshape(-1) for t in tensors if t is not None]
        if not flats:
            return 0.0
        v = torch.cat(flats)
        # keep on device, but return a CPU float
        return torch.linalg.vector_norm(v).detach().cpu().item()
    
    def _scale_grad_list_to_norm(self, grads):
        """
        Scale a list of grads so that the concatenated L2 norm equals self.normalize_target.
        None entries remain None. Returns new list.
        """
        cur = self._stack_and_norm(grads)
        if cur <= self.normalize_eps:
            return [None if g is None else g for g in grads], 0.0  # no scaling, report norm 0
        scale = self.normalize_target / (cur + self.normalize_eps)
        return [None if g is None else g * scale for g in grads]

    def _penalty_coeff_scalar(self) -> torch.Tensor:
        """
        coeff = mean( rho * relu(g) ), with g in units matching constraint mode.
        Used in normalized composition: full_grad = rg - coeff * cg  (cg = ∇g).
        """
        tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
        tmp_rho    = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
        tmp_bound  = self._bound_tensor_like(self.tmp_constraint)
        g = self.tmp_constraint - tmp_bound - tmp_lambda / tmp_rho
        return (tmp_rho * torch.clamp(g, min=0.0)).mean()

    def _materialize_grads_like_inputs(self, inputs, grads):
        # Replace None grads with zeros_like corresponding input so we can safely package returns.
        out = []
        for ref, g in zip(inputs, grads):
            if g is None:
                out.append(torch.zeros_like(ref))
            else:
                out.append(g)
        return out

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

    def _penalty_mean(self) -> torch.Tensor:
        """mean((rho/2)*relu(g)^2), with g in units matching constraint mode."""
        tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
        tmp_rho    = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
        tmp_bound  = self._bound_tensor_like(self.tmp_constraint)
        g = self.tmp_constraint - tmp_bound - tmp_lambda / tmp_rho
        relu_g = torch.clamp(g, min=0.0)
        return ((tmp_rho / 2.0) * (relu_g ** 2)).mean()

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

    # def __call__(self, x: torch.Tensor) -> torch.Tensor:
    #     # We want to maximize reward and minimize constraint
    #     # Note lambda < 0, rho > 0
    #     self.tmp_reward = self.reward_fn(x)
    #     self.tmp_constraint = self.constraint_fn(x)
    #     tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
    #     tmp_rho = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
    #     tmp_zero = torch.zeros_like(self.tmp_constraint, device=self.tmp_constraint.device)
    #     tmp_bound = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.bound
    #     self.tmp_total = (
    #             self.tmp_reward - tmp_rho/2 * torch.max(tmp_zero, self.tmp_constraint - tmp_bound - tmp_lambda/tmp_rho)**2
    #         ).mean()
    #     return self.alpha * self.tmp_total

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Maximize reward, minimize constraint penalty
        self.tmp_reward = self.reward_fn(x)

        # --- constraint forward (logits always available) ---
        logits = self.constraint_fn(x) 
        probs = torch.sigmoid(logits)
        self.tmp_constraint_logits = logits
        # probability for logging/metrics
        self.tmp_constraint_prob = probs

        # choose internal constraint representation
        self.tmp_constraint = logits if self.constraint_logits else probs

        # --- augmented Lagrangian pieces ---
        tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
        tmp_rho = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
        tmp_bound = self._bound_tensor_like(self.tmp_constraint)   # logit(b) or b

        g = self.tmp_constraint - tmp_bound - tmp_lambda / tmp_rho  # in chosen units
        self.tmp_total = ( self.tmp_reward - (tmp_rho / 2.0) * torch.clamp(g, min=0.0) ** 2 ).mean()
        return self.alpha * self.tmp_total


    def grad_augmented_reward_fn(self, x: Union[torch.Tensor, dgl.DGLGraph]) -> torch.Tensor:
        self._ensure_eval_mode()
        with torch.enable_grad():
            x, inputs = self._make_leaf_inputs(x)

            # forward: fills self.tmp_reward, self.tmp_constraint, self.tmp_total
            tmp_augmented_reward = self(x)

            if self.normalize:
                # ----- Build normalized rg and cg -----
                # raw grads
                rg_raw = torch.autograd.grad(self.tmp_reward.mean(), inputs, retain_graph=True, allow_unused=True)
                cg_raw = torch.autograd.grad(self.tmp_constraint.mean(), inputs, retain_graph=False, allow_unused=True)

                # normalize each to unit norm
                rg = self._scale_grad_list_to_norm(rg_raw)
                cg = self._scale_grad_list_to_norm(cg_raw)

                # coeff = mean( rho * relu(g) ) as a scalar
                coeff = self._penalty_coeff_scalar()

                # update logs to reflect the enforced norms
                self.last_grad_norm_reward = self.normalize_target
                self.last_grad_norm_constraint = self.normalize_target
                # penalty component in the composed gradient is: coeff * cg, and ||cg|| = 1  => norm = |coeff|
                self.last_grad_norm_penalty = float(torch.abs(coeff).detach().cpu().item()) * self.last_grad_norm_constraint

                # compose full gradient: rg - coeff * cg
                combined = [
                    None if (r is None and c is None)
                    else ( (0.0 if r is None else r) - coeff * (0.0 if c is None else c) )
                    for r, c in zip(rg, cg)
                ]

                # keep return semantics consistent with previous version (alpha scaling applied to the full obj)
                combined_alpha = [None if g is None else self.alpha * g for g in combined]

                # materialize Nones to zeros so packaging never fails
                combined_alpha = self._materialize_grads_like_inputs(inputs, combined_alpha)

                # package grads & log full norm (unscaled by alpha for parity with your logs)
                if isinstance(x, dgl.DGLGraph):
                    grad = dgl.graph((x.edges()[0], x.edges()[1]), num_nodes=x.num_nodes(), device=x.device)
                    grad.set_batch_num_nodes(x.batch_num_nodes())
                    grad.set_batch_num_edges(x.batch_num_edges())

                    # order matches inputs: x_t, a_t, c_t, e_t
                    grad.ndata['x_t'] = combined_alpha[0].clone().detach().requires_grad_(False)
                    grad.ndata['a_t'] = combined_alpha[1].clone().detach().requires_grad_(False)
                    grad.ndata['c_t'] = combined_alpha[2].clone().detach().requires_grad_(False)
                    grad.edata['e_t'] = combined_alpha[3].clone().detach().requires_grad_(False)
                    grad.edata['ue_mask'] = x.edata['ue_mask'].detach().clone()

                    full_grads = [combined_alpha[0], combined_alpha[1], combined_alpha[2], combined_alpha[3]]
                else:
                    grad = combined_alpha[0].clone().detach().requires_grad_(False)
                    full_grads = [combined_alpha[0]]

                # log full norm with alpha unscaled (to match your previous convention)
                full_grads_unscaled = [g / self.alpha for g in full_grads]
                self.last_grad_norm_full = self._stack_and_norm(full_grads_unscaled)

            else:
                # per-term grad norms
                self.last_grad_norm_reward = self._autograd_norm(self.tmp_reward.mean(), inputs)
                self.last_grad_norm_constraint = self._autograd_norm(self.tmp_constraint.mean(), inputs)
                self.last_grad_norm_penalty = self._autograd_norm(self._penalty_mean(), inputs)

                # true objective gradient via backward ---
                tmp_augmented_reward.backward()

                grad, full_grads = self._build_grad_and_collect_full_grads(x)
                self._unscale_full_grads_inplace(full_grads)  # divide by alpha for logging only
                self.last_grad_norm_full = self._stack_and_norm(full_grads)

        return grad

    def get_statistics(self) -> dict:
        total_reward = self.tmp_total.clone().detach().cpu().item()
        reward_mean  = self.tmp_reward.clone().detach().mean().cpu().item()
        prob = self.tmp_constraint_prob.clone().detach()
        mean_prob = prob.mean().cpu().item()
        violations = (prob > self.bound).float().mean().cpu().item()
        logit_mean = self.tmp_constraint_logits.clone().detach().mean().cpu().item()
        return {
            "reward": reward_mean,
            "constraint_prob": mean_prob,
            "constraint_logit": logit_mean,
            "total_reward": total_reward,
            "constraint_violations": violations,
            "grad_norm/full": self.last_grad_norm_full,
            "grad_norm/reward": self.last_grad_norm_reward,
            "grad_norm/constraint": self.last_grad_norm_constraint,
            "grad_norm/penalty": self.last_grad_norm_penalty,
        }

    def get_reward_constraint(self) -> dict:
        reward = self.tmp_reward.clone().detach().cpu().numpy()
        prob   = self.tmp_constraint_prob.clone().detach().cpu().numpy()
        logit  = self.tmp_constraint_logits.clone().detach().cpu().numpy()
        return {
            "reward": reward,
            "constraint_prob": prob,
            "constraint_logit": logit,
        }


###### OLD VERSION BELOW ######

# import copy
# import torch
# import dgl
# from typing import Union

# class AugmentedReward:
#     def __init__(
#             self, 
#             reward_fn: callable, # if torch module make sure to call .to(self.device)
#             constraint_fn: callable, # if torch module make sure to call .to(self.device)
#             alpha: float,
#             bound: float,
#             device: torch.device = None,
#         ):

#         self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         if isinstance(reward_fn, torch.nn.Module):
#             self.reward_fn = reward_fn.to(self.device)
#             self.reward_fn.eval()
#         if isinstance(constraint_fn, torch.nn.Module):
#             self.constraint_fn = constraint_fn.to(self.device)
#             self.constraint_fn.eval()

#         self.alpha = copy.deepcopy(float(alpha))
#         self.bound = copy.deepcopy(float(bound))

#         # Initialize lambda and rho
#         self.lambda_ = 0.0
#         self.rho_ = 1.0

#         # For logging
#         self.last_grad_norm_full = None
#         self.last_grad_norm_reward = None
#         self.last_grad_norm_constraint = None

#     def _stack_and_norm(self, tensors) -> float:
#         # tensors: iterable of tensors or None
#         flats = [t.reshape(-1) for t in tensors if t is not None]
#         if not flats:
#             return 0.0
#         v = torch.cat(flats)
#         # keep on device, but return a CPU float
#         return torch.linalg.vector_norm(v).detach().cpu().item()

#     def set_lambda_rho(self, lambda_: float, rho_: float):
#         self.lambda_ = copy.deepcopy(float(lambda_))
#         self.rho_ = copy.deepcopy(float(rho_))
    
#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         # We want to maximize reward and minimize constraint
#         # Note lambda < 0, rho > 0
#         self.tmp_reward = self.reward_fn(x)
#         self.tmp_constraint = self.constraint_fn(x)
#         tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
#         tmp_rho = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
#         tmp_zero = torch.zeros_like(self.tmp_constraint, device=self.tmp_constraint.device)
#         tmp_bound = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.bound
#         self.tmp_total = (
#                 self.tmp_reward - tmp_rho/2 * torch.max(tmp_zero, self.tmp_constraint - tmp_bound - tmp_lambda/tmp_rho)**2
#             ).mean()
#         return self.alpha * self.tmp_total

#     # def __call__(self, x: torch.Tensor) -> torch.Tensor:
#     #     # We want to maximize reward and minimize constraint
#     #     # Note lambda < 0, rho > 0
#     #     self.tmp_reward = self.reward_fn(x)
#     #     reward = self.tmp_reward.mean()
#     #     self.tmp_constraint = self.constraint_fn(x)
#     #     constraint = self.tmp_constraint.mean()
#     #     tmp_lambda = torch.ones_like(constraint, device=constraint.device) * self.lambda_
#     #     tmp_rho = torch.ones_like(constraint, device=constraint.device) * self.rho_
#     #     tmp_zero = torch.zeros_like(constraint, device=constraint.device)
#     #     tmp_bound = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.bound
#     #     self.tmp_total = (
#     #             reward - tmp_rho/2 * torch.max(tmp_zero, constraint - tmp_bound - tmp_lambda/tmp_rho)**2
#     #         ).mean()
#     #     return self.alpha * self.tmp_total

#     # def grad_augmented_reward_fn(self, x: Union[torch.Tensor, dgl.DGLGraph]) -> torch.Tensor:

#     #     if isinstance(self.reward_fn, torch.nn.Module):
#     #         self.reward_fn.eval()
#     #     if isinstance(self.constraint_fn, torch.nn.Module):
#     #         self.constraint_fn.eval()

#     #     with torch.enable_grad():
#     #         x = x.to(self.device)
#     #         if isinstance(x, dgl.DGLGraph):
#     #             x.ndata['x_t'] = x.ndata['x_t'].clone().detach().requires_grad_(True)
#     #             x.ndata['a_t'] = x.ndata['a_t'].clone().detach().requires_grad_(True)
#     #             x.ndata['c_t'] = x.ndata['c_t'].clone().detach().requires_grad_(True)
#     #             x.edata['e_t'] = x.edata['e_t'].clone().detach().requires_grad_(True)
#     #         if isinstance(x, torch.Tensor):
#     #             x = x.clone().detach().requires_grad_(True)

#     #         tmp_augmented_reward = self(x)

#     #         tmp_augmented_reward.backward()

#     #         if isinstance(x, dgl.DGLGraph):
#     #             grad = dgl.graph((x.edges()[0], x.edges()[1]), num_nodes=x.num_nodes(), device=x.device)
#     #             grad.set_batch_num_nodes(x.batch_num_nodes())
#     #             grad.set_batch_num_edges(x.batch_num_edges())

#     #             grad.ndata['x_t'] = x.ndata['x_t'].grad.clone().detach().requires_grad_(False)
#     #             grad.ndata['a_t'] = x.ndata['a_t'].grad.clone().detach().requires_grad_(False)
#     #             grad.ndata['c_t'] = x.ndata['c_t'].grad.clone().detach().requires_grad_(False)

#     #             grad.edata['e_t'] = x.edata['e_t'].grad.clone().detach().requires_grad_(False)
#     #             grad.edata['ue_mask'] = x.edata['ue_mask'].detach().clone()

#     #         if isinstance(x, torch.Tensor):
#     #             grad = x.grad.clone().detach().requires_grad_(False)

#     #     return grad

#     def grad_augmented_reward_fn(self, x: Union[torch.Tensor, dgl.DGLGraph]) -> torch.Tensor:
#         if isinstance(self.reward_fn, torch.nn.Module):
#             self.reward_fn.eval()
#         if isinstance(self.constraint_fn, torch.nn.Module):
#             self.constraint_fn.eval()

#         with torch.enable_grad():
#             x = x.to(self.device)
#             if isinstance(x, dgl.DGLGraph):
#                 x.ndata['x_t'] = x.ndata['x_t'].clone().detach().requires_grad_(True)
#                 x.ndata['a_t'] = x.ndata['a_t'].clone().detach().requires_grad_(True)
#                 x.ndata['c_t'] = x.ndata['c_t'].clone().detach().requires_grad_(True)
#                 x.edata['e_t'] = x.edata['e_t'].clone().detach().requires_grad_(True)
#             if isinstance(x, torch.Tensor):
#                 x = x.clone().detach().requires_grad_(True)

#             # forward: builds self.tmp_reward, self.tmp_constraint, self.tmp_total
#             tmp_augmented_reward = self(x)

#             # ------- NEW: set up leaf inputs for autograd.grad -------
#             if isinstance(x, dgl.DGLGraph):
#                 inputs = [x.ndata['x_t'], x.ndata['a_t'], x.ndata['c_t'], x.edata['e_t']]
#             else:
#                 inputs = [x]

#             # ------- NEW: gradient norms for reward and constraint (separately) -------
#             # use means so scalars are well-defined for autograd
#             reward_mean = self.tmp_reward.mean()
#             constraint_mean = self.tmp_constraint.mean()

#             reward_grads = torch.autograd.grad(
#                 reward_mean, inputs, retain_graph=True, allow_unused=True
#             )
#             constraint_grads = torch.autograd.grad(
#                 constraint_mean, inputs, retain_graph=True, allow_unused=True
#             )

#             self.last_grad_norm_reward = self._stack_and_norm(reward_grads)
#             self.last_grad_norm_constraint = self._stack_and_norm(constraint_grads)

#             # Build the same tensors as in __call__ (on correct device/shape)
#             tmp_lambda = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.lambda_
#             tmp_rho    = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.rho_
#             tmp_zero   = torch.zeros_like(self.tmp_constraint, device=self.tmp_constraint.device)
#             tmp_bound  = torch.ones_like(self.tmp_constraint, device=self.tmp_constraint.device) * self.bound

#             g = self.tmp_constraint - tmp_bound - tmp_lambda / tmp_rho
#             relu_g = torch.max(tmp_zero, g)  # same as torch.clamp(g, min=0)

#             penalty_mean = (tmp_rho / 2.0) * (relu_g ** 2)
#             penalty_mean = penalty_mean.mean()

#             penalty_grads = torch.autograd.grad(
#                 penalty_mean, inputs, retain_graph=True, allow_unused=True
#             )
#             self.last_grad_norm_penalty = self._stack_and_norm(penalty_grads)

#             # backprop for the full augmented objective (unchanged logic)
#             tmp_augmented_reward.backward()  # no retain_graph needed since we collected the others first

#             # build/return grad as before + compute full norm
#             if isinstance(x, dgl.DGLGraph):
#                 grad = dgl.graph((x.edges()[0], x.edges()[1]), num_nodes=x.num_nodes(), device=x.device)
#                 grad.set_batch_num_nodes(x.batch_num_nodes())
#                 grad.set_batch_num_edges(x.batch_num_edges())

#                 grad.ndata['x_t'] = x.ndata['x_t'].grad.clone().detach().requires_grad_(False)
#                 grad.ndata['a_t'] = x.ndata['a_t'].grad.clone().detach().requires_grad_(False)
#                 grad.ndata['c_t'] = x.ndata['c_t'].grad.clone().detach().requires_grad_(False)

#                 grad.edata['e_t'] = x.edata['e_t'].grad.clone().detach().requires_grad_(False)
#                 grad.edata['ue_mask'] = x.edata['ue_mask'].detach().clone()

#                 # ------- NEW: full grad norm over all parts -------
#                 full_grads = [x.ndata['x_t'].grad, x.ndata['a_t'].grad, x.ndata['c_t'].grad, x.edata['e_t'].grad]
#             else:
#                 grad = x.grad.clone().detach().requires_grad_(False)
#                 # ------- NEW: full grad norm for tensor -------
#                 full_grads = [x.grad]

#             # take out the alpha scaling for logging
#             for tmp in full_grads:
#                 if tmp is not None:
#                     tmp /= self.alpha

#             self.last_grad_norm_full = self._stack_and_norm(full_grads)

#         return grad
    
#     def get_statistics(self) -> dict:
#         total_reward = self.tmp_total.clone().detach().cpu().item()
#         reward = self.tmp_reward.clone().detach().mean().cpu().item()
#         constraint = self.tmp_constraint.clone().detach()
#         constraint_violations = torch.sum(constraint > self.bound).detach().cpu().item() / len(constraint)
#         constraint = torch.mean(constraint).detach().cpu().item()
#         return {
#             "reward": reward,
#             "constraint": constraint,
#             "total_reward": total_reward,
#             "constraint_violations": constraint_violations,
#             "grad_norm_full": self.last_grad_norm_full,
#             "grad_norm_reward": self.last_grad_norm_reward,
#             "grad_norm_constraint": self.last_grad_norm_constraint,
#         }

#     def get_reward_constraint(self) -> dict:
#         reward = self.tmp_reward.clone().detach().cpu().numpy()
#         constraint = self.tmp_constraint.clone().detach().cpu().numpy()
#         return {
#             f"reward": reward,
#             f"constraint": constraint,
#         }

