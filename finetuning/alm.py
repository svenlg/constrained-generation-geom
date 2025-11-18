import copy
import torch
from omegaconf import OmegaConf

class AugmentedLagrangian:

    def __init__(
            self,
            config: OmegaConf,
            constraint_fn: callable,
            bound: float,
            device: torch.device = None,
            baseline: bool = False,
        ):
        # Config
        self.rho_init = config.get("rho_init", 0.5)
        lambda_min = config.get("lambda_min", -10.0)
        self.lambda_min = -abs(lambda_min)
        self.lambda_init = config.get("lambda_init", 0.0)
        self.tau = config.get("tau", 0.99)
        self.eta = config.get("eta", 1.25)
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Models
        if isinstance(constraint_fn, torch.nn.Module):
            self.constraint_fn = constraint_fn.to(self.device)
        else:
            self.constraint_fn = constraint_fn
        self.bound = float(bound)

        # ALM 
        self.lambda_ = self.lambda_init
        self.rho_ = 0.0 if baseline else self.rho_init
        self.contraction_value = None
        self.old_contraction_value = None

        # Baseline
        self.baseline = baseline

    def get_current_lambda_rho(self):
        return copy.deepcopy(self.lambda_), copy.deepcopy(self.rho_)

    def expected_constraint(self, new_samples):
        self.exp_constraint = self.constraint_fn(new_samples).mean().detach().cpu().item()
        self.g = self.exp_constraint - self.bound
        if self.rho_ > 0.0:
            self.contraction_value = min(-self.lambda_/self.rho_ , self.g)
        return copy.deepcopy(self.g)

    def update_lambda(self, new_samples):
        # Update lambda
        # lambda_k+1 = min(0, lambda_k - rho_k * (E[constraint(x)]-B))
        ec = self.expected_constraint(new_samples)
        lambda_suggestion = self.lambda_ - self.rho_ * ec
        lambda_ = min(0, lambda_suggestion)
        lambda_ = max(lambda_, self.lambda_min)
        return lambda_

    def update_rho(self):
        # Update rho
        if self.old_contraction_value is None:
            rho = self.rho_
            print(f"k = 1")
        elif self.contraction_value < self.tau * self.old_contraction_value:
            rho = self.rho_
            print(f"k =/= 1 and contraction_value < tau * old_contraction_value")
        else:
            rho = self.eta * self.rho_
            print(f"eta * rho")
        self.old_contraction_value = self.contraction_value
        return rho

    def update_lambda_rho(self, new_samples):
        self.constraint_fn.eval()
        self.lambda_ = self.update_lambda(new_samples)
        self.rho_ = self.update_rho()
        if self.baseline:
            self.lambda_ = self.lambda_init
            self.rho_ = 0.0
    
    def get_statistics(self):
        return {
            "alm/lambda": copy.deepcopy(self.lambda_),
            "alm/rho": copy.deepcopy(self.rho_),
            "alm/expected_constraint": copy.deepcopy(self.exp_constraint),
            "alm/g": copy.deepcopy(self.g),
        }

