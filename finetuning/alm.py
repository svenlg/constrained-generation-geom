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
        ):
        # Config
        self.rho_init = config.get("rho_init", 0.5)
        self.rho_max = config.rho_max
        lambda_min = config.get("lambda_min", -10.0)
        self.lambda_min = -abs(lambda_min)
        self.tau = config.get("tau", 0.99)
        self.eta = config.get("eta", 1.25)
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Models
        if isinstance(constraint_fn, torch.nn.Module):
            self.constraint_fn = constraint_fn.to(self.device)
        else:
            self.constraint_fn = constraint_fn
        self.bound = float(bound)
        self.filter_bound = self.constraint_fn.filter_function(torch.tensor(self.bound).to(self.device)).item()
        print(f"Filter bound: {self.filter_bound}, bound: {self.bound}")

        # ALM 
        self.lambda_ = 0
        self.rho = self.rho_init
        self.contraction_value = None
        self.old_contraction_value = None

    def get_current_lambda_rho(self):
        return copy.deepcopy(self.lambda_), copy.deepcopy(self.rho)

    def expected_constraint(self, new_samples):
        self.exp_constraint, self.gnn_constraint = self.constraint_fn(new_samples, return_gnn_output=True)
        self.exp_constraint = self.exp_constraint.mean().detach().cpu().item()
        self.g = self.exp_constraint - self.filter_bound
        self.pred_g = self.gnn_constraint.detach().cpu().item() - self.filter_bound
        # self.exp_constraint = torch.mean(constraint).detach().cpu().item() - self.bound
        self.contraction_value = min(-self.lambda_/self.rho , self.g)
        return copy.deepcopy(self.g)

    def update_lambda(self, new_samples):
        # Update lambda
        # lambda_k+1 = min(0, lambda_k - rho_k * (E[constraint(x)]-B))
        ec = self.expected_constraint(new_samples)
        lambda_suggestion = self.lambda_ - self.rho * ec
        lambda_ = min(0, lambda_suggestion)
        lambda_ = max(lambda_, self.lambda_min)
        return lambda_

    def update_rho(self):
        # Update rho
        # rho_k+1 = min(eta * rho_k, rho_max)
        if self.old_contraction_value is None:
            rho = self.rho
            print(f"k = 1")
        elif self.contraction_value < self.tau * self.old_contraction_value:
            rho = self.rho
            print(f"k =/= 1 and contraction_value < tau * old_contraction_value")
        else:
            rho = self.eta * self.rho
            print(f"eta * rho")
        self.old_contraction_value = self.contraction_value
        # rho = min(rho, self.rho_max)
        return rho

    def update_lambda_rho(self, new_samples):
        self.constraint_fn.eval()
        self.lambda_ = self.update_lambda(new_samples)
        self.rho = self.update_rho()
    
    def get_statistics(self):
        return {
            "alm/lambda": copy.deepcopy(self.lambda_),
            "alm/rho": copy.deepcopy(self.rho),
            "alm/expected_constraint": copy.deepcopy(self.exp_constraint),
            "alm/pred_g": copy.deepcopy(self.pred_g),
            "alm/g": copy.deepcopy(self.g),
        }

