import copy
import torch
from omegaconf import OmegaConf

class AugmentedLagrangian:

    def __init__(
            self,
            config: OmegaConf,
            bound: float = 0.0,
            device: torch.device = None,
        ):
        # Config
        self.rho_init = config.get("rho_init", 0.5)
        self.rho_max = config.rho_max
        lambda_min = config.get("lambda_min", -10.0)
        self.lambda_min = -abs(lambda_min)
        self.tau = config.get("tau", 0.99)
        self.eta = config.eta
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Constraint function
        self.bound = float(bound)
         
        # ALM 
        self.lambda_ = 0
        self.rho = self.rho_init
        self.contraction_value = None
        self.old_contraction_value = None

        # Baseline
        self.baseline = config.get("baseline", False)
        if self.baseline:
            base_lambda = config.get("base_lambda", 0.0)
            self.lambda_ = -abs(base_lambda)
            self.rho = 0.0

    def get_current_lambda_rho(self):
        return copy.deepcopy(self.lambda_), copy.deepcopy(self.rho)

    def convergence(self):
        ### TODO: Implement convergence criterion
        return False
    
    def expected_constraint(self, constraint_set):
        if type(constraint_set) == float: # just in the first setting case for logging 
            self.exp_constraint = constraint_set - self.bound
        else:
            self.exp_constraint = constraint_set['mean'] - self.bound
        if not self.baseline:
            self.contraction_value = min(-self.lambda_/self.rho , self.exp_constraint)
        return copy.deepcopy(self.exp_constraint)

    def update_lambda(self, constraint_set):
        # Update lambda
        # lambda_k+1 = min(0, lambda_k - rho_k * (E[constraint(x)]-B))
        ec = self.expected_constraint(constraint_set)
        ac_val = copy.deepcopy(self.rho)
        lambda_suggestion = self.lambda_ - ac_val * ec
        lambda_ = min(0, lambda_suggestion)
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
        return rho

    def update_lambda_rho(self, constraint_set):
        lambda_ = self.update_lambda(constraint_set)
        self.rho = self.update_rho()
        self.lambda_ = max(lambda_, self.lambda_min)
        if self.baseline:
            self.lambda_ = self.base_lambda
            self.rho = 0.0
    
    def get_statistics(self):
        return {
            "lambda": copy.deepcopy(self.lambda_),
            "rho": copy.deepcopy(self.rho),
            "expected_constraint": copy.deepcopy(self.exp_constraint),
        }

