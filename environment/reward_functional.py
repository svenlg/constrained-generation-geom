import torch
from environment.property_calculation import compute_property_stats, compute_property_grad

class RewardFunctional:
    def __init__(
            self, 
            reward_fn: str, 
            constraint_fn: str, 
            reward_lambda: float,
            bound: float = 0.0,
            constraint_model: callable = None,
            device: torch.device = None,
        ):
        self.reward_fn = reward_fn
        self.constraint_fn = constraint_fn
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_lambda = float(reward_lambda)
        self.bound = float(bound)
        self.constraint_model = constraint_model

        # Initialize lambda and rho
        self.lambda_ = 0.0
        self.rho_ = 1.0

    def set_lambda_rho(self, lambda_:float, rho_:float):
        self.lambda_ = float(lambda_)
        self.rho_ = float(rho_)

    def reward(self, batched_molecules):
        return compute_property_stats(
            molecules = batched_molecules, 
            property = self.reward_fn,
            device = self.device,
            model = self.constraint_model
        )
    
    def reward_grad(self, batched_molecules):
        return compute_property_grad(
            molecules = batched_molecules, 
            property = self.reward_fn,
            reward_lambda = self.reward_lambda,
            device = self.device,
            model = self.constraint_model
        )
    
    def constraint(self, batched_molecules):
        return compute_property_stats(
            molecules = batched_molecules, 
            property = self.constraint_fn,
            device = self.device,
            model = self.constraint_model
        )

    def constraint_grad(self, batched_molecules):
        return compute_property_grad(
            molecules = batched_molecules, 
            property = self.constraint_fn,
            reward_lambda = self.reward_lambda,
            device = self.device,
            model = self.constraint_model
        )
    
    def functional(self, batched_molecules):
        reward_set = self.reward(batched_molecules)
        constraint_set = self.constraint(batched_molecules)
        reward = reward_set["mean"]
        constraint = constraint_set["mean"]
        constraint_violations = torch.sum(constraint_set["all_finite"] > self.bound) / max(len(constraint_set["all_finite"]), 1)
        total_reward = reward + self.lambda_ * constraint - self.rho_ / 2 * max(0, constraint - self.bound)**2
        return {
            "total_reward": total_reward,
            "reward": reward,
            "reward_median": reward_set["median"],
            "reward_std": reward_set["std"],
            "reward_num_invalid": reward_set["num_invalid"],
            "constraint": constraint,
            "constraint_median": constraint_set["median"],
            "constraint_std": constraint_set["std"],
            "constraint_num_invalid": constraint_set["num_invalid"],
            "constraint_violations": constraint_violations.item(),
        }
        
    def grad_reward_fn(self, batched_molecules):
        # We want to maximize reward and minimize constraint
        # Note lambda < 0, rho > 0
        # rtilde = reward + lambda * constraint - rho/2 * max(0, constraint)^2

        with torch.enable_grad():
            r_values, r_grads = self.reward_grad(batched_molecules) 
            c_values, c_grads = self.constraint_grad(batched_molecules)

            total_reward = []
            full_grad = []
            for r_val, r_grad, c_val, c_grad in zip(r_values, r_grads, c_values, c_grads):
                
                assert r_grad.shape == c_grad.shape, f"Reward and constraint shapes do not match: {r_val.shape} vs {c_val.shape}"
                r_grad = r_grad.to(c_grad.device)
                tmp_lambda = torch.ones_like(c_grad, device=c_grad.device) * self.lambda_
                tmp_rho = torch.ones_like(c_grad, device=c_grad.device) * self.rho_

                if torch.isfinite(c_val) and torch.isfinite(r_val):
                    c_val_bound = c_val - self.bound
                    grad = (
                        r_grad \
                        + tmp_lambda * c_grad \
                        - tmp_rho * (c_val_bound > 0).float() * c_grad
                    )
                    reward_term = (
                        r_val.cpu().item() \
                        + self.lambda_ * c_val.cpu().item() \
                        - self.rho_ / 2 * max(0, c_val_bound.cpu().item())**2
                    )
                    reward_term = torch.tensor(reward_term, device=r_grad.device)
                else:
                    grad = torch.zeros_like(r_grad, device=r_grad.device)
                    reward_term = torch.tensor(float('inf'), device=r_grad.device)

                full_grad.append(grad)
                total_reward.append(reward_term)

        return full_grad
 