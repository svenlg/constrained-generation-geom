import copy
import torch


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

    def grad_augmented_reward_fn(self, x: torch.Tensor) -> torch.Tensor:

        if isinstance(self.reward_fn, torch.nn.Module):
            self.reward_fn.eval()
        if isinstance(self.constraint_fn, torch.nn.Module):
            self.constraint_fn.eval()

        with torch.enable_grad():
            x = x.to(self.device)
            x = x.clone().detach().requires_grad_(True)

            tmp_augmented_reward = self(x)

            tmp_augmented_reward.backward()
            grad = x.grad

        x.requires_grad = False
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
    