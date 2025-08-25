# pylint: disable=missing-module-docstring
# pylint: disable=no-name-in-module
import os
import os.path as osp
import time
import copy
import wandb
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from utils.setup import parse_args, update_config_with_args
from utils.utils import extract_trailing_numbers, seed_everything

import dgl
import flowmol

from environment.reward_functional import RewardFunctional

from finetuning.alm import AugmentedLagrangian
from finetuning.flow_adjoint import AdjointMatchingFinetuningTrainerFlowMol

# Load - Flow Model
def setup_gen_model(flow_model: str, device: torch.device): 
    gen_model = flowmol.load_pretrained(flow_model)
    gen_model.to(device)
    return gen_model

# Setup PAMNet
def setup_pamnet_model(config: OmegaConf, device: torch.device):
    from PAMNet.models import PAMNet_s, Config
    abs_path = os.getcwd()
    path = osp.join(abs_path, 'pretrained_models', config.type, config.dataset, config.date)
    tmp_config = OmegaConf.load(path + '/config.yaml')
    pamnet_config = Config(
        dataset = tmp_config.dataset, 
        dim = tmp_config.dim, 
        n_layer = tmp_config.n_layer, 
        cutoff_l = tmp_config.cutoff_l, 
        cutoff_g = tmp_config.cutoff_g,
    )
    reward_model = PAMNet_s(pamnet_config)
    reward_model.load_state_dict(torch.load(path + '/model.pth', map_location='cpu'))
    reward_model.to(device)
    reward_model.eval()
    return reward_model

# Sampling
def sampling(config: OmegaConf, model: flowmol.FlowMol, device: torch.device):
    model.to(device)
    new_molecules, _ = model.sample_random_sizes(
        sampler_type = config.sampler_type,
        n_molecules = config.num_samples, 
        n_timesteps = config.num_integration_steps + 1, 
        device = device,
        keep_intermediate_graphs = True,
    )
    return new_molecules


def main():
    # Parse command line arguments
    args = parse_args()

    # Load config from file
    config_path = Path("configs/augmented_lagrangian.yaml")
    config = OmegaConf.load(config_path)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)

    # Setup - Seed and device
    seed_everything(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup - WandB
    if args.use_wandb:
        if "sweep" in config.experiment:
            wandb.init()
            run_name = wandb.run.name  # e.g., "olive-sweep-229"
            run_number = extract_trailing_numbers(run_name)  # e.g., 229
            run_id = wandb.run.id   # e.g., "ame6uc42"
            print(f"Run #{run_number} - ID: {run_id}", flush=True)
        else:
            wandb.init(
                name=config.experiment, 
                config=dict(config),
            )

    tmp_time = datetime.now().strftime("%m-%d-%H")
    save_path = Path(config.root) / Path("aa_experiments") / Path(config.experiment)
    if args.use_wandb and ("sweep" in config.experiment):
        save_path = save_path / Path(f"{run_id}")
    if (args.save_samples or args.save_model or args.save_plots) and not args.debug:
        if args.output_end is not None:
            save_path = save_path / Path(f"{args.output_end}")
        save_path = save_path / Path(tmp_time)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Run will be saved at:")
        print(save_path)

    # Plotting - Settings
    config.verbose = args.verbose

    # General Parameters
    flowmol_model = config.flow_model

    # Reward and Constraint functions
    reward_fn = config.reward.fn
    constraint_fn = config.constraint.fn
    bound = config.constraint.bound

    # Augmented Lagrangian Parameters
    rho_init = config.augmented_lagrangian.rho_init
    rho_max = config.augmented_lagrangian.rho_max
    eta = config.augmented_lagrangian.eta
    lagrangian_updates = config.augmented_lagrangian.lagrangian_updates

    # Adjoint Matching Parameters
    reward_lambda = config.reward_lambda
    learning_rate = config.adjoint_matching.lr
    clip_grad_norm = config.adjoint_matching.clip_grad_norm
    clip_loss = config.adjoint_matching.clip_loss
    batch_size = config.adjoint_matching.batch_size
    traj_samples_per_stage = config.adjoint_matching.sampling.num_samples
    traj_len = config.adjoint_matching.sampling.num_integration_steps
    finetune_steps = config.adjoint_matching.finetune_steps
    num_iterations = config.adjoint_matching.num_iterations

    num_iterations = 500 // lagrangian_updates
    plotting_freq = 2

    baseline = args.baseline
    if baseline:
        base_lambda = config.augmented_lagrangian.base_lambda

    config.augmented_lagrangian.sampling.sampler_type = "euler"
    config.adjoint_matching.sampling.sampler_type = "memoryless"
    config.reward_sampling.sampler_type = "euler"
    
    print(f"--- Start ---", flush=True)
    print(f"Finetuning {flowmol_model} in experiment {config.experiment}", flush=True)
    print(f"Reward: {reward_fn} - Constraint: {constraint_fn}", flush=True)
    print(f"Maximum Bound: {bound}", flush=True)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_time}", flush=True)
    print(f"Device: {device}", flush=True)
    start_time = time.time()

    print(f"--- Config ---", flush=True)
    print(f"Augmented Lagrangian Parameters", flush=True)
    if baseline:
        print(f"\tbaseline: {baseline}", flush=True)
        print(f"\tbase_lambda: {base_lambda}", flush=True)
    else:
        print(f"\trho_init: {rho_init}", flush=True)
        print(f"\trho_max: {rho_max}", flush=True)
        print(f"\teta: {eta}", flush=True)
        print(f"\tlagrangian_updates: {lagrangian_updates}", flush=True)

    print(f"Adjoint Matching Parameters", flush=True)
    print(f"\treward_lambda: {reward_lambda}", flush=True)
    print(f"\tlr: {learning_rate}", flush=True)
    print(f"\tclip_grad_norm: {clip_grad_norm}", flush=True)
    print(f"\tclip_loss: {clip_loss}", flush=True)
    print(f"\tbatch_size: {batch_size}", flush=True)
    print(f"\tsampling.num_samples: {traj_samples_per_stage}", flush=True)
    print(f"\tsampling.num_integration_steps: {traj_len}", flush=True)
    print(f"\tfinetune_steps: {finetune_steps}", flush=True)
    print(f"\tnum_iterations: {num_iterations}", flush=True)

    # Setup - Gen Model
    base_model = setup_gen_model(config.flow_model, device=device)
    gen_model = copy.deepcopy(base_model)

    # Setup - Reward and Gradient Functions
    if constraint_fn == "sascore":
        constraint_model = setup_pamnet_model(config.pamnet, device=device)
    else:
        constraint_model = None

    # Setup - Environment, RewardFunctional, ConstraintModel
    reward_functional = RewardFunctional(
        reward_fn = config.reward.fn,
        constraint_fn = config.constraint.fn,
        reward_lambda = config.reward_lambda,
        bound = config.constraint.bound,
        constraint_model = constraint_model,
        device = device,
    )
    
    # Set the initial lambda and rho
    reward_functional.set_lambda_rho(
        lambda_ = 0.0, 
        rho_ = rho_init,
    )

    # Initialize lists to store loss and rewards
    al_stats = {
        "lambda": [],
        "rho": [],
        "expected_constraint": [],
    }
    al_total_rewards = []
    al_rewards = []
    al_constraints = []
    al_constraint_violations = []
    al_losses = []
    al_best_reward = -1e8
    al_lowest_const_violations = 1.0
    al_best_epoch = 0
    if args.save_samples:
        from dgl import save_graphs
        sample_path = save_path / Path("samples")
        sample_path.mkdir(parents=True, exist_ok=True)

    # Generate Samples
    tmp_model = copy.deepcopy(gen_model)
    new_molecules = sampling(
        config.reward_sampling,
        tmp_model,
        device=device
    )
    del tmp_model
    if args.save_samples:
        tmp_save_path = str(sample_path / Path("samples_0.bin"))
        tmp_samples = [graph for graph in dgl.unbatch(new_molecules.cpu())]
        save_graphs(tmp_save_path, tmp_samples)
        del tmp_samples

    # Compute reward for current samples
    tmp_dict = reward_functional.functional(new_molecules)
    al_total_rewards.append(tmp_dict["total_reward"])
    al_rewards.append(tmp_dict["reward"])
    al_constraints.append(tmp_dict["constraint"])
    al_constraint_violations.append(tmp_dict["constraint_violations"])

    al_lowest_const_violations = al_constraint_violations[-1]
    al_best_reward = al_rewards[-1]

    alm = AugmentedLagrangian(
        config = config.augmented_lagrangian,
        bound = config.constraint.bound,
        device = device,
    )
    # Set initial expected constraint (only needed for logging)
    _ = alm.expected_constraint(tmp_dict["constraint"])

    # Log al initial states
    if args.use_wandb:
        logs = {}
        logs.update(tmp_dict)
        logs.update({"total_best_reward": al_best_reward})
        log = alm.get_statistics()
        logs.update(log) # lambda, rho, expected_constraint
        wandb.log(logs)

    #### AUGMENTED LAGRANGIAN - BEGIN ####
    for k in range(1, lagrangian_updates + 1):
        print(f"--- AL Round {k}/{lagrangian_updates} ---", flush=True)

        # Get the current lambda and rho
        lambda_, rho_ = alm.get_current_lambda_rho()
        log = alm.get_statistics()
        for key in al_stats:
            al_stats[key].append(float(log[key]))

        # Set the lambda and rho in the reward functional
        reward_functional.set_lambda_rho(lambda_, rho_)

        # Print lambda and rho for the current round
        print(f"Lambda: {lambda_:.4f}, rho: {rho_:.4f}", flush=True)

        # Set up - Adjoint Matching
        trainer = AdjointMatchingFinetuningTrainerFlowMol(
            config = config.adjoint_matching,
            model = copy.deepcopy(gen_model),
            base_model = copy.deepcopy(base_model),
            grad_reward_fn = reward_functional.grad_reward_fn,
            device = device,
            verbose = False,
        )

        am_total_rewards = []
        am_rewards = []
        am_constraints = []
        am_constraint_violations = []
        am_losses = []
        am_best_total_reward = -1e8
        am_best_iteration = 0

        # Run finetuning loop
        for i in range(1, num_iterations + 1):
            # Solves lean adjoint ODE to create dataset
            dataset = trainer.generate_dataset()
            
            if dataset is None:
                print("Dataset is None, skipping iteration", flush=True)
                continue

            # Fine-tune the model with adjoint matching loss
            loss = trainer.finetune(dataset, steps=config.adjoint_matching.finetune_steps)

            if i % plotting_freq == 0:
                am_losses.append(loss/reward_lambda/(traj_len//2))

                # Generate Samples
                tmp_model = copy.deepcopy(trainer.fine_model)
                new_molecules = sampling(
                    config.reward_sampling,
                    tmp_model,
                    device=device
                )
                del tmp_model
                if args.save_samples and False:
                    new_samples.extend(dgl.unbatch(new_molecules.cpu()))

                # Compute reward for current samples
                tmp_dict = reward_functional.functional(new_molecules)
                del new_molecules
                am_total_rewards.append(tmp_dict["total_reward"])
                am_rewards.append(tmp_dict["reward"])
                am_constraints.append(tmp_dict["constraint"])
                am_constraint_violations.append(tmp_dict["constraint_violations"])

                if am_total_rewards[-1] > am_best_total_reward:
                    am_best_total_reward = am_total_rewards[-1]
                    am_best_iteration = i

                if args.use_wandb:
                    logs = {}
                    logs.update(tmp_dict)
                    logs.update({"loss": am_losses[-1],
                                 "total_best_reward": am_best_total_reward})
                    log = alm.get_statistics()
                    logs.update(log)
                    wandb.log(logs)

                print(f"\tIteration {i}: Total Reward: {am_total_rewards[-1]:.4f}, Reward: {am_rewards[-1]:.4f}, Constraint: {am_constraints[-1]:.4f}, Violations: {am_constraint_violations[-1]:.4f}", flush=True)
                print(f"\tBest reward: {am_best_total_reward:.4f} in step {am_best_iteration}", flush=True)

        al_losses.extend(am_losses)
        al_total_rewards.extend(am_total_rewards)
        al_rewards.extend(am_rewards)
        al_constraints.extend(am_constraints)
        al_constraint_violations.extend(am_constraint_violations)

        gen_model = copy.deepcopy(trainer.fine_model)
        if args.save_model and (k % 5 == 0) and k != lagrangian_updates:
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(gen_model.cpu().state_dict(), save_path / Path(f"model_{k}.pth"))
            print(f"Model saved to {save_path}", flush=True)

        # Print final statistics        
        if al_constraint_violations[-1] < al_lowest_const_violations:
            al_lowest_const_violations = al_constraint_violations[-1]
            al_best_epoch = k
            al_best_reward = al_rewards[-1]

        print(f"Best overall reward: {al_best_reward:.4f} with violations {al_lowest_const_violations:.4f} at epoch {al_best_epoch}", flush=True)

        # Generate Samples and update the augmented lagrangian parameters
        tmp_model = copy.deepcopy(gen_model)
        new_molecules = sampling(
            config.reward_sampling,
            tmp_model,
            device=device
        )
        del tmp_model
        if args.save_samples:
            tmp_save_path = str(sample_path / Path(f"samples_{k}.bin"))
            tmp_samples = [graph for graph in dgl.unbatch(new_molecules.cpu())]
            save_graphs(tmp_save_path, tmp_samples)
            del tmp_samples
        
        constraint_set = reward_functional.constraint(new_molecules)
        alm.update_lambda_rho(constraint_set)
        del new_molecules

        # if alm.convergence():
        #     break

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
    
    if not args.wandb:
        OmegaConf.save(config, save_path / Path("config.yaml"))
        results = {
            "total_rewards": np.array(al_total_rewards),
            "rewards": np.array(al_rewards),
            "constraints": np.array(al_constraints),
            "constraint_violations": np.array(al_constraint_violations),
            "losses": np.array([al_losses[0]] + al_losses),
        }
        np.savez(save_path / "results.npz", **results)
        np.savez(
            save_path / "alm_stats.npz",
            lambda_=np.array(al_stats["lambda"]),
            rho=np.array(al_stats["rho"]),
            expected_constraint=np.array(al_stats["expected_constraint"]),
        )

    # Plotting if enabled
    if args.save_plots:
        from utils.plotting import plot_graphs
        # Plot rewards and constraints
        tmp_data = [al_total_rewards, al_rewards, al_constraints, al_constraint_violations]
        tmp_titles = ["Total Rewards", "Rewards", "Constraints", "Constraint Violations"]
        plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("rewards_constraints.png"), save_freq=plotting_freq)
        # Plot losses
        tmp_data = [al_losses]
        tmp_titles = ["Losses"]
        plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("losses.png"), save_freq=plotting_freq)
        # Plot lambda, rho and expected constraint
        lambda_ = al_stats["lambda"]
        rho_ = al_stats["rho"]
        expected_constraint = al_stats["expected_constraint"]
        # constraint_violations = al_stats["constraint_violations"]
        tmp_data = [lambda_, rho_, expected_constraint]
        tmp_titles = ["Lambda", "Rho", "Expected Constraint"]
        plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("al_stats.png"))
        print(f"Saved plots to {save_path}", flush=True)

    # Save the model if enabled
    if args.save_model:
        torch.save(gen_model.cpu().state_dict(), save_path / Path("final_model.pth"))
        print(f"Model saved to {save_path}", flush=True)

    # Save the samples if enabled
    if args.save_samples:
        # from dgl import save_graphs
        # tmp_save_path = str(save_path / Path("samples.bin"))
        # save_graphs(tmp_save_path, new_samples)
        print(f"Samples saved to {save_path}", flush=True)

    print(f"--- Final ---", flush=True)
    print(f"Final reward: {al_total_rewards[-1]:.4f}", flush=True)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- End ---", flush=True)
    print(f"End time: {end_time}", flush=True)
    print(f"Duration: {(time.time()-start_time)/60:.2f} mins", flush=True)
    print()
    print()


if __name__ == "__main__":
    main()

