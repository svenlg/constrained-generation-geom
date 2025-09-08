# pylint: disable=missing-module-docstring
# pylint: disable=no-name-in-module
import os
import os.path as osp
import time
import copy
import wandb
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from utils.setup import parse_args, update_config_with_args
from utils.utils import extract_trailing_numbers, set_seed
from utils.sampling import sampling

import dgl
import flowmol

from environment import AugmentedReward

from finetuning import AugmentedLagrangian, AdjointMatchingFinetuningTrainerFlowMol

from regessor import GNN

# Load - Flow Model
def setup_gen_model(flow_model: str, device: torch.device): 
    gen_model = flowmol.load_pretrained(flow_model)
    gen_model.to(device)
    return gen_model

# Setup Reward and constraint models
def load_regressor(property: str, date: str, device: torch.device) -> nn.Module:
    model_path = osp.join("pretrained_models", property, date, "best_model.pt")
    state = torch.load(model_path, map_location=device)
    model = GNN(property=property, 
                node_feats=state["config"]["node_feats"],
                edge_feats=state["config"]["edge_feats"],
                hidden_dim=state["config"]["hidden_dim"],
                depth=state["config"]["depth"],
            )
    model.load_state_dict(state["model_state"])
    return model

# # Sampling
# def sampling(config: OmegaConf, model: flowmol.FlowMol, device: torch.device,
#              min_num_atoms: int = None, max_num_atoms: int = None,
#              n_atoms: int = None):
#     model.to(device)
#     new_molecules, _ = model.sample_random_sizes(
#         sampler_type = config.sampler_type,
#         n_molecules = config.num_samples, 
#         n_timesteps = config.num_integration_steps + 1, 
#         device = device,
#         keep_intermediate_graphs = True,
#     )
#     return new_molecules

def main():
    # Parse command line arguments
    args = parse_args()

    # Load config from file
    config_path = Path("configs/augmented_lagrangian.yaml")
    config = OmegaConf.load(config_path)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)

    # Setup - Seed and device
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup - WandB
    use_wandb = args.use_wandb and not args.debug
    if use_wandb:
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
    if use_wandb and ("sweep" in config.experiment):
        save_path = save_path / Path(f"{run_id}")
    if (args.save_samples or args.save_model or args.save_plots) and not args.debug:
        if args.output_end is not None:
            save_path = save_path / Path(f"{args.output_end}")
        save_path = save_path / Path(tmp_time)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Run will be saved at:")
        print(save_path)

    # Molecular generation parameters
    n_atoms = config.get("n_atoms", None)
    min_num_atoms = config.get("min_num_atoms", None)
    max_num_atoms = config.get("max_num_atoms", None)
    # update config.adjoint_matching.sampling
    config.adjoint_matching.sampling.n_atoms = n_atoms
    config.adjoint_matching.sampling.min_num_atoms = min_num_atoms
    config.adjoint_matching.sampling.max_num_atoms = max_num_atoms

    # Augmented Lagrangian Parameters
    lagrangian_updates = config.augmented_lagrangian.lagrangian_updates

    # Adjoint Matching Parameters
    reward_lambda = config.reward_lambda
    traj_len = config.adjoint_matching.sampling.num_integration_steps
    finetune_steps = config.adjoint_matching.sampling.num_samples // config.adjoint_matching.batch_size

    num_iterations = config.total_iterations // lagrangian_updates
    plotting_freq = num_iterations // 5

    baseline = args.baseline
    if baseline:
        base_lambda = config.augmented_lagrangian.base_lambda

    if args.debug:
        config.augmented_lagrangian.sampling.num_samples = config.augmented_lagrangian.sampling.num_samples if torch.cuda.is_available() else 8
        config.adjoint_matching.sampling.num_samples = 16 if torch.cuda.is_available() else 4
        config.adjoint_matching.batch_size = 4
        config.reward_sampling.num_samples = config.reward_sampling.num_samples if torch.cuda.is_available() else 8
        finetune_steps = config.adjoint_matching.sampling.num_samples // config.adjoint_matching.batch_size
        plotting_freq = 1
        args.save_samples = False
        num_iterations = 2
        lagrangian_updates = 2
        print("Debug mode activated", flush=True)

    print(f"--- Start ---", flush=True)
    print(f"Finetuning {config.flow_model} in experiment {config.experiment}", flush=True)
    print(f"Reward: {config.reward.fn} - Constraint: {config.constraint.fn}", flush=True)
    print(f"Maximum Bound: {config.constraint.bound}", flush=True)
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
        print(f"\trho_init: {config.augmented_lagrangian.rho_init}", flush=True)
        print(f"\teta: {config.augmented_lagrangian.eta}", flush=True)
        print(f"\tlagrangian_updates: {lagrangian_updates}", flush=True)

    print(f"Adjoint Matching Parameters", flush=True)
    print(f"\treward_lambda: {reward_lambda}", flush=True)
    print(f"\tlr: {config.adjoint_matching.lr}", flush=True)
    print(f"\tclip_grad_norm: {config.adjoint_matching.clip_grad_norm}", flush=True)
    print(f"\tclip_loss: {config.adjoint_matching.clip_loss}", flush=True)
    print(f"\tbatch_size: {config.adjoint_matching.batch_size}", flush=True)
    print(f"\tsampling.num_samples: {config.adjoint_matching.sampling.num_samples}", flush=True)
    print(f"\tsampling.num_integration_steps: {config.adjoint_matching.sampling.num_integration_steps}", flush=True)
    print(f"\tfinetune_steps: {finetune_steps}", flush=True)
    print(f"\tnum_iterations: {num_iterations}", flush=True)

    print(f"Molecular Generation Parameters", flush=True)
    print(f"\tn_atoms: {n_atoms}", flush=True)
    print(f"\tmin_num_atoms: {min_num_atoms}", flush=True)
    print(f"\tmax_num_atoms: {max_num_atoms}", flush=True)

    # Setup - Gen Model
    base_model = setup_gen_model(config.flow_model, device=device)
    gen_model = copy.deepcopy(base_model)

    # Setup - Reward and Constraint Functions
    reward_model = load_regressor(config.reward.fn, config.reward.date, device=device)
    constraint_model = load_regressor(config.constraint.fn, config.constraint.date, device=device)

    # Setup - Environment, AugmentedReward, ConstraintModel
    augmented_reward = AugmentedReward(
        reward_fn = reward_model,
        constraint_fn = constraint_model,
        alpha = reward_lambda,
        bound = config.constraint.bound,
        device = device,
    )
    
    # Set the initial lambda and rho
    augmented_reward.set_lambda_rho(
        lambda_ = 0.0, 
        rho_ = config.augmented_lagrangian.rho_init,
    )

    # Initialize lists to store loss and rewards
    al_stats = []
    full_stats = []
    al_best_reward = -1e8
    al_lowest_const_violations = 1.0
    al_best_epoch = 0
    if args.save_samples:
        from dgl import save_graphs
        sample_path = save_path / Path("samples")
        sample_path.mkdir(parents=True, exist_ok=True)

    # Generate Samples
    tmp_model = copy.deepcopy(gen_model)
    dgl_mols, rd_mols = sampling(
        config.reward_sampling,
        tmp_model,
        device=device,
        n_atoms=n_atoms,
        min_num_atoms=min_num_atoms,
        max_num_atoms=max_num_atoms,
    )
    del tmp_model
    if args.save_samples:
        tmp_save_path = str(sample_path / Path("samples_0.bin"))
        tmp_samples = [graph for graph in dgl.unbatch(dgl_mols.cpu())]
        save_graphs(tmp_save_path, tmp_samples)
        del tmp_samples

    # Compute reward for current samples
    _ = augmented_reward(dgl_mols)
    tmp_log = augmented_reward.get_statistics()
    full_stats.append(tmp_log)

    al_lowest_const_violations = full_stats[-1]["constraint_violations"]
    al_best_reward = full_stats[-1]["reward"]

    alm = AugmentedLagrangian(
        config = config.augmented_lagrangian,
        constraint_fn = constraint_model,
        bound = config.constraint.bound,
        device = device,
    )
    # Set initial expected constraint (only needed for logging)
    _ = alm.expected_constraint(dgl_mols)

    # Log al initial states
    if use_wandb:
        logs = {}
        logs.update(tmp_log)
        logs.update({"total_best_reward": al_best_reward})
        log = alm.get_statistics()
        logs.update(log) # lambda, rho, expected_constraint
        wandb.log(logs)

    #### AUGMENTED LAGRANGIAN - BEGIN ####
    total_steps_made = 0
    for k in range(1, lagrangian_updates + 1):
        print(f"--- AL Round {k}/{lagrangian_updates} ---", flush=True)

        # Get the current lambda and rho
        lambda_, rho_ = alm.get_current_lambda_rho()
        log = alm.get_statistics()
        al_stats.append(log)

        # Set the lambda and rho in the reward functional
        augmented_reward.set_lambda_rho(lambda_, rho_)

        # Print lambda and rho for the current round
        print(f"Lambda: {lambda_:.4f}, rho: {rho_:.4f}", flush=True)

        # Set up - Adjoint Matching
        trainer = AdjointMatchingFinetuningTrainerFlowMol(
            config = config.adjoint_matching,
            model = copy.deepcopy(gen_model),
            base_model = copy.deepcopy(base_model),
            grad_reward_fn = augmented_reward.grad_augmented_reward_fn,
            device = device,
            verbose = args.debug,
        )

        am_stats = []
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
            loss = trainer.finetune(dataset, steps=finetune_steps)
            del dataset
            
            # if i % plotting_freq == 0:
            total_steps_made += 1
            if total_steps_made % 10 == 0:

                # Generate Samples
                tmp_model = copy.deepcopy(trainer.fine_model)
                dgl_mols, rd_mols = sampling(
                    config.reward_sampling,
                    tmp_model,
                    device=device,
                    n_atoms=n_atoms,
                    min_num_atoms=min_num_atoms,
                    max_num_atoms=max_num_atoms,
                )
                del tmp_model

                # Compute reward for current samples
                _ = augmented_reward(dgl_mols)
                del dgl_mols
                tmp_log = augmented_reward.get_statistics()
                tmp_log["loss"] = loss/reward_lambda/(traj_len//2)
                am_stats.append(tmp_log)

                if am_stats[-1]["total_reward"] > am_best_total_reward:
                    am_best_total_reward = am_stats[-1]["total_reward"]
                    am_best_iteration = i

                if use_wandb:
                    logs = {}
                    logs.update(tmp_log)
                    logs.update({"loss": tmp_log["loss"],
                                 "total_best_reward": am_best_total_reward})
                    log = alm.get_statistics()
                    logs.update(log)
                    wandb.log(logs)

                print(f"\tIteration {i}: Total Reward: {am_stats[-1]['total_reward']:.4f}, Reward: {am_stats[-1]['reward']:.4f}, "
                      f"Constraint: {am_stats[-1]['constraint']:.4f}, Violations: {am_stats[-1]['constraint_violations']:.4f}", flush=True)
                print(f"\tBest reward: {am_best_total_reward:.4f} in step {am_best_iteration}", flush=True)
        


        full_stats.extend(am_stats)

        gen_model = copy.deepcopy(trainer.fine_model)
        if args.save_model and (k % 5 == 0) and k != lagrangian_updates:
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(gen_model.cpu().state_dict(), save_path / Path(f"model_{k}.pth"))
            print(f"Model saved to {save_path}", flush=True)

        # Print final statistics
        if full_stats[-1]["constraint_violations"] < al_lowest_const_violations:
            al_lowest_const_violations = full_stats[-1]["constraint_violations"]
            al_best_epoch = k
            al_best_reward = full_stats[-1]["reward"]

        print(f"Best overall reward: {al_best_reward:.4f} with violations {al_lowest_const_violations:.4f} at epoch {al_best_epoch}", flush=True)

        # Generate Samples and update the augmented lagrangian parameters
        tmp_model = copy.deepcopy(gen_model)
        dgl_mols, rd_mols = sampling(
            config.reward_sampling,
            tmp_model,
            device=device,
            n_atoms=n_atoms,
            min_num_atoms=min_num_atoms,
            max_num_atoms=max_num_atoms
        )
        del tmp_model
        if args.save_samples:
            tmp_save_path = str(sample_path / Path(f"samples_{k}.bin"))
            tmp_samples = [graph for graph in dgl.unbatch(dgl_mols.cpu())]
            save_graphs(tmp_save_path, tmp_samples)
            del tmp_samples

        alm.update_lambda_rho(dgl_mols)
        del dgl_mols, rd_mols, trainer
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    if not args.debug:
        OmegaConf.save(config, save_path / Path("config.yaml"))
        full_stats[0]['loss'] = full_stats[1]['loss']
        df_al = pd.DataFrame.from_records(full_stats)
        df_al.to_csv(save_path / "full_stats.csv", index=False)
        df_alm = pd.DataFrame.from_dict(al_stats)
        df_alm.to_csv(save_path / "al_stats.csv", index=False)

    # Plotting if enabled
    if args.save_plots and not args.debug:
        from utils.plotting import plot_graphs
        # Plot rewards and constraints
        full_stats[0]['loss'] = full_stats[1]['loss']
        df = pd.DataFrame.from_records(full_stats)
        tmp_data = [df['total_reward'], df['reward'], df['constraint'], df['constraint_violations'], df['loss']]
        tmp_titles = ["Total Reward", "Reward", "Constraint", "Constraint Violations", "Loss"]
        plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("full_stats.png"), save_freq=plotting_freq)
        # Plot lambda, rho and expected constraint
        df_alm = pd.DataFrame.from_dict(al_stats)
        tmp_data = [df_alm["lambda"], df_alm["rho"], df_alm["expected_constraint"]]
        tmp_titles = ["Lambda", "Rho", "Expected Constraint"]
        plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("al_stats.png"))
        print(f"Saved plots to {save_path}", flush=True)

    # Save the model if enabled
    if args.save_model and not args.debug:
        torch.save(gen_model.cpu().state_dict(), save_path / Path("final_model.pth"))
        print(f"Model saved to {save_path}", flush=True)

    # Save the samples if enabled
    if args.save_samples and not args.debug:
        print(f"Samples saved to {save_path}", flush=True)

    print(f"--- Final ---", flush=True)
    print(f"Final reward: {full_stats[-1]['reward']:.4f}", flush=True)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- End ---", flush=True)
    print(f"End time: {end_time}", flush=True)
    print(f"Duration: {(time.time()-start_time)/60:.2f} mins", flush=True)
    print()
    print()


if __name__ == "__main__":
    main()

