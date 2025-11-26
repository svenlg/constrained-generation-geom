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

from utils import parse_arguments, update_config_with_args, set_seed, sampling
from true_rc import pred_vs_real

import dgl
import flowmol

from environment import AugmentedReward

from finetuning import AugmentedLagrangian, AdjointMatchingFinetuningTrainerFlowMol

from regessor import GNN, EGNN

from true_rc import bond_distance, connectivity_matrix_and_loss

# Load - Flow Model
def setup_gen_model(flow_model: str, device: torch.device): 
    gen_model = flowmol.load_pretrained(flow_model)
    gen_model.to(device)
    return gen_model

# Setup Reward and constraint models
def load_regressor(config: OmegaConf, device: torch.device) -> nn.Module:
    K_x = 3  # number of spatial dimensions (3D)
    K_a = 10 # number of atom features
    K_c = 6  # number of charge classes
    K_e = 5  # number of bond types (none, single, double, triple, aromatic)
    print(config.fn, config.model_type, config.date)
    model_path = osp.join("pretrained_models", str(config.fn), str(config.model_type), str(config.date), "best_model.pt")
    state = torch.load(model_path, map_location=device)
    if config.model_type == "gnn":
        model_config = {
            "property": state["config"]["property"],
            "node_feats": K_a + K_c + K_x,
            "edge_feats": K_e,
            "hidden_dim": state["config"]["hidden_dim"],
            "depth": state["config"]["depth"],
        }
        model = GNN(**model_config)
    elif config.model_type == "egnn":
        model_config = {
            "property": state["config"]["property"],
            "num_atom_types": K_a,
            "num_charge_classes": K_c,
            "num_bond_types": K_e,
            "hidden_dim": state["config"]["hidden_dim"],
            "depth": state["config"]["depth"],
        }
        model = EGNN(**model_config)
    model.load_state_dict(state["model_state"])
    return model

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load config from file
    config_path = Path("configs/augmented_lagrangian.yaml")
    config = OmegaConf.load(config_path)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)
    baseline = args.baseline
    config.baseline = baseline

    # Setup - Seed and device
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup - WandB
    use_wandb = args.use_wandb and not args.debug
    run_id = datetime.now().strftime("%m%d_%H%M")
    if config.experiment is not None and "sweep" in config.experiment:
        run_name = f"lu{config.augmented_lagrangian.lagrangian_updates}_rl{config.reward_lambda}_rho{config.augmented_lagrangian.rho_init}_seed{config.seed}"
    elif config.experiment is not None:
        run_name = f"{run_id}_{config.experiment}_{config.seed}"
    else:
        run_name = f"{run_id}_r{config.reward.model_type}_c{config.constraint.model_type}{config.constraint.bound}_rf{config.reward_lambda}_lu{config.augmented_lagrangian.lagrangian_updates}"
    print(f"Running: {run_name}")
    if use_wandb:
        wandb.init(name=run_name, config=OmegaConf.to_container(config, resolve=True))
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else None
        if sweep_id is not None:
            print(f"WandB sweep ID: {sweep_id} - Run ID: {wandb.run.id}", flush=True)

    save_path = Path(config.root) / Path("aa_experiments")
    if use_wandb and sweep_id is not None:
        save_path = save_path / Path(f"{config.experiment}") / Path(f"{config.seed}_{wandb.run.id}")
    else:
        save_path = save_path / Path(f"{config.experiment}") / Path(f"{run_name}")
    if (args.save_samples or args.save_model or args.save_plots) and not args.debug and not use_wandb:
        save_path = save_path / Path(run_name)
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
    config.adjoint_matching.reward_lambda = reward_lambda
    traj_len = config.adjoint_matching.sampling.num_integration_steps
    finetune_steps = config.adjoint_matching.sampling.num_samples // config.adjoint_matching.batch_size

    num_iterations = config.total_steps // lagrangian_updates
    plotting_freq = 3 if args.plotting_freq is None else args.plotting_freq

    if baseline:
        base_lambda = config.augmented_lagrangian.lambda_init

    # Setup - Gen Model
    base_model = setup_gen_model(config.flow_model, device=device)
    fine_model = copy.deepcopy(base_model)

    # Setup - Reward Functions
    reward_model = load_regressor(config.reward, device=device)

    # Setup - Constraint Functions
    if config.constraint.fn in ["score", "energy", "sascore"]:
        constraint_model = load_regressor(config.constraint, device=device)
    elif config.constraint.fn == "interatomic_distances":
        constraint_model = bond_distance
    elif config.constraint.fn == "interatomic_distances_new":
        constraint_model = connectivity_matrix_and_loss
    else:
        raise ValueError(f"Unknown constraint function: {config.constraint.fn}")

    if args.debug:
        config.augmented_lagrangian.sampling.num_samples = 8
        config.adjoint_matching.sampling.num_samples = 20 if torch.cuda.is_available() else 4
        config.adjoint_matching.batch_size = 5 if torch.cuda.is_available() else 2
        config.reward_sampling.num_samples = 8
        finetune_steps = config.adjoint_matching.sampling.num_samples // config.adjoint_matching.batch_size
        plotting_freq = 1
        args.save_samples = False
        num_iterations = 2
        lagrangian_updates = 2
        print("Debug mode activated", flush=True)

    print(f"--- Start ---", flush=True)
    print(f"Finetuning {config.flow_model}", flush=True)
    print(f"Reward: {config.reward.fn} - Constraint: {config.constraint.fn}", flush=True)
    print(f"Maximum Bound: {config.constraint.bound}", flush=True)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_time}", flush=True)
    print(f"Device: {device}", flush=True)
    start_time = time.time()

    print(f"--- Config ---", flush=True)
    print(f"Augmented Lagrangian Parameters", flush=True)
    if baseline:
        print(f"\tbaseline activated", flush=True)
        print(f"\tbase_lambda: {base_lambda}", flush=True)
    else:
        print(f"\tlagrangian_updates: {lagrangian_updates}", flush=True)
        print(f"\trho_init: {config.augmented_lagrangian.rho_init}", flush=True)
        print(f"\teta: {config.augmented_lagrangian.eta}", flush=True)
        print(f"\tau: {config.augmented_lagrangian.tau}", flush=True)
        print(f"\tlambda_min: {config.augmented_lagrangian.lambda_min}", flush=True)

    print(f"Adjoint Matching Parameters", flush=True)
    print(f"\treward_lambda: {reward_lambda}", flush=True)
    print(f"\tlr: {config.adjoint_matching.lr}", flush=True)
    print(f"\tclip_grad_norm: {config.adjoint_matching.clip_grad_norm}", flush=True)
    print(f"\tlct: {config.adjoint_matching.lct}", flush=True)
    print(f"\tbatch_size: {config.adjoint_matching.batch_size}", flush=True)
    print(f"\tsampling.num_samples: {config.adjoint_matching.sampling.num_samples}", flush=True)
    print(f"\tsampling.num_integration_steps: {config.adjoint_matching.sampling.num_integration_steps}", flush=True)
    print(f"\tfinetune_steps: {finetune_steps}", flush=True)
    print(f"\tnum_iterations: {num_iterations}", flush=True)
    if "features" in config.adjoint_matching:
        print(f"\tfeatures: {config.adjoint_matching.features}", flush=True)

    print(f"Molecular Generation Parameters", flush=True)
    print(f"\tn_atoms: {n_atoms}", flush=True)
    print(f"\tmin_num_atoms: {min_num_atoms}", flush=True)
    print(f"\tmax_num_atoms: {max_num_atoms}", flush=True)

    # Setup - Environment, AugmentedReward, ConstraintModel
    augmented_reward = AugmentedReward(
        reward_fn = reward_model,
        constraint_fn = constraint_model,
        alpha = reward_lambda,
        bound = config.constraint.bound,
        device = device,
        baseline = baseline
    )
    
    # Set the initial lambda and rho
    augmented_reward.set_lambda_rho(
        lambda_ = config.augmented_lagrangian.lambda_init, 
        rho_ = 0.0 if baseline else config.augmented_lagrangian.rho_init,
    )

    # Initialize lists to store loss and rewards
    al_stats = []
    full_stats = []
    al_best_reward = -1e8
    al_best_epoch = 0
    if args.save_samples:
        from dgl import save_graphs
        sample_path = save_path / Path("samples")
        sample_path.mkdir(parents=True, exist_ok=True)

    # Generate Samples
    tmp_model = copy.deepcopy(fine_model)
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
    
    # Compare with true value
    pred_rc = augmented_reward.get_reward_constraint()
    log_pred_vs_real, _, _ = pred_vs_real(rd_mols, dgl_mols, pred_rc, reward=config.reward.fn, constraint=config.constraint.fn)
    full_stats[-1].update(log_pred_vs_real)

    al_lowest_const = full_stats[-1]["constraint"]
    al_best_reward = full_stats[-1]["reward"]

    alm = AugmentedLagrangian(
        config = config.augmented_lagrangian,
        constraint_fn = constraint_model,
        bound = config.constraint.bound,
        device = device,
        baseline = baseline
    )
    # Set initial expected constraint (only needed for logging)
    _ = alm.expected_constraint(dgl_mols)

    # Log al initial states
    if use_wandb:
        logs = {}
        logs.update(tmp_log)
        logs.update(log_pred_vs_real)
        logs.update({"total_best_reward": al_best_reward})
        log = alm.get_statistics()
        logs.update(log) # lambda, rho, expected_constraint
        wandb.log(logs)

    # Save the model if enabled
    if args.save_model and not args.debug:
        models_list = []

    #### AUGMENTED LAGRANGIAN - BEGIN ####

    alg_time = time.time()

    total_steps_made = 0
    for k in range(1, lagrangian_updates + 1):
        print(f"--- AL Round {k}/{lagrangian_updates} ---", flush=True)

        # Get the current lambda and rho
        lambda_, rho_ = alm.get_current_lambda_rho()
        log = alm.get_statistics()
        al_stats.append(log)

        # Set the lambda and rho in the augmented reward
        augmented_reward.set_lambda_rho(lambda_, rho_)

        # Print lambda and rho for the current round
        print(f"Lambda: {lambda_:.4f}, rho: {rho_:.4f}", flush=True)

        # Set up - Adjoint Matching
        trainer = AdjointMatchingFinetuningTrainerFlowMol(
            config = config.adjoint_matching,
            model = copy.deepcopy(fine_model),
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
            
            total_steps_made += 1
            if total_steps_made % plotting_freq == 0:

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
                if args.save_samples:
                    tmp_save_path = str(sample_path / Path(f"samples_{total_steps_made}.bin"))
                    tmp_samples = [graph for graph in dgl.unbatch(dgl_mols.cpu())]
                    save_graphs(tmp_save_path, tmp_samples)
                    del tmp_samples

                # Compute reward for current samples
                _ = augmented_reward(dgl_mols)
                pred_rc = augmented_reward.get_reward_constraint()
                log_pred_vs_real, _, _ = pred_vs_real(rd_mols, dgl_mols, pred_rc, reward=config.reward.fn, constraint=config.constraint.fn)
                tmp_log = augmented_reward.get_statistics()
                tmp_log["loss"] = loss/reward_lambda/(traj_len//2)
                tmp_log.update(log_pred_vs_real)
                am_stats.append(tmp_log)

                if am_stats[-1]["total_reward"] > am_best_total_reward:
                    am_best_total_reward = am_stats[-1]["total_reward"]
                    am_best_iteration = i
                tmp_log["total_best_reward"] = am_best_total_reward                
                
                if use_wandb:
                    logs = {}
                    logs.update(tmp_log)
                    logs.update(log_pred_vs_real)
                    log = alm.get_statistics()
                    logs.update(log)
                    wandb.log(logs)

                del dgl_mols, rd_mols, pred_rc, log_pred_vs_real, tmp_log

                print(f"\tIteration {i}: Total Reward: {am_stats[-1]['total_reward']:.4f}, Reward: {am_stats[-1]['reward']:.4f}, "
                      f"Constraint: {am_stats[-1]['constraint']:.4f}, Violations: {am_stats[-1]['constraint_violations']:.4f}", flush=True)
                print(f"\tBest reward: {am_best_total_reward:.4f} in step {am_best_iteration}", flush=True)
        
        full_stats.extend(am_stats)

        fine_model = copy.deepcopy(trainer.fine_model)

        # Print final statistics
        if full_stats[-1]["constraint"] < al_lowest_const:
            al_lowest_const = full_stats[-1]["constraint"]
            al_best_epoch = k
            al_best_reward = full_stats[-1]["reward"]

        print(f"Best overall reward: {al_best_reward:.4f} with violations {al_lowest_const:.4f} at epoch {al_best_epoch}", flush=True)

        # Generate Samples and update the augmented lagrangian parameters
        tmp_model = copy.deepcopy(fine_model)
        dgl_mols, rd_mols = sampling(
            config.reward_sampling,
            tmp_model,
            device=device,
            n_atoms=n_atoms,
            min_num_atoms=min_num_atoms,
            max_num_atoms=max_num_atoms
        )
        del tmp_model

        # Save the model if enabled
        if args.save_model and not args.debug:
            models_list.append(copy.deepcopy(fine_model.cpu().state_dict()))

        alm.update_lambda_rho(dgl_mols)
        del dgl_mols, rd_mols, trainer

    alg_time = time.time() - alg_time
    print()
    print(f"--- Finished --- {config.total_steps} total-steps---", flush=True)
    print(f"Time: {alg_time/60:.2f} mins", flush=True)
    print()

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    full_stats[0]['loss'] = full_stats[1]['loss']
    df_al = pd.DataFrame.from_records(full_stats)
    df_alm = pd.DataFrame.from_dict(al_stats)
    
    if not args.debug:
        # Save configs is config path
        save_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, save_path / Path("config.yaml"))
        df_al.to_csv(save_path / "full_stats.csv", index=False)
        df_alm.to_csv(save_path / "al_stats.csv", index=False)

    # Plotting if enabled
    if args.save_plots and not args.debug:
        try:
            from utils.plotting import plot_graphs
            # Plot rewards and constraints
            tmp_data = [df_al['total_reward'], df_al['reward'], df_al['constraint'], df_al['constraint_violations'], df_al['loss']]
            tmp_titles = ["Total Reward", "Reward", "Constraint", "Constraint Violations", "Loss"]
            plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("full_stats.png"), save_freq=plotting_freq)
            # Plot lambda, rho and expected constraint
            tmp_data = [df_alm["alm/lambda"], df_alm["alm/rho"], df_alm["alm/expected_constraint"]]
            tmp_titles = ["Lambda", "Rho", "Expected Constraint"]
            plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("al_stats.png"))
            print(f"Saved plots to {save_path}", flush=True)
        except Exception as e:
            print(f"Could not save plots: {e}", flush=True)

    # Save the model if enabled
    if args.save_model and not args.debug:
        for idx, state_dict in enumerate(models_list):
            torch.save(state_dict, save_path / Path(f"model_lu{idx+1}.pth"))
        torch.save(fine_model.cpu().state_dict(), save_path / Path("final_model.pth"))
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

