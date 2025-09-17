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


from utils import parse_args, update_config_with_args, set_seed, sampling
from true_rc import pred_vs_real
from regessor import setup_fine_tuner, finetune

import dgl
import flowmol

from environment import AugmentedReward

from finetuning import AugmentedLagrangian, AdjointMatchingFinetuningTrainerFlowMol

from regessor import GNN, MoleculeGNN

# Load - Flow Model
def setup_gen_model(flow_model: str, device: torch.device): 
    gen_model = flowmol.load_pretrained(flow_model)
    gen_model.to(device)
    return gen_model

# Setup Reward and constraint models
def load_regressor(config: OmegaConf, device: torch.device) -> nn.Module:
    K_x = 3  # number of spatial dimensions (3D)
    K_a = 10 # number of atom features
    K_c = 6  # number of charge classes (0, +1, -1, +2)
    K_e = 5  # number of bond types (none, single, double, triple, aromatic)
    model_path = osp.join("pretrained_models", config.fn, config.model_type, config.date, "best_model.pt")
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
            "use_gumbel": state["config"]["use_gumbel"],
            "equivariant": state["config"]["equivariant"],
        }
        model = MoleculeGNN(**model_config)

    model.load_state_dict(state["model_state"])
    return model, OmegaConf.create(model_config)

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
    run_id = datetime.now().strftime("%m%d_%H%M")
    if config.experiment is not None:
        run_name = f"{run_id}_{config.experiment}"
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
        save_path = save_path / Path(f"{sweep_id}") /Path(f"{wandb.run.id}")
    if (args.save_samples or args.save_model or args.save_plots) and not args.debug:
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

    # Online fintuning
    config.rc_finetune = config.get("rc_finetune", OmegaConf.create({}))
    config.rc_finetune.freq = config.rc_finetune.get("freq", 0)

    num_iterations = config.total_steps // lagrangian_updates
    plotting_freq = 10

    baseline = args.baseline
    if baseline:
        base_lambda = config.augmented_lagrangian.base_lambda

    # Setup - Gen Model
    base_model = setup_gen_model(config.flow_model, device=device)
    gen_model = copy.deepcopy(base_model)

    # Setup - Reward and Constraint Functions
    reward_model, reward_model_config = load_regressor(config.reward, device=device)
    if config.rc_finetune is not None and config.reward.fine_tuning:
        reward_finetuner = setup_fine_tuner(config.reward.fn, reward_model, config.rc_finetune)
    constraint_model, constraint_model_config = load_regressor(config.constraint, device=device)
    if config.rc_finetune is not None and config.constraint.fine_tuning:
        constraint_finetuner = setup_fine_tuner(config.constraint.fn, constraint_model, config.rc_finetune)
    rc_fine_tune_freq = config.rc_finetune.freq if config.reward.fine_tuning or config.constraint.fine_tuning else 0

    if args.debug:
        config.augmented_lagrangian.sampling.num_samples = config.augmented_lagrangian.sampling.num_samples if torch.cuda.is_available() else 8
        config.adjoint_matching.sampling.num_samples = 20 if torch.cuda.is_available() else 4
        config.adjoint_matching.batch_size = 5 if torch.cuda.is_available() else 2
        config.reward_sampling.num_samples = config.reward_sampling.num_samples if torch.cuda.is_available() else 8
        finetune_steps = config.adjoint_matching.sampling.num_samples // config.adjoint_matching.batch_size
        plotting_freq = 1
        args.save_samples = False
        num_iterations = 2
        lagrangian_updates = 2
        rc_fine_tune_freq = 1
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
    print(f"\tlct: {config.adjoint_matching.lct}", flush=True)
    print(f"\tbatch_size: {config.adjoint_matching.batch_size}", flush=True)
    print(f"\tsampling.num_samples: {config.adjoint_matching.sampling.num_samples}", flush=True)
    print(f"\tsampling.num_integration_steps: {config.adjoint_matching.sampling.num_integration_steps}", flush=True)
    print(f"\tfinetune_steps: {finetune_steps}", flush=True)
    print(f"\tnum_iterations: {num_iterations}", flush=True)

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
        config = config.augmented_reward,
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
    # Compare with true value
    pred_rc = augmented_reward.get_reward_constraint()
    log_pred_vs_real, _, _ = pred_vs_real(rd_mols, pred_rc, reward=config.reward.fn, constraint=config.constraint.fn)

    al_lowest_const = full_stats[-1]["constraint"]
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
        logs.update(log_pred_vs_real)
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
                log_pred_vs_real, true_reward, true_constraint = pred_vs_real(rd_mols, pred_rc, reward=config.reward.fn, constraint=config.constraint.fn)
                tmp_log = augmented_reward.get_statistics()
                tmp_log["loss"] = loss/reward_lambda/(traj_len//2)
                am_stats.append(tmp_log)

                if am_stats[-1]["total_reward"] > am_best_total_reward:
                    am_best_total_reward = am_stats[-1]["total_reward"]
                    am_best_iteration = i
                tmp_log["total_best_reward"] = am_best_total_reward                
                
                if rc_fine_tune_freq > 0 and (i % rc_fine_tune_freq == 0):
                    if config.reward.fine_tuning:
                        r_history = finetune(
                            finetuner = reward_finetuner,
                            data = [mol for mol in dgl.unbatch(dgl_mols.cpu())],
                            targets = true_reward,
                            config = config.rc_finetune,
                        )
                    if config.constraint.fine_tuning:
                        c_history = finetune(
                            finetuner = constraint_finetuner,
                            data = [mol for mol in dgl.unbatch(dgl_mols.cpu())],
                            targets = true_constraint,
                            config = config.rc_finetune,
                        )
                
                if use_wandb:
                    logs = {}
                    logs.update(tmp_log)
                    logs.update(log_pred_vs_real)
                    log = alm.get_statistics()
                    if config.reward.fine_tuning:
                        logs.update(r_history)
                    if config.constraint.fine_tuning:
                        logs.update(c_history)
                    logs.update(log)
                    wandb.log(logs)

                del dgl_mols, rd_mols, true_reward, true_constraint, pred_rc, log_pred_vs_real, tmp_log

                print(f"\tIteration {i}: Total Reward: {am_stats[-1]['total_reward']:.4f}, Reward: {am_stats[-1]['reward']:.4f}, "
                      f"Constraint: {am_stats[-1]['constraint']:.4f}, Violations: {am_stats[-1]['constraint_violations']:.4f}", flush=True)
                print(f"\tBest reward: {am_best_total_reward:.4f} in step {am_best_iteration}", flush=True)
        
        full_stats.extend(am_stats)

        gen_model = copy.deepcopy(trainer.fine_model)

        # Print final statistics
        if full_stats[-1]["constraint"] < al_lowest_const:
            al_lowest_const = full_stats[-1]["constraint"]
            al_best_epoch = k
            al_best_reward = full_stats[-1]["reward"]

        print(f"Best overall reward: {al_best_reward:.4f} with violations {al_lowest_const:.4f} at epoch {al_best_epoch}", flush=True)

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

        alm.update_lambda_rho(dgl_mols)
        del dgl_mols, rd_mols, trainer
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    if not args.debug:
        # Save configs is config path
        config_save_path = save_path / Path("configs")
        config_save_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config_save_path / Path("config.yaml"))
        OmegaConf.save(reward_model_config, config_save_path / Path("reward_model_config.yaml"))
        OmegaConf.save(constraint_model_config, config_save_path / Path("constraint_model_config.yaml"))
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

