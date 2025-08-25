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

from environment.property_calculation import compute_property_stats, compute_property_grad 

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
    config_path = Path("configs/adjoint_matching.yaml")
    config = OmegaConf.load(config_path)

    # Update config with command line arguments
    config = update_config_with_args(config, args)
    if args.debug:
        print(OmegaConf.to_yaml(config, resolve=True))

    # Setup - Seed and device
    seed_everything(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup - Save path
    save_path = Path(config.root) / Path("aa_experiments") / Path(config.experiment) / Path(config.reward.fn)
    tmp_time = datetime.now().strftime("%m-%d-%H")
    save_path = save_path / Path(tmp_time)
    if (args.save_samples or args.save_model or args.save_plots or (not args.use_wandb)) and not args.debug:
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Run will be saved at:")
        print(save_path)

    # Settings
    config.verbose = args.verbose

    # General Parameters
    reward = config.reward.fn
    flow_model = config.flow_model

    # Adjoint Matching Parameters
    reward_lambda = config.reward_lambda
    learning_rate = config.adjoint_matching.lr
    clip_grad_norm = config.adjoint_matching.clip_grad_norm
    clip_loss = config.adjoint_matching.clip_loss
    batch_size = config.adjoint_matching.batch_size
    traj_len = config.adjoint_matching.sampling.num_integration_steps
    finetune_steps = config.adjoint_matching.finetune_steps
    num_iterations = config.adjoint_matching.num_iterations
    config.adjoint_matching.sampling.num_samples = finetune_steps * batch_size
    traj_samples_per_stage = config.adjoint_matching.sampling.num_samples

    if args.debug:
        config.reward_sampling.sampling = 2
        config.adjoint_matching.batch_size = 2
        config.adjoint_matching.finetune_steps = 2

    print(f"--- Start ---", flush=True)
    # Setup - WandB
    if args.use_wandb and not args.debug:
        wandb.init()
        run_name = wandb.run.name  # e.g., "test-sweep-12"
        run_number = extract_trailing_numbers(run_name)  # 12
        run_id = wandb.run.id # e.g., "ame6uc42"
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "No_sweep"
        print(f"Run #{run_number} - ID: {run_id}", flush=True)

    # Prints
    print(f"Finetuning {flow_model} in experiment {config.experiment}", flush=True)
    print(f"Reward: {reward}", flush=True)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_time}", flush=True)
    print(f"Device: {device}", flush=True)
    start_time = time.time()

    print(f"--- Config ---", flush=True)

    print(f"Adjoint Matching Parameters", flush=True)
    print(f"\treward_lambda: {reward_lambda}", flush=True)
    print(f"\tlr: {learning_rate}", flush=True)
    print(f"\tclip_grad_norm: {clip_grad_norm}", flush=True)
    print(f"\tclip_loss: {clip_loss}", flush=True)
    print(f"\tbatch_size: {batch_size}", flush=True)
    print(f"\tfinetune_steps: {finetune_steps}", flush=True)
    print(f"\tnum_iterations: {num_iterations}", flush=True)
    print(f"\tsampling.num_samples: {traj_samples_per_stage}", flush=True)
    print(f"\tsampling.num_integration_steps: {traj_len}", flush=True)

    # Setup - Gen Model
    gen_model = setup_gen_model(config.flow_model, device=device)

    # Setup - Reward and Gradient Functions
    if reward == "sascore":
        reward_model = setup_pamnet_model(config.pamnet, device=device)
    else:
        reward_model = None

    def reward_fn(molecules):
        return compute_property_stats(
            molecules = molecules,
            property = reward,
            device = device,
            model = reward_model,
        )

    def grad_reward_fn(molecules):
        with torch.enable_grad():
            vals, grads = compute_property_grad(
                molecules = molecules,
                property = reward,
                reward_lambda = reward_lambda,
                device = device,
                model = reward_model,
            )
            return grads

    # Set up - Adjoint Matching
    trainer = AdjointMatchingFinetuningTrainerFlowMol(
        config = config.adjoint_matching,
        model = copy.deepcopy(gen_model),
        base_model = copy.deepcopy(gen_model),
        grad_reward_fn = grad_reward_fn,
        device = device,
        verbose = False,
    )

    # Initialize lists to store loss and rewards
    losses = []
    metrics = {
        "mean": [], "std": [], "median": [], "max": [], "min": [], "num_invalid": [],
    }
    if args.save_samples and not args.debug:
        new_samples = []

    # Generate Samples
    new_molecules = sampling(
        config.reward_sampling,
        copy.deepcopy(gen_model),
        device=device
    )
    if args.save_samples and not args.debug:
        new_samples.extend(dgl.unbatch(new_molecules.cpu()))

    tmp_dict = reward_fn(new_molecules)
    for key in metrics:
        metrics[key].append(tmp_dict[key])
    current_best_reward = metrics['mean'][-1]

    print(f"Initial reward: {metrics['mean'][-1]:.4f} at {metrics['num_invalid'][-1]} invalid", flush=True)

    best_epoch = 0
    if args.use_wandb and not args.debug:
        tmp_dict['best_reward'] = current_best_reward
        wandb.log(tmp_dict)

    # Run finetuning loop
    for i in range(1, num_iterations + 1):
        # Solves lean adjoint ODE to create dataset
        dataset = trainer.generate_dataset()
        
        if dataset is None:
            print("Dataset is None, skipping iteration", flush=True)
            continue
        
        # Fine-tune the model with adjoint matching loss
        loss = trainer.finetune(dataset)

        if i%1 == 0:
            losses.append(loss/(traj_len//2))
            
            # Generate Samples
            new_molecules = sampling(
                config.reward_sampling,
                copy.deepcopy(trainer.fine_model),
                device=device
            )
            if args.save_samples and not args.debug:
                new_samples.extend(dgl.unbatch(new_molecules.cpu()))

            # Compute appropriate reward for evaluation
            tmp_dict = reward_fn(new_molecules)
            for key in metrics:
                metrics[key].append(tmp_dict[key])

            if metrics['mean'][-1] > current_best_reward:
                current_best_reward = metrics['mean'][-1]
                best_epoch = i

            if args.use_wandb and not args.debug:
                tmp_dict['best_reward'] = current_best_reward
                wandb.log(tmp_dict)

            print(f"Iteration {i}:", flush=True)
            print(f"Reward: {metrics['mean'][-1]:.4f} ({metrics['std'][-1]:.4f}) at {metrics['num_invalid'][-1]} invalid", flush=True)
            print(f"Loss: {losses[-1]:.4f}", flush=True)
            print(f"Best reward: {current_best_reward:.4f} at epoch {best_epoch}", flush=True)
            print()
        
        if i % 10 == 0 and args.save_model and i != num_iterations:
            # Save the model every 10 iterations
            gen_model = copy.deepcopy(trainer.fine_model)
            torch.save(gen_model.cpu().state_dict(), save_path / Path(f"model_{i}.pth"))
            print(f"Model saved to {save_path}", flush=True)

    # Finish wandb run
    if args.use_wandb and not args.debug:
        wandb.finish()
    
    losses = [losses[0]] + losses
    metrics["losses"] = losses

    # Save data
    if not args.use_wandb and not args.debug:
        OmegaConf.save(config, save_path / Path("config.yaml"))
        np.savez(save_path / "results.npz", **metrics)

    # Plotting if enabled
    if args.save_plots and not args.debug:
        from utils.plotting import plot_graphs
        # Plot rewards and losses
        tmp_data = [metrics['mean'], metrics['std'], metrics['median'], metrics['losses'], metrics['num_invalid']]
        tmp_titles = ["Rewards", "Std", "Median", "Losses", "Num Invalids"]
        plot_graphs(tmp_data, tmp_titles, save_path=save_path / Path("stats.png"))
        print(f"Saved plots to {save_path}", flush=True)
    
    # Save the model if enabled
    if args.save_model and not args.debug:
        gen_model = copy.deepcopy(trainer.fine_model)
        torch.save(gen_model.cpu().state_dict(), save_path / Path("final_model.pth"))
        print(f"Model saved to {save_path}", flush=True)
    
    # Save the samples if enabled
    if args.save_samples and not args.debug:
        from dgl import save_graphs
        tmp_save_path = str(save_path / Path("samples.bin"))
        save_graphs(tmp_save_path, new_samples)
        print(f"Samples saved to {save_path}", flush=True)

    print(f"--- Final ---", flush=True)
    print(f"Final reward: {metrics['mean'][-1]:.4f}", flush=True)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- End ---", flush=True)
    print(f"End time: {end_time}", flush=True)
    print(f"Duration: {(time.time()-start_time)/60:.2f} mins", flush=True)
    print()
    print()


if __name__ == "__main__":
    main()

