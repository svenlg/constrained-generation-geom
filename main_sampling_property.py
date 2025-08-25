# pylint: disable=missing-module-docstring
# pylint: disable=no-name-in-module
import time

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from utils.utils import seed_everything

import flowmol

from environment.property_calculation import compute_property_stats #, compute_property_grad

# python main_sampling_property.py --property dipole --number_samples 10 --batchsize 10
# Energy - 10000
# mean_of_means: -4.225072824954987
# std_of_means_of_means: 0.1521501666044272
# median_of_means: -4.221149206161499
# mean_of_medians: -4.176584553718567
# std_of_medians_of_means: 0.20006210613906478
# median_of_medians: -4.182209014892578
# mean_of_all_finite: -4.225072860717773
# std_of_all_finite: 2.3384053707122803
# mean_of_num_invalids: 0.0
# std_of_num_invalids: 0.0
# sum_of_num_invalids: 0

# Dipole - 10000
# mean_of_means: 1.2967738538980484
# std_of_means_of_means: 0.053677259954304175
# median_of_means: 1.3017678260803223
# mean_of_medians: 1.15841403901577
# std_of_medians_of_means: 0.05125829615654835
# median_of_medians: 1.1526404023170471
# mean_of_all_finite: 1.2967737913131714
# std_of_all_finite: 0.7352674603462219
# mean_of_num_invalids: 0.0
# std_of_num_invalids: 0.0
# sum_of_num_invalids: 0

# Foreces - 10000
# mean_of_means: 16.296460
# std_of_means_of_means: 0.136924
# median_of_means: 16.307869
# mean_of_medians: 16.237001
# std_of_medians_of_means: 0.181127
# median_of_medians: 16.250904
# mean_of_all_finite: 16.296551
# std_of_all_finite: 2.044292
# mean_of_num_invalids: 0.250000
# std_of_num_invalids: 0.536190
# sum_of_num_invalids: 10.000000


def sampling(n_molecules: int, model: flowmol.FlowMol, n_atoms: int ,device: torch.device):
    with torch.no_grad():
        if n_atoms is not None:
            # Sample molecules with a fixed number of atoms
            atoms = torch.tensor([n_atoms] * n_molecules, device=device)
            new_molecules, _ = model.sample(
                n_atoms=atoms,
                device=device,
                keep_intermediate_graphs=True,
            )
        else:
            new_molecules, _ = model.sample_random_sizes(
                n_molecules=n_molecules, 
                n_timesteps=100 + 1, 
                device=device,
                keep_intermediate_graphs=True,
            )
    return new_molecules

def setup_gen_model(flow_model: str, device: torch.device):
    # Load - Flow Model
    gen_model = flowmol.load_pretrained(flow_model)
    gen_model.to(device)
    return gen_model

def analyze_property_dicts(property_dicts):
    means = []
    stds = []
    medians = []
    num_invalids = []
    all_finite_list = []

    for d in property_dicts:
        means.append(d["mean"])
        stds.append(d["std"])
        medians.append(d["median"])
        num_invalids.append(d["num_invalid"])
        all_finite_list.append(d["all_finite"])

    # Create DataFrame
    stats_df = pd.DataFrame({
        "mean": means,
        "std": stds,
        "median": medians,
    })
    flat_array = np.concatenate([np.array(l) for l in all_finite_list])

    # Compute global stats
    overall = {
        "mean_of_means": stats_df["mean"].mean(),
        "std_of_means_of_means": stats_df["mean"].std(),
        "median_of_means": stats_df["mean"].median(),
        "mean_of_medians":  stats_df["median"].mean(),
        "std_of_medians_of_means": stats_df["median"].std(),
        "median_of_medians": stats_df["median"].median(),
        "mean_of_all_finite": flat_array.mean() if flat_array.size > 0 else None,
        "std_of_all_finite": flat_array.std() if flat_array.size > 0 else None,
        "mean_of_num_invalids": np.mean(num_invalids),
        "std_of_num_invalids": np.std(num_invalids),
        "sum_of_num_invalids": np.sum(num_invalids),
    }

    return stats_df, flat_array, overall

def parse_args():
    parser = argparse.ArgumentParser(description="Run ALM with optional parameter overrides")
    # Settings
    parser.add_argument("--output_path", type=str,
                        help="Path to config file")
    parser.add_argument("--experiment", type=str, default="sampling",
                        help="Experiment Name")
    parser.add_argument("--save_stats", action='store_true',
                        help="Save the stats, default: false")
    parser.add_argument("--seed", type=int, default=42,
                        help="Set Seed")
    parser.add_argument("--debug", action='store_true',
                        help="Debug mode, default: false")
    # FlowMol arguments
    flowmol_choices = ['qm9_ctmc', 'geom_ctmc']
    parser.add_argument('--flow_model', type=str, default='qm9_ctmc',
                        choices=flowmol_choices,
                        help='pretrained model to be used')
    parser.add_argument('--n_atoms', type=int, 
                        help='Number of atoms in the molecules, default: None')
    # Property arguments
    property_choices = ['energy', 'dipole', 'forces']
    parser.add_argument("--property", type=str, default="energy",
                        choices=property_choices,
                        help="Property to compute")
    # Sampling arguments
    parser.add_argument("--number_samples", type=int, default=50,
                        help="Number of samples to generate")
    parser.add_argument("--batchsize", type=int, default=10,
                        help="Batch size for sampling")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Setup - Seed and device and root directory
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # General Parameters
    flowmol_model = args.flow_model

    print(f"--- Start ---", flush=True)
    
    # Prints
    print(f"Sampling form {flowmol_model} in experiment {args.experiment}", flush=True)
    print(f"Property: {args.property}", flush=True)
    if args.n_atoms is not None:
        print(f"Sampling molecules with {args.n_atoms} atoms", flush=True)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_time}", flush=True)
    print(f"Device: {device}", flush=True)
    start_time = time.time()

    # Setup - Gen Model
    gen_model = setup_gen_model(args.flow_model, device=device)
    
    def reward_fn(molecules):
        return compute_property_stats(
            molecules = molecules,
            property = args.property,
            device=device,
        )
    
    n_atoms = None
    if args.n_atoms is not None:
        n_atoms = args.n_atoms

    # Initialize list
    property_dicts = []
    
    # Generate Samples
    epochs = args.number_samples // args.batchsize
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}", flush=True)
        new_molecules = sampling(
            n_molecules = args.batchsize,
            model = gen_model,
            n_atoms = n_atoms,
            device = device,
        )

        # Compute appropriate reward for evaluation
        property_dict = reward_fn(new_molecules)
        property_dicts.append(property_dict)

        # delete new_molecules to free memory
        del new_molecules
        
        print(f"{args.property}: {property_dict['mean']:.4f} ({property_dict['std']:.4f})", flush=True)
        print(f"Median: {property_dict['median']:.4f} | Max: {property_dict['max']:.4f} | Min: {property_dict['min']:.4f}", flush=True)
        print(f"at {property_dict['num_invalid']} invalids", flush=True)
        print()

    stats_df, flat_array, overall = analyze_property_dicts(property_dicts)
    print("Overall stats:", flush=True)
    for key, value in overall.items():
        print(f"{key}: {value:.6f}", flush=True)

    if args.save_stats and not args.debug:
        dataset = args.flow_model.split('_')[0]
        save_path = Path(args.output_path) / Path(args.experiment) / Path(dataset) / Path(args.property)
        save_path.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(save_path / Path("stats.csv"), index=False)
        np.save(save_path / Path("all_finite.npy"), flat_array)
        print(f"Stats saved to {save_path}", flush=True)
    
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- End ---", flush=True)
    print(f"End time: {end_time}", flush=True)
    print(f"Duration: {(time.time()-start_time)/60:.2f} mins", flush=True)
    print()
    print()

if __name__ == "__main__":
    main()


## OLD - C H change ###

# QM9 - Energy - 10000
# mean_of_means: 15.545686340332031
# std_of_means_of_means: 0.13207440249244431
# median_of_means: 15.520474433898926
# mean_of_medians: 15.920872068405151
# std_of_medians_of_means: 0.17310195945551918
# median_of_medians: 15.89293384552002
# mean_of_all_finite: 15.545721054077148
# std_of_all_finite: 3.4954936504364014
# mean_of_num_invalids: 6.45
# std_of_num_invalids: 2.7290108097990378
# sum_of_num_invalids: 129


# QM9 - Dipole - 10000
# mean_of_means: 3.3665730237960814
# std_of_means_of_means: 0.13841214448269556
# median_of_means: 3.3252893686294556
# mean_of_medians: 2.3399870872497557
# std_of_medians_of_means: 0.05745568197354845
# median_of_medians: 2.349852681159973
# mean_of_all_finite: 3.366593837738037
# std_of_all_finite: 3.405334949493408
# mean_of_num_invalids: 6.55
# std_of_num_invalids: 2.8543825952384165
# sum_of_num_invalids: 131