import sys
import pickle
import gzip
import time
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime

from posebusters import PoseBusters

from reg_data_generation.posebuster_scorer import posebusters_score
from reg_data_generation.xtb_calc import compute_xtb

from utils.utils import seed_everything
import flowmol

import logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

REMOVE_NODE_KEYS = ['x_0', 'a_0', 'c_0', 'x_1_pred', 'a_1_pred', 'c_1_pred', 'x_1', 'a_1', 'c_1']
REMOVE_EDGE_KEYS = ['e_0', 'e_1_pred', 'e_1']

# Dump molecule to pickle
def dump_pickle(path: str, mol):
    with gzip.open(path, "wb") as f:
        pickle.dump(mol, f, protocol=pickle.HIGHEST_PROTOCOL)

# Zero-pad a number
def zero_pad(num: int, width: int=6) -> str:
    return f"{num:0{width}d}"

# Load - Flow Model
def setup_gen_model(flow_model: str, device: torch.device):
    gen_model = flowmol.load_pretrained(flow_model)
    gen_model.to(device)
    gen_model.eval()
    return gen_model

# Sampling
def sampling_and_processing(
        model: flowmol.FlowMol, 
        n_samples: int,
        n_timesteps: int = 100,
        min_mol_size: int = None,
        max_mol_size: int = None,
        device: torch.device = 'cpu',
    ):
    new_molecules = model.sample_random_sizes(
        n_molecules = n_samples, 
        n_timesteps = n_timesteps,
        min_mol_size = min_mol_size,
        max_mol_size = max_mol_size,
        device = device,
    )
    rd_mols, clean_mols = [], []
    for mol in new_molecules:
        rd_mols.append(mol.rdkit_mol)
        for tmp in REMOVE_NODE_KEYS:
            mol.g.ndata.pop(tmp, None)
        for tmp in REMOVE_EDGE_KEYS:
            mol.g.edata.pop(tmp, None)
        mol.g = mol.g.to('cpu')
        clean_mols.append(mol)

    return rd_mols, clean_mols


def main(args):
    # Setup - Seed and device and root directory
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    experiment = f"{datetime.now().strftime('%m%d_%H%M')}_{args.experiment}_{args.n_samples}"
    root_dir = Path(args.root) / Path(experiment)
    root_dir.mkdir(parents=True, exist_ok=True)
    mol_dir = root_dir / Path("molecules")
    mol_dir.mkdir(parents=True, exist_ok=True)

    # Setup - Gen Model
    gen_model = setup_gen_model(args.flow_model, device=device)

    epochs = args.n_samples // args.batch_size

    data_rows = []

    with tqdm(range(epochs), desc="Sampling Progress", dynamic_ncols=True, file=sys.stdout) as pbar:
        for i in pbar:
            print("", flush=True)
            # Generate Samples
            tmp_time = time.time()
            rd_mols, clean_mols = sampling_and_processing(
                gen_model,
                args.batch_size,
                args.integration_steps,
                args.min_atoms,
                args.max_atoms,
                device=device
            )
            print(f"Sampling time: {time.time() - tmp_time:.2f} seconds", flush=True)

            ######
            # Get PoseBusters Feedback
            tmp_time = time.time()
            buster = PoseBusters(config="mol")
            df_scores = buster.bust(rd_mols)
            df_scores = df_scores.reset_index(drop=True)

            # Make Scores
            scores = posebusters_score(df_scores)
            df_scores['score'] = scores.to_numpy()
            print(f"PoseBusters time: {time.time() - tmp_time:.2f} seconds", flush=True)

            ######
            # XTB-Calulations
            tmp_time = time.time()
            properties = []
            for tmp_mol in rd_mols:
                rtn_dict = compute_xtb(tmp_mol, "rdkit")
                properties.append(rtn_dict)
            print(f"XTB time: {time.time() - tmp_time:.2f} seconds", flush=True)

            df_props = pd.DataFrame.from_records(properties)
            df = pd.concat([df_scores, df_props], axis=1)

            records = df.to_dict(orient="records")
            data_rows.extend(records)

            for k, mol in enumerate(clean_mols):
                mum = i * args.batch_size + k
                path = mol_dir / Path(f"mol_{zero_pad(mum)}.pkl.gz")
                dump_pickle(path, mol)

    df = pd.DataFrame.from_records(data_rows)
    df["id_str"] = df.index.map(lambda x: f"mol_{zero_pad(x)}")
    df.to_csv(root_dir / Path("results.csv"), index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ALM with optional parameter overrides")
    # Settings
    parser.add_argument("--root", type=str, default="data",
                        help="Path to config file")
    parser.add_argument("--experiment", type=str, default="gen_data",
                        help="Name of the experiment")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for initialization')
    # FlowMol arguments
    flowmol_choices = ['geom_ctmc', 'geom_gaussian']
    parser.add_argument('--flow_model', type=str, choices=flowmol_choices, default='geom_gaussian',
                        help='pretrained model to be used')
    parser.add_argument('--integration_steps', type=int, default=100,
                        help='Number of integration steps')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for sampling')
    # Data Generation
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--min_atoms", type=int, default=30,
                        help="Minimum number of atoms in the generated molecules")
    parser.add_argument("--max_atoms", type=int, default=75,
                        help="Maximum number of atoms in the generated molecules")
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    if args.debug:
        args.n_samples = 10
        args.batch_size = 5
    # start time
    start_time = time.time()
    main(args)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds", flush=True)

