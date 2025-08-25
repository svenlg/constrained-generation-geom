import flowmol
import argparse
from PAMNet.models import PAMNet_s, Config

# import hydra
# from omegaconf import DictConfig
# import wandb
# from src.sampler import get_model
# from src.true_reward import get_reward_function
# from src.constraint import get_constraint_function
# from src.utils.logging import setup_wandb

from true_reward import xtb_simulation
# @hydra.main(config_path="configs", config_name="config")
# def main(cfg: DictConfig):
#     # Setup wandb
#     setup_wandb(cfg)
    
#     # Initialize model
#     model = get_model(cfg.model)
    
#     # Initialize reward functions
#     learned_reward = get_reward_function(cfg.reward, learned=True)
#     true_reward = get_reward_function(cfg.reward, learned=False)
    
#     # Initialize constraint functions
#     learned_constraint = get_constraint_function(cfg.constraint, learned=True)
#     true_constraint = get_constraint_function(cfg.constraint, learned=False)
    
#     # Training loop would go here
    

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline Script')

    # General arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run the model on')
    
    # Flowmol arguments
    model_names = ['qm9_ctmc', 'qm9_gaussian', 'qm9_simplexflow', 'qm9_dirichlet']
    parser.add_argument('--model', default='qm9_ctmc', choices=model_names,
                        help='Path to a model checkpoint to seed the model with')
    parser.add_argument('--n_molecules', type=int, default=10, 
                        help='The number of molecules to generate.')
    parser.add_argument('--n_timesteps', type=int, default=40,
                        help="Number of timesteps for integration via Euler's method")

    # PAMNet arguments
    parser.add_argument('--dataset', type=str, default='QM9',
                        help='Dataset to be used')
    parser.add_argument('--reward_model', type=str, default='PAMNet_s', choices=['PAMNet', 'PAMNet_s'],
                        help='Reward model to be used')
    parser.add_argument('--n_layer', type=int, default=6,
                        help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, 
                        help='Size of input hidden units.')
    parser.add_argument('--target', type=int, default=7, help='Index of target for prediction') # TODO: Change it to the correct names of the targets
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='cutoff in global layer')
    
    return parser.parse_args()


def main():
    args = parse_args()
    model = flowmol.load_pretrained(args.model)
    model = model.to(args.device)
    model.eval()

    print(f"Sampling {args.n_molecules} molecules...")
    sampled_molecules = model.sample_random_sizes(n_molecules=args.n_molecules, n_timesteps=args.n_timesteps, device=args.device)
    
    config = Config(dataset="QM9", dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)
    reward_model = PAMNet_s(config).to(args.device)
    reward_model.eval()

    # targets = []
    # for data in sampled_molecules:
    #     data.pyg_mol.pos.requires_grad_()
    #     tmp = reward_model(data.pyg_mol)
    #     targets.append(tmp)
    #     tmp.backward()
    #     pos_grad = data.pyg_mol.pos.grad
    # print(len(targets))

    true_rewards = []
    for mol in sampled_molecules:
        quantity_value = xtb_simulation.compute_true_reward(mol.g, "dgl", "homolumo")
        homolumo_gap, lumo, homo = quantity_value
        print(f"HOMO-LUMO gap: {homolumo_gap:.6f} eV")
        print(f"LUMO: {lumo} eV\nHOMO: {homo} eV")

    moin = 1
    test = 2        

if __name__ == '__main__':
    main()
