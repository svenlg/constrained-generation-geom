import argparse
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description="Run ALM with optional parameter overrides")
    # Settings
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int,
                        help="Random seed.")
    parser.add_argument("--experiment", type=str,
                        help="Set experiment name.")
    parser.add_argument("--use_wandb", action='store_true',
                        help="Use wandb, default: false")
    parser.add_argument("--save_model", action='store_true',
                        help="Save the model, default: false")
    parser.add_argument("--save_samples", action='store_true',
                        help="Create animation of the samples and save the samples, default: false")
    parser.add_argument("--save_plots", action='store_true',
                        help="Save plots of rewards and constraints, default: false")
    # FlowMol arguments
    flowmol_choices = ['qm9_ctmc', 'qm9_gaussian', 'geom_ctmc', 'geom_gaussian']
    parser.add_argument('--flow_model', type=str, choices=flowmol_choices,
                        help='pretrained model to be used')
    # Reward and Constraint
    reward_choices = ['dipole', 'score', 'dipole_zero']
    parser.add_argument("--reward", type=str, choices=reward_choices,
                        help="Override reward in config")
    parser.add_argument("--constraint", type=str, choices=reward_choices,
                        help="Override reward in config")
    parser.add_argument("--bound", type=float,
                        help="Override bound in config")
    # Augmented Lagrangian Parameters
    parser.add_argument("--rho_init", type=float,
                        help="Override rho_init in config")
    parser.add_argument("--rho_max", type=float,
                        help="Override rho_max in config")
    parser.add_argument("--eta", type=float,
                        help="Override eta in config")
    parser.add_argument("--lagrangian_updates", type=int,
                        help="Override lagrangian_updates in config")
    parser.add_argument("--update_base_model", type=bool,
                        help="Override update_base_model in config, default: true")
    parser.add_argument("--baseline", action='store_true',
                        help="Do baseline, default: false")
    parser.add_argument("--base_lambda", type=float,
                        help="Override base_lambda in config")
    # Adjoint Matching Parameters
    parser.add_argument("--reward_lambda", type=float,
                        help="Override reward_lambda in config")
    parser.add_argument("--lr", type=float,
                        help="Override adjoint_matching.lr in config")
    parser.add_argument("--clip_grad_norm",  type=float,
                        help="Override adjoint_matching.clip_grad_norm in config")
    parser.add_argument("--clip_loss",  type=float,
                        help="Override adjoint_matching.clip_loss in config")
    parser.add_argument("--batch_size", type=int,
                        help="Override adjoint_matching.batch_size in config")
    parser.add_argument("--samples_per_update", type=int,
                        help="Override adjoint_matching.num_samples in config")
    parser.add_argument("--num_integration_steps", type=int,
                        help="Override adjoint_matching.num_integration_steps in config")
    parser.add_argument("--finetune_steps", type=int,
                        help="Override adjoint_matching.finetune_steps in config")
    parser.add_argument("--num_iterations", type=int,
                        help="Override number of iterations")
    # Number of Atoms per Molecule
    parser.add_argument("--n_atoms", type=int,
                        help="Number of atoms per molecule, int or null")
    parser.add_argument("--min_num_atoms", type=int,
                        help="Minimum number of atoms per molecule, int or null")
    parser.add_argument("--max_num_atoms", type=int,
                        help="Maximum number of atoms per molecule, int or null")
    # Finetuning RC models
    parser.add_argument("--reward_finetuning", action='store_true',
                        help="Whether to finetune the reward model, default: false")
    parser.add_argument("--constraint_finetuning", action='store_true',
                        help="Whether to finetune the constraint model, default: false")
    parser.add_argument("--finetune_freq", type=int,
                        help="Frequency of finetuning steps, int or null")
    parser.add_argument("--finetune_lr", type=float,
                        help="Learning rate for finetuning, float or null")
    parser.add_argument("--finetune_weight_decay", type=float,
                        help="Weight decay for finetuning, float or null")
    parser.add_argument("--finetune_clip_grad_norm", type=float,
                        help="Clip grad norm for finetuning, float or null")
    # Augmented Reward Normalization
    parser.add_argument("--augmented_reward_normalize", type=bool,
                        help="Whether to normalize the terms in the augmented reward")
    parser.add_argument("--augmented_reward_target", type=float,
                        help="Target value for normalization")
    parser.add_argument("--augmented_reward_eps", type=float,
                        help="Epsilon value for normalization")
    return parser.parse_args()


def update_config_with_args(config, args):
    # Settings
    if args.seed is not None:
        config.seed = args.seed
    if args.experiment is not None:
        config.experiment = args.experiment
    # FlowMol arguments
    if args.flow_model is not None:
        config.flow_model = args.flow_model
    # Reward and Constraint
    if args.reward is not None:
        config.reward.fn = args.reward
    if args.constraint is not None:
        config.constraint.fn = args.constraint
    if args.bound is not None:
        config.constraint.bound = args.bound
    # Augmented Lagrangian Parameters
    if args.rho_init is not None:
        config.augmented_lagrangian.rho_init = args.rho_init
    if args.rho_max is not None:
        config.augmented_lagrangian.rho_max = args.rho_max
    if args.eta is not None:
        config.augmented_lagrangian.eta = args.eta
    if args.lagrangian_updates is not None:
        config.augmented_lagrangian.lagrangian_updates = args.lagrangian_updates
    if args.update_base_model:
        config.augmented_lagrangian.update_base_model = args.update_base_model
    if args.baseline:
        config.augmented_lagrangian.baseline = True
    if args.base_lambda is not None:
        config.augmented_lagrangian.base_lambda = args.base_lambda
    # Adjoint Matching Parameters
    if args.reward_lambda is not None:
        config.reward_lambda = args.reward_lambda
    if args.lr is not None:
        config.adjoint_matching.lr = args.lr
    if args.clip_grad_norm is not None:
        config.adjoint_matching.clip_grad_norm = args.clip_grad_norm
    if args.clip_loss is not None:
        config.adjoint_matching.clip_loss = args.clip_loss
    if args.batch_size is not None:
        config.adjoint_matching.batch_size = args.batch_size
    if args.samples_per_update is not None:
        config.adjoint_matching.sampling.num_samples = args.samples_per_update
    if args.num_integration_steps is not None:
        config.adjoint_matching.sampling.num_integration_steps = args.num_integration_steps
    if args.finetune_steps is not None:
        config.adjoint_matching.finetune_steps = args.finetune_steps
    if args.num_iterations is not None:
        config.adjoint_matching.num_iterations = args.num_iterations
    # Number of atoms
    if args.n_atoms is not None:
        config.n_atoms = args.n_atoms
    if args.min_num_atoms is not None:
        config.min_num_atoms = args.min_num_atoms
    if args.max_num_atoms is not None:
        config.max_num_atoms = args.max_num_atoms
    # Finetuning RC models
    if args.reward_finetuning:
        config.reward.fine_tuning = True
    if args.constraint_finetuning:
        config.constraint.fine_tuning = True
    if args.finetune_freq is not None:
        config.rc_finetune.freq = args.finetune_freq
    if args.finetune_lr is not None:
        config.rc_finetune.lr = args.finetune_lr
    if args.finetune_weight_decay is not None:
        config.rc_finetune.weight_decay = args.finetune_weight_decay
    if args.finetune_clip_grad_norm is not None:
        config.rc_finetune.clip_grad_norm = args.finetune_clip_grad_norm
    # Augmented Reward Normalization
    if args.augmented_reward_normalize is not None:
        config.augmented_reward.normalize = args.augmented_reward_normalize
    if args.augmented_reward_target is not None:
        config.augmented_reward.target = args.augmented_reward_target
    if args.augmented_reward_eps is not None:
        config.augmented_reward.eps = args.augmented_reward_eps
    return config

