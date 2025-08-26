import os
import os.path as osp
import copy
from omegaconf import OmegaConf
import wandb
import time
import argparse
from utils.utils import seed_everything
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR


def args_parser():
    parser = argparse.ArgumentParser(description='PAMNet Training')
    parser.add_argument("--use_wandb", action='store_true', help="Use wandb, default: false")
    parser.add_argument('--experiment', type=str, default='PAMNet_QM9', help='Name of the experiment for wandb')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='QM9', help='Dataset to be used')
    parser.add_argument('--model', type=str, default='PAMNet_s', choices=['PAMNet', 'PAMNet_s'], help='Model to be used')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--target', type=int, default=19, help='Index of target for prediction')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='Cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='Cutoff in global layer')
    parser.add_argument('--debug', action='store_true', help='Debug mode, default: false')
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(model, loader, ema, device):
    mae = 0
    ema.assign(model)
    for data in loader:
        data = data.to(device)
        output = model(data)
        mae += (output - data.y).abs().sum().item()
    ema.resume(model)
    return mae / len(loader.dataset)

def main():
    args = args_parser()

    # Setup - Seed and device
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup - WandB
    use_wandb = args.use_wandb and not args.debug
    if use_wandb:
        wandb.init(
            project=f"PAMNet_{args.dataset}_sascore",
            name=args.experiment, 
            config=vars(args),
        )

    class MyTransform(object):
        def __call__(self, data):
            target = args.target
            if target in [7, 8, 9, 10]:
                target = target + 5
            data.y = data.y[:, target]
            return data

    abs_path = os.getcwd()
    path = osp.join(abs_path, 'data', args.dataset)
    dataset = QM9(path, transform=MyTransform()).shuffle()

    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!", flush=True)

    config = Config(
        dataset=args.dataset, 
        dim=args.dim, 
        n_layer=args.n_layer, 
        cutoff_l=args.cutoff_l, 
        cutoff_g=args.cutoff_g,
    )

    # Model initialization
    model = PAMNet(config).to(device) if args.model == 'PAMNet' else PAMNet_s(config).to(device)
    print(f"Number of model parameters: {count_parameters(model)}", flush=True)

    # EMA
    ema = EMA(model, decay=0.999)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)

    # Learning rate schedulers
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main_scheduler = ExponentialLR(optimizer, gamma=0.9961697)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5])

    if use_wandb:
        wandb.watch(model, log="all", log_freq=500)

    print("Start training!", flush=True)
    day = time.strftime("%m-%d-%H", time.localtime())
    if not args.debug:
        save_folder = osp.join(abs_path, "pretrained_models", args.model, args.dataset, args.experiment, day)
        os.makedirs(save_folder, exist_ok=True)

    best_val_loss = None
    log_step_interval = len(train_loader) // 9 # 9 log steps per epoch
    global_step = 0
    for epoch in range(args.epochs):
        loss_all = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = F.l1_loss(output, data.y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
            optimizer.step()
            ema(model)

            batch_loss = loss.item() * data.num_graphs
            loss_all += batch_loss
            global_step += 1

            if use_wandb and global_step % log_step_interval == 0:
                wandb.log({
                    "step_train_loss": batch_loss / data.num_graphs,
                    "lr": optimizer.param_groups[0]["lr"],
                })

        train_mae = loss_all / len(train_loader.dataset)
        val_mae = test(model, val_loader, ema, device)

        scheduler.step()

        if (best_val_loss is None or val_mae <= best_val_loss) and not args.debug:
            test_mae = test(model, test_loader, ema, device)
            best_val_loss = val_mae
            save_model = copy.deepcopy(model)
            save_model = save_model.to("cpu")
            torch.save(save_model.state_dict(), osp.join(save_folder, "best_model.pth"))
            # Save config
            config_path = osp.join(save_folder, "config.json")
            config = {
                "experiment": args.experiment,
                "seed": args.seed,
                "epoch": epoch + 1,
                "dataset": args.dataset,
                "model": args.model,
                "dim": args.dim,
                "n_layer": args.n_layer,
                "cutoff_l": args.cutoff_l,
                "cutoff_g": args.cutoff_g,
                "epochs": args.epochs,
                "lr": args.lr,
                "wd": args.wd,
                "batch_size": args.batch_size,
                "target": args.target
            }
            OmegaConf.save(config, config_path)


        print(f"Epoch: {epoch+1:03d}, Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}, Test MAE: {test_mae}", flush=True)

        if use_wandb:
            wandb.log({
                "epoch_train_mae": train_mae,
                "epoch_val_mae": val_mae,
                "epoch_test_mae": test_mae,
                "epoch_lr": optimizer.param_groups[0]["lr"]
            })


    print(f"Best Validation MAE: {best_val_loss:.6f}", flush=True)
    print(f"Testing MAE: {test_mae:.6f}", flush=True)

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()