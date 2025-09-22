# train_plain.py
import os
import math
import wandb
import argparse
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from dgl.dataloading import GraphDataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regessor import GNN, EGNN, make_loaders  

from utils.utils import set_seed


def build_model(property: str, model: str, hidden_dim: int, depth: int) -> nn.Module:
    K_x = 3  # number of spatial dimensions (3D)
    K_a = 10 # number of atom features
    K_c = 6  # number of charge classes (0, +1, -1, +2)
    K_e = 5  # number of bond types (none, single, double, triple, aromatic)

    if model == "egnn":
        model = EGNN(
            property = property,
            num_atom_types = K_a,
            num_charge_classes = K_c,
            num_bond_types = K_e,
            hidden_dim = hidden_dim,
            depth = depth,
        )
    if model == "gnn":
        model = GNN(
            property = property,
            node_feats = K_a + K_c + K_x,
            edge_feats = K_e,
            hidden_dim = hidden_dim,
            depth = depth,
        )
    return model


def warmup_cosine_scheduler(optimizer: torch.optim.Optimizer,
                            total_steps: int,
                            warmup_steps: int,
                            min_lr: float = 1e-6):
    # Linear warmup, then cosine to min_lr
    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, warmup_steps)
    )
    cosine_steps = max(1, total_steps - warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=min_lr
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


@torch.no_grad()
def evaluate(model: nn.Module, loader: GraphDataLoader, device: torch.device,
             loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()
    losses = []
    all_preds = []
    all_tgts = []

    for graphs, targets in loader:
        graphs = graphs.to(device)
        targets = targets.to(device)

        preds = model(graphs).squeeze(-1)
        loss = loss_fn(preds, targets)

        losses.append(loss.item())
        all_preds.append(preds.detach().cpu())
        all_tgts.append(targets.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    tgts = torch.cat(all_tgts).numpy()

    mae = mean_absolute_error(tgts, preds)
    rmse = float(np.sqrt(mean_squared_error(tgts, preds)))
    r2 = r2_score(tgts, preds)

    return {
        "loss": float(np.mean(losses)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }


def save_checkpoint(state: Dict, ckpt_dir: str, filename: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    return path


# ---------------------------
# Training Loop
# ---------------------------
def train(
    experiment: str,
    property: str,
    model_type: str,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 256,
    depth: int = 8,
    max_epochs: int = 100,
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    seed: int = 0,
    num_workers: int = 4,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_name: str = None,
    early_stop_patience: int = 20,
):
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = make_loaders(
        experiment, property, batch_size, num_workers
    )

    # Model / Optim / Sched / Loss
    model = build_model(property, model_type, hidden_dim=hidden_dim, depth=depth).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max_epochs * max(1, len(train_loader))
    scheduler = warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    # Logging / Checkpoints
    run_id = datetime.now().strftime("%m%d_%H%M")
    ckpt_dir = f"pretrained_models/{property}/{model_type}/{run_id}"
    best_val = math.inf
    best_path = None
    epochs_no_improve = 0

    config ={
        "experiment": experiment,
        "model_type": model_type,
        "property": property,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "max_epochs": max_epochs,
        "warmup_steps": warmup_steps,
        "grad_clip": grad_clip,
        "seed": seed,
    }

    # WandB
    if use_wandb:
        if wandb_project is None:
            wandb_project = f"gnn-molecular-{property}"
        if wandb_name is None:
            wandb_name = f"{run_id}_hd{hidden_dim}_d{depth}"
        wandb.init(project=wandb_project, name=wandb_name, config=config)
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else None
        if sweep_id is not None:
            print(f"WandB sweep ID: {sweep_id}")
            tmp_name = wandb_name if wandb_name is not None else wandb.run.id
            ckpt_dir = f"pretrained_models/{property}/{model_type}/{sweep_id}/{tmp_name}/"

    # ---------------------------
    # Main loop
    # ---------------------------
    print(f"Starting training on {property} for max {max_epochs} epochs:", flush=True)
    tmp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters in model: {tmp}")
    print(f"Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = []

        for batch_idx, (graphs, targets) in enumerate(train_loader, start=1):
            graphs = graphs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)


            preds = model(graphs).squeeze(-1)
            loss = loss_fn(preds, targets)

            loss.backward()

            # Gradient clipping
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()  # step per batch so warmup works

            running.append(loss.item())

            if use_wandb and (batch_idx % 50 == 0):
                wandb.log({
                    "train/step_loss": loss.item(),
                    "epoch": epoch,
                })

        train_loss = float(np.mean(running)) if running else float("nan")

        # Validation
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        val_loss = val_metrics["loss"]

        print(
            f"[{epoch:03d}/{max_epochs}] "
            f"train/loss={train_loss:.4f} | "
            f"val/loss={val_loss:.4f} | "
            f"val/mae={val_metrics['mae']:.4f} | "
            f"val/rmse={val_metrics['rmse']:.4f} | "
            f"val/r2={val_metrics['r2']:.4f}",
            flush=True
        )

        # Logging per-epoch
        if use_wandb:
            wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/mae": val_metrics["mae"],
            "val/rmse": val_metrics["rmse"],
            "val/r2": val_metrics["r2"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Checkpointing (save best)
        if val_loss < best_val:
            print(f"- New best: with val_loss={val_loss:.4f}")
            best_val = val_loss
            epochs_no_improve = 0
            best_path = save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": best_val,
                    "config": config,
                },
                ckpt_dir,
                filename=f"best_model.pt",
            )
            
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # ---------------------------
    # Test (best checkpoint if available)
    # ---------------------------
    if best_path and os.path.isfile(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded best checkpoint: {best_path} (val_loss={state.get('val_loss','?')})")

    test_metrics = evaluate(model, test_loader, device, loss_fn)
    print(
        f"[TEST] loss={test_metrics['loss']:.4f} | mae={test_metrics['mae']:.4f} | "
        f"rmse={test_metrics['rmse']:.4f} | r2={test_metrics['r2']:.4f}"
    )
    if use_wandb:
        wandb.log({
            "test/loss": test_metrics["loss"],
            "test/mae": test_metrics["mae"],
            "test/rmse": test_metrics["rmse"],
            "test/r2": test_metrics["r2"],
        })
        wandb.finish()


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("Train GNN (plain PyTorch + DGL)")
    p.add_argument("--property", type=str, required=True,
                   help="One of: score, energy, homo, lumo, homolumo_gap, dipole, dipole_zero")
    p.add_argument("-e", "--experiment", type=str, required=True, help="Path to data folder (same as before)")
    model_type_choices = ["egnn", "gnn"]
    p.add_argument("-m", "--model_type", type=str, required=True, choices=model_type_choices, help="One of: egnn, gnn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-bs", "--batch_size", type=int, default=64)
    p.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    p.add_argument("-wd", "--weight_decay", type=float, default=1e-5)
    p.add_argument("-hd", "--hidden_dim", type=int, default=96)
    p.add_argument("-d", "--depth", type=int, default=4)
    p.add_argument("-me", "--max_epochs", type=int, default=100)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--debug", action="store_true", help="Debug mode (smaller model, fewer epochs)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        print("DEBUG MODE")
        args.max_epochs = 10
        if torch.cuda.is_available():
            args.batch_size = 64
            args.hidden_dim = 192
            args.depth = 6
        else:
            args.batch_size = 16
            args.hidden_dim = 64
            args.depth = 3
        args.use_wandb = False
    train(
        experiment=args.experiment,
        property=args.property,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        early_stop_patience=args.early_stop_patience,
    )

