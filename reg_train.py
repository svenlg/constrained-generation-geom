import argparse
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb

from regessor.dataset import GeomDataModule
from regessor.lightning_module import GNNLightningModule


def train_gnn(
    experiment: str,
    property: str,
    use_wandb: bool = False,
    seed: int = 0,
    **trainer_kwargs
) -> None:
    """Main training function."""
    
    # Set seeds
    pl.seed_everything(seed, workers=True)
    
    # Default hyperparameters
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'hidden_dim': 256,
        'depth': 6,
        'max_epochs': 100,
        'scheduler': 'cosine',
        'warmup_steps': 1000,
        **trainer_kwargs
    }

    if use_wandb:
        # Initialize wandb logger
        wandb_logger = WandbLogger(
            project=f"gnn-molecular-{property}",
            name=experiment,
            config=config
        )
    
    # Data module
    data_module = GeomDataModule(
        experiment=experiment,
        property=property,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
    )
    
    # Get feature dimensions
    node_feats = 19
    edge_feats = 5
    
    # Model
    model = GNNLightningModule(
        property=property,
        node_feats=node_feats,
        edge_feats=edge_feats,
        hidden_dim=config['hidden_dim'],
        depth=config['depth'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler=config['scheduler'],
        warmup_steps=config['warmup_steps'],
    )
    
    # Callbacks
    ckpt_dir = f"pretrained_models/{experiment}/checkpoints/"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            filename='{epoch:02d}-{val_loss:.3f}',
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=20,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        logger=wandb_logger if use_wandb else None,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN on GEOM dataset with XTB properties")
    parser.add_argument("-e", "--experiment", type=str, required=True, 
                        help="Path to data folder")
    parser.add_argument("--property", type=str, default="dipole",
                        help="Property to calculate with XTB (e.g., 'energy', 'homo', 'lumo', 'gap', 'dipole', 'dipole_zero') or 'score'")
    parser.add_argument("--seed", type=int, default=0,
                        help="Set seed")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("-d", "--depth", type=int, default=6,
                        help="Number of GNN layers")
    parser.add_argument("-me", "--max_epochs", type=int, default=100,
                        help="Maximum epochs")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="use WandB in this run.")
    args = parser.parse_args()
    
    train_gnn(**vars(args))

