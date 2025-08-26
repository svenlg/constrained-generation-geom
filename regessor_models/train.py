import argparse
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb

from dataset import GeomDataModule
from lightning_module import GNNLightningModule


def train_gnn(
    data_path: str,
    property_name: str,
    project_name: str = "gnn-molecular-property",
    experiment_name: Optional[str] = None,
    **trainer_kwargs
) -> None:
    """Main training function."""
    
    # Default hyperparameters
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'hidden_dim': 256,
        'depth': 6,
        'max_epochs': 100,
        'min_atoms': 20,
        'max_atoms': 75,
        'scheduler': 'cosine',
        'warmup_steps': 1000,
        **trainer_kwargs
    }
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name or f"gnn-{property_name}",
        config=config
    )
    
    # Data module
    data_module = GeomDataModule(
        data_path=data_path,
        property_name=property_name,
        batch_size=config['batch_size'],
        min_atoms=config['min_atoms'],
        max_atoms=config['max_atoms']
    )
    
    # Get feature dimensions
    # Node features: 100 (one-hot atomic numbers) + 3 (coordinates) + 1 (atomic number)
    node_feats = 104
    # Edge features: 1 (distance) + 3 (relative coordinates)  
    edge_feats = 4
    
    # Model
    model = GNNLightningModule(
        node_feats=node_feats,
        edge_feats=edge_feats,
        hidden_dim=config['hidden_dim'],
        depth=config['depth'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler=config['scheduler'],
        warmup_steps=config['warmup_steps']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
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
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=50
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN on GEOM dataset with XTB properties")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data folder")
    parser.add_argument("--property_name", type=str, required=True, 
                        help="Property to calculate with XTB (e.g., 'energy', 'homo', 'lumo', 'gap', 'dipole')")
    parser.add_argument("--project_name", type=str, default="gnn-molecular-property", help="WandB project name")
    parser.add_argument("--experiment_name", type=str, help="WandB experiment name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=6, help="Number of GNN layers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--min_atoms", type=int, default=20, help="Minimum number of atoms")
    parser.add_argument("--max_atoms", type=int, default=75, help="Maximum number of atoms")
    
    args = parser.parse_args()
    
    train_gnn(**vars(args))

