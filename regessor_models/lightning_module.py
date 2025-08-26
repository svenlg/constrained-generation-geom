from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regessor_models.gnn import GNN


class GNNLightningModule(pl.LightningModule):
    """PyTorch Lightning module for GNN training."""
    
    def __init__(
        self,
        node_feats: int,
        edge_feats: int,
        hidden_dim: int = 256,
        depth: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler: str = "cosine",
        warmup_steps: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = GNN(
            node_feats=node_feats,
            edge_feats=edge_feats,
            hidden_dim=hidden_dim,
            depth=depth
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
    
    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return self.model(graph)
    
    def training_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> torch.Tensor:
        graphs, targets = batch
        
        # Skip empty batches
        if graphs is None:
            return None
        
        # Forward pass
        predictions = self(graphs)
        
        # Calculate loss
        loss = self.loss_fn(predictions.squeeze(), targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        graphs, targets = batch
        
        # Skip empty batches
        if graphs is None:
            return None
        
        # Forward pass
        predictions = self(graphs)
        
        # Calculate loss
        loss = self.loss_fn(predictions.squeeze(), targets)
        
        # Store predictions and targets for epoch-end metrics
        return {
            'val_loss': loss,
            'predictions': predictions.squeeze(),
            'targets': targets
        }
    
    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Filter out None values
        outputs = [x for x in outputs if x is not None]
        
        if len(outputs) == 0:
            return
        
        # Aggregate predictions and targets
        all_predictions = torch.cat([x['predictions'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        # Convert to numpy for sklearn metrics
        preds_np = all_predictions.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Calculate metrics
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mae = mean_absolute_error(targets_np, preds_np)
        rmse = np.sqrt(mean_squared_error(targets_np, preds_np))
        r2 = r2_score(targets_np, preds_np)
        
        # Log metrics
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('val_r2', r2, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Same as validation_epoch_end but with 'test_' prefix
        outputs = [x for x in outputs if x is not None]
        
        if len(outputs) == 0:
            return
        
        all_predictions = torch.cat([x['predictions'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        preds_np = all_predictions.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        test_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mae = mean_absolute_error(targets_np, preds_np)
        rmse = np.sqrt(mean_squared_error(targets_np, preds_np))
        r2 = r2_score(targets_np, preds_np)
        
        self.log('test_loss', test_loss, on_epoch=True)
        self.log('test_mae', mae, on_epoch=True)
        self.log('test_rmse', rmse, on_epoch=True)
        self.log('test_r2', r2, on_epoch=True)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        elif self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        else:
            return optimizer

