from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regessor.gnn import GNN


class GNNLightningModule(pl.LightningModule):
    """PyTorch Lightning module for GNN training (Lightning v2+)."""

    def __init__(
        self,
        property: str,
        node_feats: int = 19,
        edge_feats: int = 5,
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

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps

        # Model
        self.property = property
        self.model = GNN(
            property=self.property,
            node_feats=node_feats,
            edge_feats=edge_feats,
            hidden_dim=hidden_dim,
            depth=depth,
            **kwargs
        )

        # Loss
        self.loss_fn = nn.MSELoss()

        # Buffers for epoch-end metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_losses: List[torch.Tensor] = []

        self._test_preds: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._test_losses: List[torch.Tensor] = []
    
    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return self.model(graph)

    # ---------------------------
    # Training
    # ---------------------------
    def training_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> torch.Tensor:
        graphs, targets = batch

        preds = self(graphs).squeeze(-1)
        loss = self.loss_fn(preds, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ---------------------------
    # Validation
    # ---------------------------
    def on_validation_start(self) -> None:
        # reset buffers each epoch
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_losses.clear()

    def validation_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> torch.Tensor:
        graphs, targets = batch

        preds = self(graphs).squeeze(-1)
        loss = self.loss_fn(preds, targets)

        # Collect for epoch-end (move to CPU to avoid GPU memory growth)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(targets.detach().cpu())
        self._val_losses.append(loss.detach().cpu())
        
        # CRITICAL: Return the loss tensor for proper step-level logging
        return loss

    def on_validation_epoch_end(self) -> None:
        if len(self._val_preds) == 0:
            return

        preds = torch.cat(self._val_preds).numpy()
        targets = torch.cat(self._val_targets).numpy()
        val_loss = torch.stack(self._val_losses).mean().item()

        mae = mean_absolute_error(targets, preds)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        r2 = r2_score(targets, preds)

        # Log epoch metrics (keep key 'val_loss' for checkpoint monitor)
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/mae", mae, prog_bar=True, sync_dist=True)
        self.log("val/rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("val/r2", r2, prog_bar=False, sync_dist=True)

    # ---------------------------
    # Test
    # ---------------------------
    def on_test_start(self) -> None:
        self._test_preds.clear()
        self._test_targets.clear()
        self._test_losses.clear()

    def test_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> torch.Tensor:
        graphs, targets = batch

        preds = self(graphs).squeeze(-1)
        loss = self.loss_fn(preds, targets)

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(targets.detach().cpu())
        self._test_losses.append(loss.detach().cpu())
        
        return loss

    def on_test_epoch_end(self) -> None:
        if len(self._test_preds) == 0:
            return

        preds = torch.cat(self._test_preds).numpy()
        targets = torch.cat(self._test_targets).numpy()
        test_loss = torch.stack(self._test_losses).mean().item()

        mae = mean_absolute_error(targets, preds)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        r2 = r2_score(targets, preds)

        self.log("test/loss", test_loss, sync_dist=True)
        self.log("test/mae", mae, sync_dist=True)
        self.log("test/rmse", rmse, sync_dist=True)
        self.log("test/r2", r2, sync_dist=True)
    
    # ---------------------------
    # Optimizers / Schedulers
    # ---------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler == "cosine":
            # Cosine annealing with warmup
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            
            # Calculate total training steps
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback calculation if estimated_stepping_batches not available
                steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) if hasattr(self.trainer, 'datamodule') else 250
                total_steps = self.trainer.max_epochs * steps_per_epoch
            
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.hparams.warmup_steps
            )
            
            # Main cosine scheduler (total steps minus warmup steps)
            cosine_steps = max(1, total_steps - self.hparams.warmup_steps)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-6
            )
            
            # Combine schedulers
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.hparams.warmup_steps]
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Step every batch for warmup to work
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def on_train_epoch_end(self) -> None:
        """Log learning rate at the end of each epoch."""
        if hasattr(self, 'lr_schedulers') and self.lr_schedulers():
            scheduler = self.lr_schedulers()
            if hasattr(scheduler, 'get_last_lr'):
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_epoch=True)
