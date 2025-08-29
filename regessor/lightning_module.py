from typing import Dict, List, Tuple, Any, Optional

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
        self.save_hyperparameters("learning_rate", "weight_decay", "scheduler", "warmup_steps")

        # Model
        self.model = GNN(
            node_feats=node_feats,
            edge_feats=edge_feats,
            hidden_dim=hidden_dim,
            depth=depth
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
        if graphs is None:
            # Safeguard against empty batches
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        preds = self(graphs).squeeze(-1)
        loss = self.loss_fn(preds, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ---------------------------
    # Validation
    # ---------------------------
    def on_validation_start(self) -> None:
        # reset buffers each epoch
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_losses.clear()

    def validation_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> None:
        graphs, targets = batch
        if graphs is None:
            return

        preds = self(graphs).squeeze(-1)
        loss = self.loss_fn(preds, targets)

        # Collect for epoch-end (move to CPU to avoid GPU memory growth)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(targets.detach().cpu())
        self._val_losses.append(loss.detach().cpu())

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
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae, prog_bar=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("val_r2", r2, prog_bar=False, sync_dist=True)

    # ---------------------------
    # Test
    # ---------------------------
    def on_test_start(self) -> None:
        self._test_preds.clear()
        self._test_targets.clear()
        self._test_losses.clear()

    def test_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> None:
        graphs, targets = batch
        if graphs is None:
            return

        preds = self(graphs).squeeze(-1)
        loss = self.loss_fn(preds, targets)

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(targets.detach().cpu())
        self._test_losses.append(loss.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if len(self._test_preds) == 0:
            return

        preds = torch.cat(self._test_preds).numpy()
        targets = torch.cat(self._test_targets).numpy()
        test_loss = torch.stack(self._test_losses).mean().item()

        mae = mean_absolute_error(targets, preds)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        r2 = r2_score(targets, preds)

        self.log("test_loss", test_loss, sync_dist=True)
        self.log("test_mae", mae, sync_dist=True)
        self.log("test_rmse", rmse, sync_dist=True)
        self.log("test_r2", r2, sync_dist=True)

    # ---------------------------
    # Optimizers / Schedulers
    # ---------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler == "cosine":
            # Step the scheduler every epoch (default) with T_max in epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # not strictly needed for cosine, but fine
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
