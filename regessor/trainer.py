# fine_tune_trainer.py
from pyexpat import model
from typing import List, Optional, Tuple, Union, Dict, Any
import itertools
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader

from omegaconf import OmegaConf


class SampleDataset(torch.utils.data.Dataset):
    """
    Minimal dataset for fine-tuning from in-memory samples.
    Accepts a list of DGLGraphs and corresponding targets (Tensor/ndarray/list).
    """
    def __init__(self, graphs: List[DGLGraph], targets: Union[torch.Tensor, np.ndarray, List[float]]):
        assert len(graphs) > 0, "Provide at least one graph."
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        elif isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float32)
        if targets.ndim == 1:
            targets = targets.float()
        self.graphs = graphs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[DGLGraph, torch.Tensor]:
        tgt = self.targets[idx]
        # Ensure shape [1] so later .squeeze(-1) is safe
        if tgt.ndim == 0:
            tgt = tgt.unsqueeze(0)
        return self.graphs[idx], tgt


class GNNFineTuner:
    """
    Simple fine-tuning trainer for GNNs (DGL + PyTorch).

    - Pass an existing model (already constructed).
    - Call .fit(new_graphs, new_targets, steps=...) to run N gradient steps.
    - Supports gradient clipping and AMP.
    - Provides evaluate() for quick diagnostics.
    """
    def __init__(
        self,
        property: str,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-6,
        grad_clip: Optional[float] = 1.0,
        amp: bool = True,
    ):
        self.property = property
        self.model = model
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()
        self.grad_clip = grad_clip
        self.amp = amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.amp)

    def _make_loader(
        self,
        data: List[DGLGraph],
        targets: Union[torch.Tensor, np.ndarray, List[float]],
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
    ) -> GraphDataLoader:
        
        assert len(data) == len(targets), "Graphs and targets must have the same length."
        ds = SampleDataset(data, targets)
        return GraphDataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @torch.no_grad()
    def evaluate(self, loader: GraphDataLoader) -> Dict[str, float]:
        """Evaluate loss/MAE/RMSE/R2 on a loader."""
        self.model.eval()
        losses = []
        preds_all = []
        tgts_all = []

        for graphs, targets in loader:
            graphs = graphs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(graphs)
            loss = self.loss_fn(outputs, targets)
            losses.append(loss.item())
            preds_all.append(outputs.detach().cpu())
            tgts_all.append(targets.detach().cpu())

        if not losses:
            return {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

        preds = torch.cat(preds_all).numpy()
        tgts = torch.cat(tgts_all).numpy()

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = float(mean_absolute_error(tgts, preds))
        rmse = float(np.sqrt(mean_squared_error(tgts, preds)))
        r2 = float(r2_score(tgts, preds))

        return {"loss": float(np.mean(losses)), "mae": mae, "rmse": rmse, "r2": r2}

    def fit(
        self,
        data: List[DGLGraph],
        targets: Union[torch.Tensor, np.ndarray, List[float]],
        steps: int = 200,
        batch_size: int = 32,
        num_workers: int = 0,
        log_interval: int = 20,
        eval_split: float = 0.2,
        set_to_none: bool = True,
    ) -> Dict[str, Any]:
        """
        Fine-tune the model for a fixed number of gradient steps.
        Returns a small history with last train loss and (optional) eval metrics.
        """
        split = int(len(data) * (1 - eval_split))
        train_data, train_targets = data[:split], targets[:split]
        eval_data, eval_targets = data[split:], targets[split:]

        loader = self._make_loader(
            data=train_data, targets=train_targets, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        eval_loader = self._make_loader(
            data=eval_data, targets=eval_targets, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        it = itertools.cycle(loader)  # cycle through small datasets for step-based updates

        self.model.train()
        step_loss = []

        for step in range(1, steps + 1):
            graphs, tgts = next(it)
            graphs = graphs.to(self.device)
            tgts = tgts.to(self.device)

            self.optimizer.zero_grad(set_to_none=set_to_none)

            with autocast(enabled=self.amp):
                outputs = self.model(graphs)
                loss = self.loss_fn(outputs, tgts)

            # backward + step with AMP + grad clipping
            self.scaler.scale(loss).backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                # unscale before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if (step % log_interval) == 0 or step == steps:
                step_loss.append(float(loss.item()))

        step_loss = np.mean(step_loss)
        history = {f"{self.property}/step_loss": step_loss}
        # Evaluate after fine-tune
        history[f"{self.property}/eval"] = self.evaluate(eval_loader)

        self.model.eval()
        return history

    def save(self, path: str):
        state = {"model": self.model.state_dict()}
        torch.save(state, path)

    def load(self, path: str, strict: bool = True):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=strict)


def setup_fine_tuner(property: str, model: nn.Module, config: OmegaConf = None) -> GNNFineTuner:
    fine_tuner = GNNFineTuner(
        property=property,
        model=model,
        learning_rate=1e-5 if config is None else config.get("lr", 1e-5),
        weight_decay=1e-6 if config is None else config.get("weight_decay", 1e-6),
        grad_clip=1.0 if config is None else config.get("grad_clip", 1.0),
        amp=True,
    )
    return fine_tuner


def finetune(
    property: str,
    finetuner: GNNFineTuner,
    data: List[DGLGraph],
    targets: Union[torch.Tensor, np.ndarray, List[float]],
    config: OmegaConf = None,
) -> Dict[str, Any]:
    """
    Fine-tuning helper.
    - Runs fine-tuning on the provided new graphs/targets.
    - Returns the training history.
    """
    # print(f"Fine-tuning for property '{property}' with {len(data)} samples...")
    history = finetuner.fit(
        data=data,
        targets=targets,
        steps=200 if config is None else config.get("steps", 200),
        batch_size=32 if config is None else config.get("batch_size", 32),
        num_workers=0 if config is None else config.get("num_workers", 0),
        log_interval=20 if config is None else config.get("log_interval", 20),
    )
    eval_metrics = history.get(f"{property}/eval", {})
    # print(f"\tEval: loss: {eval_metrics.get('loss', 0):.4f} | mae: {eval_metrics.get('mae', 0):.4f} | rmse: {eval_metrics.get('rmse', 0):.4f} | r2: {eval_metrics.get('r2', 0):.4f}", flush=True)
    return history
