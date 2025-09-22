import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import dgl
from omegaconf import OmegaConf

from regessor import GNN, EGNN


class RCModel(nn.Module):
    def __init__(self, property: str, config: OmegaConf, model_config: dict) -> None:
        super().__init__()

        # General settings
        self.property = property
        
        # Model
        if self.config.model_type == "GNN":
            self.gnn = GNN(**model_config)
        elif self.config.model_type == "EGNN":
            self.gnn = EGNN(**model_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_config.model_type}")
        
        # Filter function for data preprocessing
        self.filter_config = config.filter_config
        self.filter_function = None
        if self.filter_config.function == "threshold":
            self.filter_function = lambda x: torch.where(x > self.filter_config.threshold, x, torch.tensor(0.0))
        elif self.filter_config.function == "gaussian":
            self.filter_function = lambda x: torch.exp(-0.5 * ((x - self.filter_config.mu) / self.filter_config.sigma) ** 2)
        elif self.filter_config.function == "max_parabel":
            self.filter_function = lambda x: -torch.square(x-self.filter_config.mu)
        elif self.filter_config.function == "min_parabel":
            self.filter_function = lambda x: torch.square(x-self.filter_config.mu)        
        elif self.filter_config.function == "linear":
            self.filter_function = lambda x: x
        else:
            raise ValueError(f"Unknown filter function: {self.filter_config.function}")

        if property == "score" and self.filter_config.linear_output:
            self.gnn.output = nn.Identity()
        
    def forward(self, g: dgl.DGLGraph, return_logits: bool = False) -> Tensor:
        self.gnn.train()
        gnn_out = self.gnn(g)
        if self.filter_function is not None:
            gnn_out = self.filter_function(gnn_out)
        return gnn_out

