import os
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import dgl
from dgl.dataloading import GraphDataLoader

from flowmol.analysis.molecule_builder import SampledMolecule

DATA_FRAME_COLS = [
    "score",
    "homolumo_gap",
    "lumo",
    "homo",
    "dipole",
    "energy",
    "id_str"
]

class GenDataset(Dataset):
    """Dataset for GEOM molecular data with XTB property calculation."""
    
    def __init__(
        self, 
        experiment: str, 
        split: str = "train",
        property_name: str = "dipole",
    ):
        self.experiment = experiment
        self.split = split
        self.property_name = property_name
        self.path = f"data/{experiment}"
        df = pd.read_csv(f"{self.path}/{self.split}.csv")
        df = df[DATA_FRAME_COLS]
        self.df = df
        self.mol_path = f"data/{experiment}/molecules"
        self.property_name = property_name
    
    def _load_mols(self, idx) -> SampledMolecule:
        mol_idx = self.df.iloc[idx]['id_str']
        mol_path = os.path.join(self.mol_path, f"{mol_idx}.bin")
        mol, aux = dgl.load_graphs(mol_path)
        return mol

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        mol_data = self._load_mols(idx)
        mol = mol_data.g

        target_value = torch.tensor(self.df.iloc[idx][self.property_name], dtype=torch.float32)

        return mol, target_value

class GeomDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for GEOM dataset."""
    
    def __init__(
        self,
        experiment: str,
        property_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.experiment = experiment
        self.property_name = property_name
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = GenDataset(
                self.experiment, "train", self.property_name
            )
            self.val_dataset = GenDataset(
                self.experiment, "val", self.property_name
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = GenDataset(
                self.experiment, "test", self.property_name
            )
    
    def train_dataloader(self) -> DataLoader:
        return GraphDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return GraphDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return GraphDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

