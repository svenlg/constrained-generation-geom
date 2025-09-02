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
        property: str,
        experiment: str, 
        split: str = "train",
    ):
        self.experiment = experiment
        self.split = split
        if property == "dipole_zero":
            self.property = "dipole"
            self.set_zero_dipole = True
        else:
            self.property = property
            self.set_zero_dipole = False

        self.path = f"data/{experiment}"
        self.mol_path = self.path + "/molecules"
        df = pd.read_csv(f"{self.path}/{self.split}.csv")
        df = df[DATA_FRAME_COLS]
        self.df = df

    def _load_mols(self, idx) -> SampledMolecule:
        mol_idx = self.df.iloc[idx]['id_str']
        mol_path = os.path.join(self.mol_path, f"{mol_idx}.bin")
        mols, aux = dgl.load_graphs(mol_path)
        return mols[0]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        mol = self._load_mols(idx)

        target_value = torch.tensor(self.df.iloc[idx][self.property], dtype=torch.float32)
        if self.set_zero_dipole:
            target_value = torch.tensor(0.0, dtype=torch.float32) if self.df.iloc[idx]['score'] > 0.0 else target_value

        return mol, target_value

class GeomDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for GEOM dataset."""
    
    def __init__(
        self,
        experiment: str,
        property: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.experiment = experiment
        self.property = property
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = GenDataset(
                self.property, self.experiment, "train",
            )
            self.val_dataset = GenDataset(
                self.property, self.experiment, "val"
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = GenDataset(
                self.property, self.experiment, "test"
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

