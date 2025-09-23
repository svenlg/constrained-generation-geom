import os
import numpy as np
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader

from flowmol.analysis.molecule_builder import SampledMolecule

class GenDataset(Dataset):
    """Dataset for GEOM molecular data with XTB property calculation."""
    
    def __init__(
        self, 
        property: str,
        experiment: str, 
        split: str,
        max_nodes: int = 60,
    ):
        self.experiment = experiment
        self.split = split
        self.max_nodes = max_nodes
        if property == "dipole_zero":
            self.property = "dipole"
            self.set_zero_dipole = True
        else:
            self.property = property
            self.set_zero_dipole = False

        self.path = f"data/{experiment}"
        self.mol_path = self.path + "/molecules"
        df = pd.read_csv(f"{self.path}/{self.split}.csv")
        self.df = df

    def _load_mols(self, idx) -> SampledMolecule:
        mol_idx = self.df.iloc[idx]['id_str']
        mol_path = os.path.join(self.mol_path, f"{mol_idx}.bin")
        mols, aux = dgl.load_graphs(mol_path)
        return mols[0]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        while True:
            if np.isnan(self.df.iloc[idx][self.property]):
                idx = self.df.sample().index[0]
                continue
            mol = self._load_mols(idx)
            num_atoms = mol.num_nodes()
            if num_atoms <= self.max_nodes:
                break
            idx = self.df.sample().index[0]

        target_value = torch.tensor(self.df.iloc[idx][self.property], dtype=torch.float32)
        if self.set_zero_dipole:
            target_value = torch.tensor(0.0, dtype=torch.float32) if not self.df.iloc[idx]['all_atoms_connected'] else target_value
        if self.property == "score": # punish disconnected graphs
            target_value = torch.tensor(1.0, dtype=torch.float32) if not self.df.iloc[idx]['all_atoms_connected'] else target_value

        return mol, target_value


def make_loaders(experiment: str, property: str, batch_size: int, num_workers: int):
    train_set = GenDataset(property=property, experiment=experiment, split="train")
    val_set   = GenDataset(property=property, experiment=experiment, split="val")
    test_set  = GenDataset(property=property, experiment=experiment, split="test")

    train_loader = GraphDataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = GraphDataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = GraphDataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader