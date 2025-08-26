import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import dgl
from dgl.dataloading import GraphDataLoader
import numpy as np

from xtb_calculator import XTBCalculator


class GeomDataset(Dataset):
    """Dataset for GEOM molecular data with XTB property calculation."""
    
    def __init__(
        self, 
        data_path: str, 
        property_name: str,
        min_atoms: int = 20,
        max_atoms: int = 75,
        split: str = "train",
        cache_properties: bool = True
    ):
        self.data_path = data_path
        self.property_name = property_name
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.split = split
        self.cache_properties = cache_properties
        
        # Initialize XTB calculator
        self.xtb_calculator = XTBCalculator(property_name)
        
        # Load and filter data
        self.data = self._load_and_filter_data()
        
        # Cache for computed properties
        self.property_cache = {}
        self.cache_file = os.path.join(data_path, f"{split}_{property_name}_cache.pkl")
        
        if cache_properties and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.property_cache = pickle.load(f)
        
        print(f"Loaded {len(self.data)} molecules for {split} split")
    
    def _load_and_filter_data(self) -> List[Dict]:
        """Load GEOM data and filter by atom count."""
        # Adjust path for your data structure: data/train_data.pickle, etc.
        split_path = os.path.join(self.data_path, f"{self.split}_data.pickle")
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Data file not found: {split_path}")
        
        with open(split_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        filtered_data = []
        for i, mol_data in enumerate(raw_data):
            # Count atoms
            if 'coords' in mol_data:
                n_atoms = len(mol_data['coords'])
            elif 'positions' in mol_data:
                n_atoms = len(mol_data['positions'])
            elif 'atomic_numbers' in mol_data:
                n_atoms = len(mol_data['atomic_numbers'])
            else:
                continue
            
            # Filter by atom count
            if self.min_atoms <= n_atoms <= self.max_atoms:
                # Add index for caching
                mol_data['index'] = i
                filtered_data.append(mol_data)
        
        return filtered_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        mol_data = self.data[idx]
        
        # Create DGL graph from molecular data
        graph = self._mol_to_dgl_graph(mol_data)
        
        # Get or calculate target property
        mol_index = mol_data['index']
        if mol_index in self.property_cache:
            target_value = self.property_cache[mol_index]
        else:
            # Calculate property using XTB
            target_value = self.xtb_calculator.calculate_property(mol_data)
            
            # Cache the result
            if self.cache_properties and not np.isnan(target_value):
                self.property_cache[mol_index] = target_value
                # Periodically save cache
                if len(self.property_cache) % 100 == 0:
                    self._save_cache()
        
        # Skip molecules with failed calculations
        if np.isnan(target_value):
            # Return a dummy sample, will be filtered out in collate_fn
            return None, None
        
        target = torch.tensor(target_value, dtype=torch.float32)
        return graph, target
    
    def _save_cache(self):
        """Save property cache to disk."""
        if self.cache_properties:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.property_cache, f)
    
    def finalize_cache(self):
        """Save final cache state."""
        self._save_cache()
    
    def _mol_to_dgl_graph(self, mol_data: Dict) -> dgl.DGLGraph:
        """Convert molecular data to DGL graph."""
        # Extract atomic numbers and coordinates
        atomic_numbers = torch.tensor(mol_data['atomic_numbers'], dtype=torch.long)
        coords = torch.tensor(mol_data['coords'], dtype=torch.float32)
        
        n_atoms = len(atomic_numbers)
        
        # Create node features
        # One-hot encode atomic numbers (up to atomic number 100)
        atom_features = F.one_hot(atomic_numbers.clamp(0, 99), num_classes=100).float()
        
        # Add coordinate features
        node_features = torch.cat([atom_features, coords], dim=1)
        
        # Create edges (complete graph or based on distance threshold)
        edge_threshold = 5.0  # Angstroms
        
        # Calculate pairwise distances
        coords_expanded_1 = coords.unsqueeze(1).expand(-1, n_atoms, -1)
        coords_expanded_2 = coords.unsqueeze(0).expand(n_atoms, -1, -1)
        distances = torch.norm(coords_expanded_1 - coords_expanded_2, dim=2)
        
        # Create edges for atoms within threshold (excluding self-loops)
        src_nodes, dst_nodes = torch.where((distances < edge_threshold) & (distances > 0))
        
        # Edge features (distances and relative coordinates)
        edge_distances = distances[src_nodes, dst_nodes].unsqueeze(1)
        relative_coords = coords[dst_nodes] - coords[src_nodes]
        edge_features = torch.cat([edge_distances, relative_coords], dim=1)
        
        # Create DGL graph
        graph = dgl.graph((src_nodes, dst_nodes))
        
        # Set node and edge features
        graph.ndata['a_t'] = atomic_numbers.float().unsqueeze(1)  # Atomic numbers
        graph.ndata['c_t'] = coords  # Coordinates
        graph.ndata['x_t'] = atom_features  # One-hot atomic features
        graph.edata['e_t'] = edge_features  # Edge features
        
        return graph


def collate_fn(batch):
    """Custom collate function to handle None values from failed XTB calculations."""
    # Filter out None values
    batch = [(graph, target) for graph, target in batch if graph is not None]
    
    if len(batch) == 0:
        return None, None
    
    graphs, targets = zip(*batch)
    
    # Batch graphs using DGL
    batched_graph = dgl.batch(graphs)
    batched_targets = torch.stack(targets)
    
    return batched_graph, batched_targets


class GeomDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for GEOM dataset."""
    
    def __init__(
        self,
        data_path: str,
        property_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        min_atoms: int = 20,
        max_atoms: int = 75,
    ):
        super().__init__()
        self.data_path = data_path
        self.property_name = property_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = GeomDataset(
                self.data_path, self.property_name, self.min_atoms, self.max_atoms, "train"
            )
            self.val_dataset = GeomDataset(
                self.data_path, self.property_name, self.min_atoms, self.max_atoms, "val"
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = GeomDataset(
                self.data_path, self.property_name, self.min_atoms, self.max_atoms, "test"
            )
    
    def train_dataloader(self) -> DataLoader:
        return GraphDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return GraphDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        return GraphDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Save property caches when done."""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.finalize_cache()
        if hasattr(self, 'val_dataset'):
            self.val_dataset.finalize_cache()
        if hasattr(self, 'test_dataset'):
            self.test_dataset.finalize_cache()