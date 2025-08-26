import os
import pickle
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Any
import argparse
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb

import dgl
from dgl.dataloading import GraphDataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem

from model.gnn import GNN


class XTBCalculator:
    """Wrapper for XTB calculations."""
    
    def __init__(self, property_name: str):
        self.property_name = property_name.lower()
        self.temp_dir = None
        
        # Check if XTB is available
        if not shutil.which("xtb"):
            raise RuntimeError("XTB not found in PATH. Please install XTB.")
    
    def calculate_property(self, mol_data: Dict) -> float:
        """Calculate molecular property using XTB."""
        try:
            # Create temporary directory for XTB calculation
            with tempfile.TemporaryDirectory() as temp_dir:
                xyz_file = os.path.join(temp_dir, "molecule.xyz")
                
                # Write XYZ file
                self._write_xyz_file(mol_data, xyz_file)
                
                # Run XTB calculation based on property type
                result = self._run_xtb_calculation(xyz_file, temp_dir)
                
                return result
                
        except Exception as e:
            print(f"XTB calculation failed: {e}")
            return np.nan
    
    def _write_xyz_file(self, mol_data: Dict, xyz_file: str):
        """Write molecular data to XYZ file."""
        atomic_numbers = mol_data['atomic_numbers']
        coords = mol_data['coords']
        
        # Convert atomic numbers to element symbols
        element_symbols = []
        for atomic_num in atomic_numbers:
            if atomic_num == 1:
                element_symbols.append('H')
            elif atomic_num == 6:
                element_symbols.append('C')
            elif atomic_num == 7:
                element_symbols.append('N')
            elif atomic_num == 8:
                element_symbols.append('O')
            elif atomic_num == 9:
                element_symbols.append('F')
            elif atomic_num == 15:
                element_symbols.append('P')
            elif atomic_num == 16:
                element_symbols.append('S')
            elif atomic_num == 17:
                element_symbols.append('Cl')
            elif atomic_num == 35:
                element_symbols.append('Br')
            else:
                # Add more elements as needed
                from rdkit.Chem import GetPeriodicTable
                pt = GetPeriodicTable()
                element_symbols.append(pt.GetElementSymbol(atomic_num))
        
        with open(xyz_file, 'w') as f:
            f.write(f"{len(atomic_numbers)}\n")
            f.write("Generated molecule\n")
            for symbol, coord in zip(element_symbols, coords):
                f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    def _run_xtb_calculation(self, xyz_file: str, temp_dir: str) -> float:
        """Run XTB calculation and extract property."""
        
        if self.property_name in ['energy', 'total_energy']:
            # Single point energy calculation
            cmd = ['xtb', xyz_file, '--sp']
            
        elif self.property_name in ['homo', 'lumo', 'gap', 'homo_lumo_gap']:
            # Electronic structure calculation
            cmd = ['xtb', xyz_file, '--sp', '--etemp', '300']
            
        elif self.property_name in ['dipole', 'dipole_moment']:
            # Dipole moment calculation
            cmd = ['xtb', xyz_file, '--sp', '--dipole']
            
        elif self.property_name in ['polarizability']:
            # Polarizability calculation
            cmd = ['xtb', xyz_file, '--sp', '--polar']
            
        elif self.property_name in ['frequencies', 'freq']:
            # Frequency calculation
            cmd = ['xtb', xyz_file, '--ohess']
            
        else:
            # Default: single point energy
            cmd = ['xtb', xyz_file, '--sp']
        
        try:
            # Run XTB calculation
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"XTB failed with return code {result.returncode}")
            
            # Parse output based on property type
            return self._parse_xtb_output(result.stdout, temp_dir)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("XTB calculation timed out")
    
    def _parse_xtb_output(self, output: str, temp_dir: str) -> float:
        """Parse XTB output to extract the desired property."""
        
        if self.property_name in ['energy', 'total_energy']:
            # Extract total energy in Hartree
            for line in output.split('\n'):
                if 'TOTAL ENERGY' in line or 'total energy' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Eh' in part or 'hartree' in part.lower():
                            return float(parts[i-1])
                        elif part.replace('-', '').replace('.', '').isdigit() and len(part) > 3:
                            return float(part)
        
        elif self.property_name in ['homo']:
            # Extract HOMO energy
            for line in output.split('\n'):
                if 'HOMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            return float(part)
        
        elif self.property_name in ['lumo']:
            # Extract LUMO energy
            for line in output.split('\n'):
                if 'LUMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            return float(part)
        
        elif self.property_name in ['gap', 'homo_lumo_gap']:
            # Extract HOMO-LUMO gap
            homo, lumo = None, None
            for line in output.split('\n'):
                if 'HOMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            homo = float(part)
                if 'LUMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            lumo = float(part)
            if homo is not None and lumo is not None:
                return lumo - homo
        
        elif self.property_name in ['dipole', 'dipole_moment']:
            # Extract dipole moment
            for line in output.split('\n'):
                if 'dipole moment' in line.lower() and 'Debye' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('.', '').isdigit():
                            return float(part)
        
        # If we couldn't parse the property, try to extract any energy value
        for line in output.split('\n'):
            if any(keyword in line.lower() for keyword in ['energy', 'total']):
                parts = line.split()
                for part in parts:
                    try:
                        value = float(part)
                        if abs(value) > 0.1:  # Reasonable energy value
                            return value
                    except ValueError:
                        continue
        
        raise ValueError(f"Could not extract {self.property_name} from XTB output")


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