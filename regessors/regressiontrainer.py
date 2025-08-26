import time
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class RegressionTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: Optional[float] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            criterion: Loss function
            optimizer: Optimization algorithm
            device: Device to run the training on
            grad_clip: Maximum norm of the gradients
        """
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'acc': val_acc}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, list]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
            
        Returns:
            Dictionary containing training history
        """
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['acc'])
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.2f}%")
            if val_loader is not None:
                print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.2f}%")
            print(f"Time: {epoch_time:.2f}s")
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs: Any):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            **kwargs
        }
        torch.save(checkpoint, path)

    def save_model(self, path: str):
        """Save the model state."""
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint



if __name__ == "__main__":
    # Create your model, criterion, and optimizer

    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)
        
    model = TestModel()

    # Initialize the trainer
    trainer = RegressionTrainer(
        model=model,
        grad_clip=1.0  # Optional gradient clipping
    )

    # Create Data 
    train_data = torch.randn(1000, 10)
    train_labels = torch.randn(1000, 1)

    val_data = torch.randn(100, 10)
    val_labels = torch.randn(100, 1)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10
    )
    print(history)
