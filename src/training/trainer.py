"""
Model trainer for disease classification.

Handles complete training pipeline with validation and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np
from pathlib import Path
from src.models.model_utils import save_checkpoint
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import Metrics


class Trainer:
    """Trainer for disease classification models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        checkpoint_dir: str = './checkpoints',
    ) -> None:
        """Initialize Trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.evaluator = ModelEvaluator(model, device)
        self.train_history = {'loss': [], 'accuracy': [], 'f1': []}
        self.val_history = {'loss': [], 'accuracy': [], 'f1': []}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        metrics = Metrics.compute_metrics(np.array(all_targets), np.array(all_preds))
        metrics['loss'] = total_loss / len(self.train_loader)
        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        return self.evaluator.evaluate(self.val_loader, self.criterion)

    def train(self, num_epochs: int, scheduler=None, save_best: bool = True) -> Dict:
        """Complete training loop (AdamW optimizer - approved specification)."""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.train_history['f1'].append(train_metrics['f1'])

            val_metrics = self.validate()
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['accuracy'].append(val_metrics['accuracy'])
            self.val_history['f1'].append(val_metrics['f1'])

            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(self.model, self.optimizer, epoch, 
                              str(self.checkpoint_dir / 'best_model.pth'), val_metrics)

            if scheduler is not None:
                scheduler.step(val_metrics['loss'])

        return {'train': self.train_history, 'val': self.val_history}

    def get_training_history(self) -> Dict:
        """Get training history."""
        return {'train': self.train_history, 'val': self.val_history}
