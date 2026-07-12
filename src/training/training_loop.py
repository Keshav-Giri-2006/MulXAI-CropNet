"""Training loop utilities for flexible training procedures."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


def train_step(model, dataloader, criterion, optimizer, device='cpu') -> float:
    """Single training step."""
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def val_step(model, dataloader, criterion, device='cpu') -> float:
    """Single validation step."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def reset(self) -> None:
        self.counter = 0
        self.best_loss = None
