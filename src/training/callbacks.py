"""Training callbacks for monitoring and controlling training."""

import torch
from typing import Callable, Optional


class Callback:
    """Base callback class."""

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        pass

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass


class ModelCheckpoint(Callback):
    """Save model checkpoint when metric improves."""

    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf') if 'loss' in monitor else float('-inf')

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return

        should_save = False
        if 'loss' in self.monitor:
            should_save = current < self.best
            if should_save:
                self.best = current
        else:
            should_save = current > self.best
            if should_save:
                self.best = current

        if should_save or not self.save_best_only:
            print(f"Saving model to {self.filepath}")


class LearningRateMonitor(Callback):
    """Monitor learning rate during training."""

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        if 'lr' in metrics:
            print(f"Learning rate: {metrics['lr']:.6f}")


class MetricsLogger(Callback):
    """Log training metrics."""

    def __init__(self):
        self.history = {'train': {}, 'val': {}}

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        for key, value in metrics.items():
            if key not in self.history['train']:
                self.history['train'][key] = []
            self.history['train'][key].append(value)

    def get_history(self):
        return self.history
