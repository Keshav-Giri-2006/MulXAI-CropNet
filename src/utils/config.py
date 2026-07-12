"""Configuration management for MulXAI-CropNet project."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_root: str = 'datasets/PlantVillage/'
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    image_size: int = 224


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = 'efficientnetb0'
    num_classes: int = 10
    pretrained: bool = True
    dropout_rate: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    device: str = 'cpu'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    seed: int = 42
    optimizer: str = 'AdamW'  # Research-approved optimizer


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        data = DataConfig(**config_dict.get('data', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        return cls(data=data, model=model, training=training)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
        }
