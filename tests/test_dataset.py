"""Tests for dataset loader."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset_loader import get_class_names, load_dataset_paths


@pytest.fixture
def temp_dataset():
    """Create temporary test dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        classes = ['healthy', 'diseased']
        
        for class_name in classes:
            class_dir = tmpdir / class_name
            class_dir.mkdir()
            
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(class_dir / f'image_{i}.jpg')
        
        yield tmpdir


def test_get_class_names(temp_dataset):
    """Test class name discovery."""
    classes = get_class_names(str(temp_dataset))
    assert len(classes) == 2
    assert 'diseased' in classes
    assert 'healthy' in classes


def test_load_dataset_paths(temp_dataset):
    """Test dataset loading."""
    classes = get_class_names(str(temp_dataset))
    paths, labels = load_dataset_paths(str(temp_dataset), classes)
    
    assert len(paths) == 10
    assert len(labels) == 10
    assert all(isinstance(p, str) for p in paths)


def test_get_dataloaders(temp_dataset):
    """Test dataloader creation."""
    from src.training.dataset_loader import get_dataloaders
    
    dataloaders = get_dataloaders(str(temp_dataset), batch_size=4)
    
    assert 'train' in dataloaders
    assert 'val' in dataloaders
    assert 'test' in dataloaders
    assert 'class_names' in dataloaders


if __name__ == '__main__':
    pytest.main([__file__])
