"""Create a dummy checkpoint for testing without training."""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_utils import create_model


def create_dummy_checkpoint(model_name: str, num_classes: int, output_path: str) -> None:
    """
    Create a dummy model checkpoint for testing.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of disease classes
        output_path: Path to save checkpoint
    """
    # Create model
    model = create_model(model_name, num_classes=num_classes, device='cpu')
    
    # Create dummy optimizer state
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create checkpoint
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': dummy_optimizer.state_dict(),
        'metrics': {
            'accuracy': 0.0,
            'loss': 0.0,
        }
    }
    
    # Save checkpoint
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    
    print(f"✅ Dummy checkpoint created: {output_path}")
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    print(f"   Size: {Path(output_path).stat().st_size / 1024:.2f} KB")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create dummy checkpoint for testing')
    parser.add_argument('--model', type=str, default='efficientnetb0', 
                       help='Model name (efficientnetb0, mobilenetv3, efficientnetb0_se)')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--output', type=str, default='checkpoints/dummy_model.pth',
                       help='Output checkpoint path')
    
    args = parser.parse_args()
    create_dummy_checkpoint(args.model, args.num_classes, args.output)


if __name__ == '__main__':
    main()
