def generate_stratified_folds(
    labels: List[int],
    n_splits: int = 10,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Returns list of (train_idx, val_idx) index arrays into the CV pool."""

def build_fold_dataloaders(
    cv_pool_paths: List[str],
    cv_pool_labels: List[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    class_names: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Returns {'train': DataLoader, 'val': DataLoader} for one fold."""

def train_single_fold(
    fold_idx: int,
    fold_loaders: Dict[str, DataLoader],
    model_name: str,
    num_classes: int,
    num_epochs: int = 30,
    lr: float = 0.001,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Trains one fresh model instance for this fold, returns validation metrics
    dict: {'accuracy', 'precision', 'recall', 'f1', 'loss'}."""

def run_cross_validation(
    dataset_root: str,
    model_name: str = 'efficientnetb0_se',
    n_splits: int = 10,
    batch_size: int = 32,
    num_epochs: int = 30,
    lr: float = 0.001,
    device: str = 'cpu',
    random_seed: int = 42,
) -> List[Dict[str, float]]:
    """Top-level orchestrator. Returns list of n_splits per-fold metric dicts."""
