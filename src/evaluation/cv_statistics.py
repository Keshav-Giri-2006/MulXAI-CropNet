# src/evaluation/cv_statistics.py (NEW FILE)
def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Returns (ci_low, ci_high)."""

def aggregate_cv_metrics(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Returns {metric_name: {'mean', 'std', 'ci_low', 'ci_high'}} for
    accuracy, precision, recall, f1."""

def format_cv_summary(aggregated: Dict[str, Dict[str, float]]) -> str:
    """Formatted text report."""

def save_cv_results(
    fold_metrics: List[Dict[str, float]],
    aggregated: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """Writes CSV to outputs/metrics/."""