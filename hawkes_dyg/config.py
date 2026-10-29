"""Dataset and experiment configuration for the Hawkes-DyG baseline.

The configuration deliberately keeps the first implementation small.  The
old ``IJCNN.py`` file remains the original FHGSN implementation; this package
does not change its assumptions or outputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    filename: str
    case_col: str
    activity_col: str
    timestamp_col: str
    attribute_cols: List[str]
    numeric_attribute_cols: List[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return DATASET_DIR / self.filename


@dataclass
class ExperimentConfig:
    seed: int = 42
    # ``split_seed`` is deliberately independent from ``seed``.  Hyperparameter
    # trials can therefore keep exactly the same case partitions while changing
    # model initialization.
    split_seed: int = 42
    folds: int = 3
    validation_ratio: float = 0.15
    min_prefix_len: int = 2
    time_bins: int = 20
    embedding_dim: int = 64
    hidden_dim: int = 64
    graph_layers: int = 2
    history_window: int = 32
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    selection_metric: str = "validation_nll"
    initial_alpha: float = 0.5
    initial_beta: float = 0.5
    initial_mu: float = 0.1
    intensity_scale_init: float = 0.1
    intensity_scale_max: float = 1.0
    omega_scale_init: float = 0.1
    omega_scale_max: float = 1.0
    skip_test: bool = False
    num_workers: int = 0
    cpu_threads: int = 4
    output_dir: Path = PROJECT_ROOT / "results"


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "BPIC_2012_W": DatasetConfig(
        name="BPIC_2012_W",
        filename="BPIC_2012_W.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=["lifecycle:transition"],
    ),
    "BPI_Challenge_2013_C": DatasetConfig(
        name="BPI_Challenge_2013_C",
        filename="BPI_Challenge_2013_C.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=[
            "impact",
            "lifecycle:transition",
        ],
    ),
    "Loan_Application_events": DatasetConfig(
        name="Loan_Application_events",
        filename="Loan_Application_events.csv",
        case_col="case:concept:name",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=[
            "Action",
            "org:resource",
        ],
    ),
    "BPIC_2020_D": DatasetConfig(
        name="BPIC_2020_D",
        filename="BPIC_2020_D.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=["org:resource", "org:role"],
    ),
    "BPIC_2020_I": DatasetConfig(
        name="BPIC_2020_I",
        filename="BPIC_2020_I.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=["org:resource", "org:role"],
    ),
    "BPIC_2020_Pe": DatasetConfig(
        name="BPIC_2020_Pe",
        filename="BPIC_2020_Pe.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=["org:resource", "org:role"],
    ),
    "BPIC_2020_Pr": DatasetConfig(
        name="BPIC_2020_Pr",
        filename="BPIC_2020_Pr.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=["org:resource", "org:role"],
    ),
    "BPIC_2020_R": DatasetConfig(
        name="BPIC_2020_R",
        filename="BPIC_2020_R.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=["org:resource", "org:role"],
    ),
    "Receipt": DatasetConfig(
        name="Receipt",
        filename="Receipt.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=["org:resource", "org:group"],
    ),
    "Helpdesk": DatasetConfig(
        name="Helpdesk",
        filename="Helpdesk_normalized.csv",
        case_col="case",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        attribute_cols=[
            "org:resource",
            "service_level",
        ],
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Return a dataset configuration, accepting a few convenient aliases."""

    aliases = {
        "BPIC2012": "BPIC_2012_W",
        "BPIC_2012": "BPIC_2012_W",
        "BPIC2013": "BPI_Challenge_2013_C",
        "BPIC_2013": "BPI_Challenge_2013_C",
        "BPI2013": "BPI_Challenge_2013_C",
        "Loan": "Loan_Application_events",
        "BPIC2020D": "BPIC_2020_D",
        "BPIC2020I": "BPIC_2020_I",
        "BPIC2020Pe": "BPIC_2020_Pe",
        "BPIC2020Pr": "BPIC_2020_Pr",
        "BPIC2020R": "BPIC_2020_R",
        "Helpdesk_normalized": "Helpdesk",
    }
    canonical = aliases.get(name, name)
    if canonical not in DATASET_CONFIGS:
        choices = ", ".join(DATASET_CONFIGS)
        raise KeyError(f"Unknown dataset {name!r}. Choose one of: {choices}")
    return DATASET_CONFIGS[canonical]
