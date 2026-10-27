"""Validation-only random search for Hawkes-DyG hyperparameters.

The tuner deliberately keeps the test split out of model selection.  Each trial
uses a fresh output directory and the same case split seed, while model
initialization may be changed with ``--model-seed``.  The runner must expose a
``skip_test`` argument; this prevents an accidental test evaluation during
search and gives a clear error when an older runner is used.

Example::

    python scripts/tune_hawkes_dyg.py --dataset Helpdesk --device cpu \
        --trials 28 --output-dir tuning/helpdesk
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Make ``python scripts/tune_hawkes_dyg.py`` work from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawkes_dyg.config import DATASET_CONFIGS, ExperimentConfig
from hawkes_dyg.runner import run_dataset

LOGGER = logging.getLogger(__name__)

# Deliberately small random-search space.  The values are part of the protocol
# and are kept stable so that a published tuning run is reproducible.
SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "hidden_dim": (32, 64, 128),
    "dropout": (0.1, 0.2, 0.4),
    "weight_decay": (0.0, 1e-5, 1e-4),
    "graph_layers": (1, 2),
}
DEFAULT_TRIALS = 28
DEFAULT_SEARCH_SEED = 42
DIAGNOSTIC_KEYS = (
    "validation_f1", "best_epoch", "alpha_mean", "beta", "mu_mean",
    "intensity_scale", "omega_scale", "attention_entropy",
)


def sample_trials(
    count: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEARCH_SEED,
    *,
    search_space: Mapping[str, Sequence[Any]] = SEARCH_SPACE,
) -> list[dict[str, Any]]:
    """Return ``count`` deterministic, unique random-search configurations.

    The search is sampled without replacement from the finite Cartesian product.
    A local ``random.Random`` object is used so calling this helper never changes
    global Python or torch random state.
    """
    if count < 1 or count > 32:
        raise ValueError("count must be between 1 and 32")
    keys = tuple(search_space)
    choices = [tuple(search_space[key]) for key in keys]
    if any(not values for values in choices):
        raise ValueError("search_space values must be non-empty")
    total = 1
    for values in choices:
        total *= len(values)
    if count > total:
        raise ValueError(f"count={count} exceeds {total} unique configurations")
    rng = random.Random(seed)
    selected: set[tuple[Any, ...]] = set()
    ordered: list[tuple[Any, ...]] = []
    while len(ordered) < count:
        candidate = tuple(rng.choice(values) for values in choices)
        if candidate in selected:
            continue
        selected.add(candidate)
        ordered.append(candidate)
    return [dict(zip(keys, values)) for values in ordered]


def config_for_trial(
    base: ExperimentConfig,
    trial: Mapping[str, Any],
    *,
    model_seed: int,
    split_seed: int,
) -> ExperimentConfig:
    """Apply a trial while keeping the split seed invariant across trials."""
    fields = set(getattr(ExperimentConfig, "__dataclass_fields__", {}))
    kwargs = {name: trial[name] for name in trial if name in fields}
    if "seed" in fields:
        kwargs["seed"] = model_seed
    if "split_seed" in fields:
        kwargs["split_seed"] = split_seed
    return replace(base, **kwargs)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def validation_nll_from_results(results: Any) -> float | None:
    """Extract validation NLL from runner return data, never test NLL.

    New runners return fold dictionaries containing ``validation_nll`` (or
    ``best_validation_loss``).  The fallback names make this compatible with
    intermediate versions while still refusing to use a bare ``nll`` key,
    which is the test metric in the historical runner.
    """
    if not isinstance(results, Iterable) or isinstance(results, (str, bytes, Mapping)):
        return None
    values: list[float] = []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        value = None
        for key in ("validation_nll", "best_validation_nll", "best_validation_loss"):
            value = _as_float(row.get(key))
            if value is not None:
                break
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def diagnostics_from_results(results: Any) -> dict[str, float]:
    """Average fold diagnostics for the trial report.

    Metrics are intentionally reduced over folds rather than selecting a fold
    by its test score.  Missing diagnostics are simply omitted so the tuner
    remains compatible with a runner that predates a particular statistic.
    """
    if not isinstance(results, Iterable) or isinstance(results, (str, bytes, Mapping)):
        return {}
    values: dict[str, list[float]] = {key: [] for key in DIAGNOSTIC_KEYS}
    for row in results:
        if not isinstance(row, Mapping):
            continue
        for key in DIAGNOSTIC_KEYS:
            value = _as_float(row.get(key))
            if value is not None:
                values[key].append(value)
    return {
        key: sum(numbers) / len(numbers)
        for key, numbers in values.items() if numbers
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def validation_nll_from_artifacts(output_dir: Path) -> float:
    """Read validation-only artifacts emitted by a trial.

    Accepted formats are a top-level ``validation_summary.json``/CSV or one
    JSON metrics file per fold.  Test ``nll`` fields are deliberately ignored.
    """
    output_dir = Path(output_dir)
    for name in ("validation_summary.json", "tuning_metrics.json"):
        data = _read_json(output_dir / name)
        value = validation_nll_from_results(data)
        if value is not None:
            return value
    for csv_name in ("validation_summary.csv", "summary.csv"):
        csv_path = output_dir / csv_name
        if csv_path.is_file():
            values = []
            with csv_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    # ``summary.csv`` has Average/Std rows as well as folds;
                    # only fold rows are included to avoid double averaging.
                    if row.get("fold") in {"Average", "Std"}:
                        continue
                    for key in ("validation_nll", "best_validation_nll", "best_validation_loss"):
                        value = _as_float(row.get(key))
                        if value is not None:
                            values.append(value)
                            break
            if values:
                return sum(values) / len(values)
    values = []
    for metrics_path in sorted(output_dir.glob("fold_*/metrics.json")):
        row = _read_json(metrics_path)
        value = validation_nll_from_results([row])
        if value is not None:
            values.append(value)
    if values:
        return sum(values) / len(values)
    raise ValueError(
        f"No validation NLL found under {output_dir}; the runner must emit "
        "validation_nll/best_validation_loss when --skip-test is used"
    )


def _build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--search-seed", type=int, default=DEFAULT_SEARCH_SEED)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--time-bins", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-prefixes", type=int, default=None)
    # Tuning always skips test.  The explicit flag documents the protocol and
    # makes shell command lines self-auditing.
    parser.add_argument("--skip-test", dest="skip_test", action="store_true", default=True)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def run_trial(args: argparse.Namespace, trial_index: int, trial: Mapping[str, Any]) -> dict[str, Any]:
    """Run one validation-only trial and return a JSON-serializable record."""
    trial_dir = Path(args.output_dir) / f"trial_{trial_index:03d}"
    if trial_dir.exists() and any(trial_dir.iterdir()):
        raise FileExistsError(f"Trial output already exists: {trial_dir}")
    config_fields = set(getattr(ExperimentConfig, "__dataclass_fields__", {}))
    base_kwargs: dict[str, Any] = {
        "seed": args.model_seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "time_bins": args.time_bins,
        "cpu_threads": args.cpu_threads,
        "num_workers": args.num_workers,
        "output_dir": trial_dir,
    }
    if "split_seed" in config_fields:
        base_kwargs["split_seed"] = args.split_seed
    if "selection_metric" in config_fields:
        base_kwargs["selection_metric"] = "validation_nll"
    if "skip_test" in config_fields:
        base_kwargs["skip_test"] = bool(args.skip_test)
    base = ExperimentConfig(**base_kwargs)
    config = config_for_trial(base, trial, model_seed=args.model_seed, split_seed=args.split_seed)
    signature = inspect.signature(run_dataset)
    supports_skip_test = "skip_test" in signature.parameters or "skip_test" in config_fields
    if not supports_skip_test:
        raise RuntimeError(
            "The installed hawkes_dyg.runner does not support skip_test. "
            "Use the updated runner before starting validation-only tuning."
        )
    call_kwargs: dict[str, Any] = {
        "experiment": config,
        "device_name": args.device,
        "max_cases": args.max_cases,
        "max_prefixes": args.max_prefixes,
        "data_dir": args.data_dir,
    }
    # Current runners carry this protocol switch on ExperimentConfig; an
    # older transitional runner may expose it as a run_dataset argument.
    if "skip_test" in signature.parameters:
        call_kwargs["skip_test"] = bool(args.skip_test)
    if "split_seed" in signature.parameters:
        call_kwargs["split_seed"] = args.split_seed
    results = run_dataset(args.dataset, **call_kwargs)
    score = validation_nll_from_results(results)
    if score is None:
        score = validation_nll_from_artifacts(trial_dir / DATASET_CONFIGS[args.dataset].name)
    record = {
        "trial": trial_index,
        **dict(trial),
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "validation_nll": float(score),
        "output_dir": str(trial_dir),
    }
    record.update(diagnostics_from_results(results))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 24 <= args.trials <= 32:
        parser.error("--trials must be between 24 and 32")
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials = sample_trials(args.trials, args.search_seed)
    records: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, 1):
        LOGGER.info("trial=%d/%d params=%s", index, len(trials), trial)
        records.append(run_trial(args, index, trial))
        records.sort(key=lambda row: row["validation_nll"])
        (args.output_dir / "trials.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (args.output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["trial", *SEARCH_SPACE, "model_seed", "split_seed",
                            "validation_nll", *DIAGNOSTIC_KEYS, "output_dir"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(records)
    LOGGER.info("Best validation NLL: %.6f (trial %03d)", records[0]["validation_nll"], records[0]["trial"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
