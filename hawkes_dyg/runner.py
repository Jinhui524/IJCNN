"""Case-level cross-validation for the tensorized Hawkes-DyG model."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
import inspect
import json
import logging
import math
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DATASET_CONFIGS, ExperimentConfig, get_dataset_config
from .data import FeatureEncoder, PrefixDataset, collate_prefix_batch, load_event_log
from .model import HawkesDYGNN

LOGGER = logging.getLogger(__name__)
METRICS = ["accuracy", "precision", "recall", "f1", "gmean", "nll"]
CASE_METRICS = [f"case_{name}" for name in METRICS]


def _union_metric_labels(labels, predictions) -> np.ndarray:
    """Return the FHGSN label universe for one metric computation."""
    return np.unique(np.concatenate([
        np.asarray(labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    ]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Requested {name} but CUDA is unavailable; use --device cpu or auto")
    return device


def macro_gmean(labels, predictions, num_classes=None) -> float:
    """Return FHGSN-compatible macro sensitivity/specificity G-Mean.

    Like the original IJCNN.py evaluation, sklearn infers the label universe
    from the union of true and predicted labels for this split.  ``num_classes``
    remains in the signature for caller compatibility but is intentionally not
    used to add absent classes to the average.
    """
    metric_labels = _union_metric_labels(labels, predictions)
    # With one observed/predicted class, specificity is undefined because
    # there are no negative examples.  The previous implementation reduced
    # this case to zero; return that value directly to avoid sklearn's warning
    # about a 1x1 confusion matrix.
    if metric_labels.size == 1:
        return 0.0
    cm = confusion_matrix(labels, predictions, labels=metric_labels)
    tp = np.diag(cm).astype(float)
    positive = cm.sum(axis=1).astype(float)
    negative = cm.sum() - positive
    tn = negative - (cm.sum(axis=0) - tp)
    sensitivity = np.divide(tp, positive, out=np.zeros_like(tp), where=positive != 0)
    specificity = np.divide(tn, negative, out=np.zeros_like(tn), where=negative != 0)
    return float(np.sqrt(sensitivity.mean() * specificity.mean()))


def _metric_dict(labels, predictions, nll: float, num_classes=None) -> dict:
    """Compute metrics with the same label-universe rule as FHGSN."""
    metric_labels = _union_metric_labels(labels, predictions)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions,
                                            labels=metric_labels,
                                            average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions,
                                      labels=metric_labels,
                                      average="macro", zero_division=0)),
        "f1": float(f1_score(labels, predictions,
                              labels=metric_labels,
                              average="macro", zero_division=0)),
        "gmean": macro_gmean(labels, predictions, num_classes=num_classes),
        "nll": float(nll),
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    nested_keys = {"attributes", "numeric_attributes", "attribute_missing"}
    return {
        key: ({name: tensor.to(device) for name, tensor in value.items()}
              if key in nested_keys else value.to(device))
        for key, value in batch.items()
    }


def _forward(model: HawkesDYGNN, batch: dict, return_aux: bool = False):
    kwargs = {key: value for key, value in batch.items() if key != "labels"}
    if return_aux:
        try:
            output = model(**kwargs, return_aux=True)
        except TypeError as error:
            # Keep evaluate useful for small external models that implement
            # only logits; the production Hawkes-DyG model exposes aux data.
            if "return_aux" not in str(error):
                raise
            return model(**kwargs), {}
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        return output, {}
    return model(**kwargs)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _mean_finite(values):
    """Return a JSON-safe mean for optional scalar/tensor diagnostics."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return None if finite.size == 0 else float(finite.mean())


def _attention_entropy(aux: dict, edge_mask: torch.Tensor | None = None):
    """Extract or derive a mean attention entropy from model auxiliary output."""
    if "attention_entropy" in aux:
        value = aux["attention_entropy"]
        if torch.is_tensor(value):
            value = value.detach().float().cpu().reshape(-1).tolist()
        elif np.isscalar(value):
            value = [float(value)]
        return value
    weights = aux.get("attention_weights", aux.get("attention"))
    if weights is None or not torch.is_tensor(weights):
        return []
    weights = weights.detach().float()
    if edge_mask is not None and weights.shape == edge_mask.shape:
        weights = weights.masked_fill(~edge_mask, 0.0)
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log())
    if edge_mask is not None and weights.shape == edge_mask.shape:
        entropy = entropy.masked_fill(~edge_mask, 0.0)
        denominator = edge_mask.sum(dim=-1).clamp_min(1)
        entropy = entropy.sum(dim=-1) / denominator
    else:
        entropy = entropy.sum(dim=-1)
    return entropy.reshape(-1).cpu().tolist()


def _case_level_metrics(case_ids, case_labels, case_predictions, case_losses, num_classes):
    """Average prefix metrics per case so long cases do not dominate reports."""
    grouped = {}
    for case_id, label, prediction, loss in zip(
        case_ids, case_labels, case_predictions, case_losses
    ):
        bucket = grouped.setdefault(str(case_id), {"labels": [], "predictions": [], "losses": []})
        bucket["labels"].append(label)
        bucket["predictions"].append(prediction)
        bucket["losses"].append(loss)
    if not grouped:
        return {}
    reports = [
        _metric_dict(value["labels"], value["predictions"],
                     float(np.mean(value["losses"])), num_classes=num_classes)
        for value in grouped.values()
    ]
    return {
        f"case_{key}": float(np.mean([report[key] for report in reports]))
        for key in METRICS
    }


def evaluate(model, loader, criterion, device, num_classes, predictions_path=None):
    model.eval()
    total_loss, count = 0.0, 0
    labels, predictions, sample_losses = [], [], []
    diagnostics = {"beta": [], "intensity_scale": [], "omega_scale": [], "attention_entropy": []}
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            logits, aux = _forward(model, batch, return_aux=True)
            loss = criterion(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")
            size = batch["labels"].numel()
            total_loss += float(loss) * size
            count += size
            batch_labels = batch["labels"].cpu().tolist()
            batch_predictions = logits.argmax(dim=-1).cpu().tolist()
            batch_losses = F.cross_entropy(logits, batch["labels"], reduction="none")
            labels.extend(batch_labels)
            predictions.extend(batch_predictions)
            sample_losses.extend(batch_losses.cpu().tolist())
            if isinstance(aux, dict):
                for key in ("beta", "intensity_scale", "omega_scale"):
                    if key in aux:
                        value = aux[key]
                        if torch.is_tensor(value):
                            value = value.detach().float().cpu().reshape(-1).tolist()
                        elif np.isscalar(value):
                            value = [float(value)]
                        diagnostics[key].extend(value)
                diagnostics["attention_entropy"].extend(
                    _attention_entropy(aux, aux.get("edge_mask"))
                )
    if count == 0:
        raise ValueError("Evaluation split contains no prefixes of length >= 2")
    nll = total_loss / count
    report = _metric_dict(labels, predictions, nll, num_classes=num_classes)
    # Validation/test loaders are deterministic (shuffle=False), so dataset
    # order maps exactly to the accumulated predictions.  If a caller passes a
    # shuffled loader, omit case aggregation rather than reporting misleading
    # case IDs.
    case_ids, case_labels, case_predictions = [], [], []
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "locate") and len(dataset) == len(labels):
        for index in range(len(labels)):
            trace_idx, _ = dataset.locate(index)
            case_ids.append(dataset.traces[trace_idx].case_id)
            case_labels.append(labels[index])
            case_predictions.append(predictions[index])
        report.update(_case_level_metrics(case_ids, case_labels, case_predictions,
                                          sample_losses, num_classes))
    for key, values in diagnostics.items():
        mean = _mean_finite(values)
        if mean is not None:
            report[key] = mean
    if predictions_path is not None:
        # Test loaders are ordered. Include case/prefix keys for later auditing.
        with predictions_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case_id", "prefix_length", "label_id", "prediction_id"])
            for index, (label, prediction) in enumerate(zip(labels, predictions)):
                trace_idx, prefix_length = dataset.locate(index)
                writer.writerow([dataset.traces[trace_idx].case_id, prefix_length, label, prediction])
    return nll, report


def make_optimizer(model: HawkesDYGNN, config: ExperimentConfig):
    # Influence parameters get only the task gradient: no L1, frequency prior,
    # or AdamW decay of unobserved pairs. Standard weight decay is on the rest.
    other = [parameter for name, parameter in model.named_parameters()
             if not name.startswith("influence.")]
    return torch.optim.AdamW([
        {"params": other, "weight_decay": config.weight_decay},
        {"params": list(model.influence.parameters()), "weight_decay": 0.0},
    ], lr=config.learning_rate)


def write_history_plot(path: Path, history: list) -> None:
    """A standalone SVG curve without a plotting-library dependency."""
    max_loss = max(max(row["train_loss"], row["validation_loss"]) for row in history) or 1.0
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300" viewBox="0 0 760 300">',
                '<rect width="760" height="300" fill="white"/>',
                '<g font-family="sans-serif" font-size="13" fill="#222">',
                '<text x="55" y="22">Cross-entropy by epoch</text>',
                '<path d="M55 40 V250 H735" fill="none" stroke="#999"/>',
                '<text x="5" y="48">' + f'{max_loss:.3f}' + '</text>',
                '<text x="35" y="250">0</text>',
                '<text x="55" y="272">1</text>',
                f'<text x="710" y="272">{len(history)}</text>']
    for name, color in [("train_loss", "#2563eb"), ("validation_loss", "#d97706")]:
        points = " ".join(f'{55 + 670*i/max(len(history)-1,1):.2f},{250 - 210*row[name]/max_loss:.2f}'
                          for i, row in enumerate(history))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    elements.extend(['<text x="420" y="22" fill="#2563eb">Training</text>',
                     '<text x="550" y="22" fill="#d97706">Validation</text>', '</g></svg>'])
    path.write_text("\n".join(elements), encoding="utf-8")


def load_trained_fold(fold_dir, device="cpu"):
    """Restore a model and its exact preprocessing without refitting anything."""
    fold_dir = Path(fold_dir)
    saved_config = json.loads((fold_dir / "model_config.json").read_text(encoding="utf-8"))
    model_version = saved_config.pop("model_version", None)
    if model_version != HawkesDYGNN.MODEL_VERSION:
        raise ValueError(
            f"Incompatible model version {model_version!r}; "
            f"expected {HawkesDYGNN.MODEL_VERSION!r}"
        )
    model_kwargs = saved_config
    encoder = FeatureEncoder.from_dict(json.loads((fold_dir / "preprocessor.json").read_text(encoding="utf-8")))
    model = HawkesDYGNN(**model_kwargs).to(device)
    model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, encoder


def _selection_value(metrics: dict, selection_metric: str):
    key = selection_metric.removeprefix("validation_")
    if key not in metrics:
        valid = ", ".join(sorted(metrics))
        raise ValueError(f"Unknown selection_metric {selection_metric!r}; available metrics: {valid}")
    value = float(metrics[key])
    # Loss/NLL are minimized; all report scores are maximized.
    return value, key == "nll"


def _influence_statistics(model, attention_entropy=None):
    """Collect checkpoint diagnostics without imposing a model implementation."""
    stats = {}
    influence = getattr(model, "influence", None)
    if influence is not None and hasattr(influence, "positive_parameters"):
        with torch.no_grad():
            alpha, beta, mu = influence.positive_parameters()
            alpha = alpha.detach().float().cpu()
            mu = mu.detach().float().cpu()
            stats.update({
                "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
                "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std(unbiased=False)),
                "beta": float(beta.detach().cpu()),
                "mu_min": float(mu.min()), "mu_max": float(mu.max()),
                "mu_mean": float(mu.mean()), "mu_std": float(mu.std(unbiased=False)),
            })
    if hasattr(model, "theta_intensity_scale"):
        with torch.no_grad():
            if hasattr(model, "intensity_scale_max"):
                scale = float(model.intensity_scale_max) * torch.sigmoid(
                    model.theta_intensity_scale.detach().float()
                )
            else:
                scale = F.softplus(model.theta_intensity_scale.detach().float())
            stats["intensity_scale"] = float(scale.cpu())
    scales = []
    for layer in getattr(model, "graph_layers", []):
        if hasattr(layer, "omega_scale_value"):
            with torch.no_grad():
                value = layer.omega_scale_value().detach().float()
                scales.append(float(value.cpu()))
    if scales:
        stats["omega_scale"] = float(np.mean(scales))
        stats["omega_scale_min"] = float(np.min(scales))
        stats["omega_scale_max"] = float(np.max(scales))
    if attention_entropy is not None:
        stats["attention_entropy"] = float(attention_entropy)
    return stats


def train_fold(train_loader, validation_loader, test_loader, model, config, fold_dir, device):
    optimizer = make_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    best_score = float("inf") if config.selection_metric.endswith("nll") else -float("inf")
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    checkpoint = fold_dir / "best_model.pt"
    with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss, count = 0.0, 0
            progress = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False, disable=not sys.stderr.isatty())
            for raw_batch in progress:
                batch = _move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(_forward(model, batch), batch["labels"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                size = batch["labels"].numel()
                total_loss += float(loss.detach()) * size
                count += size
                progress.set_postfix(loss=f"{loss.item():.4f}")
            if not count:
                raise ValueError("Empty training loader")
            val_loss, val_metrics = evaluate(model, validation_loader, criterion, device, model.num_activities)
            influence_stats = _influence_statistics(model)
            beta = influence_stats.get("beta", float("nan"))
            row = {"epoch": epoch, "train_loss": total_loss / count, "validation_loss": val_loss,
                   "beta": beta, **{f"validation_{key}": value for key, value in val_metrics.items()}}
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            LOGGER.info("epoch=%d train=%.5f val=%.5f val_acc=%.4f val_f1=%.4f beta=%.5f",
                        epoch, row["train_loss"], val_loss, val_metrics["accuracy"], val_metrics["f1"], beta)
            selection_value, minimize = _selection_value(val_metrics, config.selection_metric)
            improved = selection_value < best_score if minimize else selection_value > best_score
            if improved:
                best_score, best_loss, best_epoch, stale = selection_value, val_loss, epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= config.patience:
                    LOGGER.info("Early stopping at epoch %d", epoch)
                    break
    write_history_plot(fold_dir / "loss_curve.svg", history)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    _, validation_metrics = evaluate(model, validation_loader, criterion, device, model.num_activities)
    if config.skip_test or test_loader is None:
        # Tuning runs must not touch test predictions.  Keep the conventional
        # top-level metric names populated with validation scores so summaries
        # remain useful and machine-readable.
        metrics = dict(validation_metrics)
        metrics["test_evaluated"] = False
    else:
        _, metrics = evaluate(model, test_loader, criterion, device, model.num_activities,
                              predictions_path=fold_dir / "test_predictions.csv")
        metrics["test_evaluated"] = True
    metrics.update({f"validation_{key}": value for key, value in validation_metrics.items()})
    metrics.update(best_epoch=best_epoch, best_validation_loss=best_loss,
                   best_validation_nll=validation_metrics["nll"],
                   selection_metric=config.selection_metric,
                   best_selection_value=best_score)
    diagnostics = _influence_statistics(
        model, attention_entropy=validation_metrics.get("attention_entropy")
    )
    metrics.update(diagnostics)
    alpha = mu = None
    with torch.no_grad():
        if hasattr(model.influence, "positive_parameters"):
            alpha_tensor, _, mu_tensor = model.influence.positive_parameters()
            alpha = alpha_tensor.cpu().numpy()
            mu = mu_tensor.cpu().numpy()
    if alpha is not None:
        np.save(fold_dir / "alpha.npy", alpha)
    if mu is not None:
        np.save(fold_dir / "mu.npy", mu)
    save_json(fold_dir / "influence.json", {
        **diagnostics,
        "initial_alpha": config.initial_alpha,
        "initial_beta": config.initial_beta,
        "initial_mu": config.initial_mu,
        "intensity_scale_init": config.intensity_scale_init,
        "intensity_scale_max": config.intensity_scale_max,
        "omega_scale_init": config.omega_scale_init,
        "omega_scale_max": config.omega_scale_max,
        "selection_metric": config.selection_metric,
    })
    return metrics


def write_summary(path: Path, results: list) -> None:
    rows = list(results)
    if not results:
        return
    numeric_keys = []
    for key in results[0]:
        if key == "fold" or isinstance(results[0][key], (bool, str)):
            continue
        if all(key in result and isinstance(result[key], (int, float, np.number))
               for result in results):
            numeric_keys.append(key)
    for name, reducer in [("Average", np.mean), ("Std", np.std)]:
        rows.append({"fold": name, **{
            key: float(reducer([r[key] for r in results])) for key in numeric_keys
        }})
    fieldnames = ["fold"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_loader(dataset, batch_size, shuffle, workers, pin_memory=False):
    if not len(dataset):
        raise ValueError("A split has no prefixes of length >= 2; increase --max-cases for smoke tests")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=pin_memory, collate_fn=collate_prefix_batch, drop_last=False,
                      persistent_workers=workers > 0)


def validate_experiment(config, max_cases=None, max_prefixes=None):
    for name in ["epochs", "batch_size", "patience", "hidden_dim", "embedding_dim", "cpu_threads", "history_window"]:
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    # Zero layers is the no-message-passing ablation and is allowed.
    if config.graph_layers < 0:
        raise ValueError("graph_layers must be non-negative")
    if config.folds < 2 or config.time_bins < 2 or config.min_prefix_len < 1:
        raise ValueError("folds/time_bins must be >= 2 and min_prefix_len >= 1")
    if not 0 < config.validation_ratio < 1 or not 0 <= config.dropout < 1:
        raise ValueError("Invalid validation ratio or dropout")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.num_workers < 0:
        raise ValueError("Invalid learning rate, weight decay or worker count")
    if config.split_seed < 0:
        raise ValueError("split_seed must be non-negative")
    if config.selection_metric.removeprefix("validation_") not in METRICS:
        raise ValueError(f"selection_metric must be one of: {', '.join('validation_' + key for key in METRICS)}")
    if config.intensity_scale_init <= 0:
        raise ValueError("intensity_scale_init must be positive")
    if config.initial_alpha <= 0 or config.initial_beta <= 0 or config.initial_mu <= 0:
        raise ValueError("initial Hawkes parameters must be positive")
    if config.intensity_scale_max <= 0 or not 0 < config.intensity_scale_init <= config.intensity_scale_max:
        raise ValueError("intensity scale must satisfy 0 < init <= max")
    if config.omega_scale_max <= 0 or not 0 < config.omega_scale_init <= config.omega_scale_max:
        raise ValueError("omega scale must satisfy 0 < init <= max")
    for value in (max_cases, max_prefixes):
        if value is not None and value < 1:
            raise ValueError("Smoke-test limits must be positive")


def run_dataset(dataset_name, experiment=None, device_name="auto", max_cases=None,
                max_prefixes=None, data_dir=None):
    config = experiment or ExperimentConfig()
    validate_experiment(config, max_cases, max_prefixes)
    torch.set_num_threads(config.cpu_threads)
    set_seed(config.seed)
    dataset_config = get_dataset_config(dataset_name)
    device = resolve_device(device_name)
    output_root = Path(config.output_dir) / dataset_config.name
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Results already exist: {output_root}. Use a fresh --output-dir to avoid overwriting.")
    df = load_event_log(dataset_config, data_dir=data_dir)
    all_activities = df[dataset_config.activity_col].drop_duplicates().tolist()
    cases = df[dataset_config.case_col].drop_duplicates().to_numpy(dtype=object)
    if max_cases is not None:
        # Fixed random case subset rather than the first chronological cases.
        rng = np.random.default_rng(config.split_seed)
        selected = sorted(rng.choice(len(cases), size=min(max_cases, len(cases)), replace=False))
        cases = cases[selected]
        df = df[df[dataset_config.case_col].isin(cases)].copy()
    if len(cases) < config.folds:
        raise ValueError("Fewer cases than requested folds")
    smallest_train = len(cases) - math.ceil(len(cases) / config.folds)
    if smallest_train - math.ceil(smallest_train * config.validation_ratio) < 1:
        raise ValueError("Not enough cases for disjoint train/validation/test splits")
    output_root.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output_root / "experiment.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        run_config = asdict(config)
        run_config["output_dir"] = str(config.output_dir)
        save_json(output_root / "run_config.json", {
            "experiment": run_config, "dataset": asdict(dataset_config), "device": str(device),
            "data_path": str(Path(data_dir) / dataset_config.filename if data_dir else dataset_config.path),
            "max_cases": max_cases, "max_prefixes_per_split": max_prefixes,
            "debug_run": max_cases is not None or max_prefixes is not None,
            "skip_test": config.skip_test,
            "activity_schema": all_activities, "python": platform.python_version(),
            "versions": {package: version(package) for package in ("torch", "numpy", "pandas", "scikit-learn", "tqdm")},
            "command": sys.argv, "cuda_runtime": torch.version.cuda,
        })
        results = []
        kfold = KFold(n_splits=config.folds, shuffle=True, random_state=config.split_seed)
        for fold, (train_indices, test_indices) in enumerate(kfold.split(cases), 1):
            set_seed(config.seed + fold)
            train_cases, val_cases = train_test_split(cases[train_indices],
                test_size=config.validation_ratio, random_state=config.split_seed + fold, shuffle=True)
            split_cases = {"train": train_cases, "validation": val_cases, "test": cases[test_indices]}
            split_frames = {name: df[df[dataset_config.case_col].isin(ids)].copy() for name, ids in split_cases.items()}
            encoder = FeatureEncoder(dataset_config, all_activities, config.time_bins).fit(split_frames["train"])
            datasets = {name: PrefixDataset(encoder.transform_cases(frame), config.min_prefix_len, max_prefixes)
                        for name, frame in split_frames.items()}
            loaders = {
                name: _build_loader(ds, config.batch_size, name == "train", config.num_workers,
                                    pin_memory=device.type == "cuda")
                for name, ds in datasets.items()
                if name != "test" or not config.skip_test
            }
            model_kwargs = {"num_activities": encoder.num_activities,
                            "attribute_vocab_sizes": encoder.attribute_vocab_sizes,
                            "duration_bin_count": encoder.duration_bin_count,
                            "elapsed_bin_count": encoder.elapsed_bin_count,
                            "embedding_dim": config.embedding_dim, "hidden_dim": config.hidden_dim,
                            "graph_layers": config.graph_layers, "dropout": config.dropout,
                            "history_window": config.history_window,
                            "attribute_names": encoder.attribute_cols,
                            "numeric_attribute_names": [
                                name for name in encoder.attribute_cols
                                if name in dataset_config.numeric_attribute_cols
                            ],
                            "initial_alpha": config.initial_alpha,
                            "initial_beta": config.initial_beta,
                            "initial_mu": config.initial_mu,
                            "intensity_scale_init": config.intensity_scale_init,
                            "intensity_scale_max": config.intensity_scale_max,
                            "omega_scale_init": config.omega_scale_init,
                            "omega_scale_max": config.omega_scale_max,
                            "time_scale": float(encoder.time_scale),
                            "ablate_edge_features": config.ablate_edge_features,
                            "ablate_edge_change": config.ablate_edge_change,
                            "ablate_time": config.ablate_time,
                            "model_version": HawkesDYGNN.MODEL_VERSION}
            # Keep compatibility with checkpoints produced by model versions
            # that do not yet expose the bounded-prior constructor arguments.
            supported = inspect.signature(HawkesDYGNN).parameters
            model_kwargs = {key: value for key, value in model_kwargs.items()
                            if key == "model_version" or key in supported}
            model = HawkesDYGNN(**model_kwargs).to(device)
            fold_dir = output_root / f"fold_{fold}"
            fold_dir.mkdir()
            save_json(fold_dir / "preprocessor.json", encoder.to_dict())
            save_json(fold_dir / "model_config.json", model_kwargs)
            save_json(fold_dir / "split_cases.json", {name: list(map(str, ids)) for name, ids in split_cases.items()})
            save_json(fold_dir / "metadata.json", {
                "fold": fold, "seed": config.seed + fold, "split_seed": config.split_seed,
                "attributes": encoder.attribute_cols,
                "case_counts": {name: len(ids) for name, ids in split_cases.items()},
                "prefix_counts": {name: len(ds) for name, ds in datasets.items()},
                "unseen_training_activities": sorted(set(encoder.id_to_activity) - set(split_frames["train"][dataset_config.activity_col])),
                "parameters": sum(p.numel() for p in model.parameters()),
                "selection_metric": config.selection_metric,
                "test_evaluated": not config.skip_test,
            })
            LOGGER.info("%s fold=%d/%d prefixes=%s attributes=%s", dataset_config.name, fold, config.folds,
                        {name: len(ds) for name, ds in datasets.items()}, encoder.attribute_cols)
            metrics = train_fold(loaders["train"], loaders["validation"],
                                 loaders.get("test"), model, config, fold_dir, device)
            metrics["fold"] = fold
            save_json(fold_dir / "metrics.json", metrics)
            results.append(metrics)
            write_summary(output_root / "summary.csv", results)
            LOGGER.info("fold=%d %s metrics=%s", fold,
                        "test" if not config.skip_test else "validation", metrics)
            del model, loaders, datasets
        LOGGER.info("Completed: %s", output_root / "summary.csv")
        return results
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def build_argument_parser(default_dataset=None):
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=default_dataset, choices=list(DATASET_CONFIGS), required=default_dataset is None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    for flag, kind, value in [
        ("folds", int, defaults.folds), ("epochs", int, defaults.epochs),
        ("batch-size", int, defaults.batch_size), ("learning-rate", float, defaults.learning_rate),
        ("patience", int, defaults.patience), ("seed", int, defaults.seed),
        ("split-seed", int, defaults.split_seed),
        ("time-bins", int, defaults.time_bins), ("graph-layers", int, defaults.graph_layers),
        ("history-window", int, defaults.history_window),
        ("hidden-dim", int, defaults.hidden_dim), ("embedding-dim", int, defaults.embedding_dim),
        ("dropout", float, defaults.dropout), ("weight-decay", float, defaults.weight_decay),
        ("num-workers", int, defaults.num_workers), ("cpu-threads", int, defaults.cpu_threads),
        ("initial-alpha", float, defaults.initial_alpha),
        ("initial-beta", float, defaults.initial_beta),
        ("initial-mu", float, defaults.initial_mu),
        ("intensity-scale-init", float, defaults.intensity_scale_init),
        ("intensity-scale-max", float, defaults.intensity_scale_max),
        ("omega-scale-init", float, defaults.omega_scale_init),
        ("omega-scale-max", float, defaults.omega_scale_max),
        ("max-cases", int, None), ("max-prefixes", int, None),
    ]:
        parser.add_argument("--" + flag, type=kind, default=value)
    parser.add_argument("--selection-metric", choices=["validation_" + key for key in METRICS],
                        default=defaults.selection_metric)
    parser.add_argument("--skip-test", action="store_true", default=defaults.skip_test,
                        help="Train and report validation metrics without evaluating the test split")
    parser.add_argument("--ablate-edge-features", action="store_true",
                        default=defaults.ablate_edge_features,
                        help="Zero the hand-crafted edge change descriptors (ablation)")
    parser.add_argument("--ablate-edge-change", action="store_true",
                        default=defaults.ablate_edge_change,
                        help="Remove all change information from the edge: descriptors, "
                             "target-source difference, pair view and Hawkes bias (ablation)")
    parser.add_argument("--ablate-time", action="store_true",
                        default=defaults.ablate_time,
                        help="Remove every temporal signal, keeping all parameters (ablation)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def main(default_dataset=None):
    parser = build_argument_parser(default_dataset)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    fields = set(ExperimentConfig.__dataclass_fields__)
    config = ExperimentConfig(**{name: value for name, value in vars(args).items() if name in fields})
    try:
        run_dataset(args.dataset, config, args.device, args.max_cases, args.max_prefixes, args.data_dir)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
