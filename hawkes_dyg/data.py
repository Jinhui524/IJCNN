"""Leakage-safe preprocessing and tensorized prefix graphs.

The dynamic graph for a prefix is a directed chain with self-loops.  Since
that topology is deterministic, the dataset stores compact encoded traces and
the collate function produces padded node/edge tensors instead of materialized
DGL graphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import DatasetConfig


LOGGER = logging.getLogger(__name__)
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
NA_TOKEN = "[NA]"


def load_event_log(config: DatasetConfig, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read, validate, sort and augment one event log.

    Timestamp-derived features use only events within the same case. Invalid
    required fields raise errors: silently dropping events would change targets.
    """

    path = Path(data_dir) / config.filename if data_dir is not None else config.path
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={
        config.case_col: "string", config.activity_col: "string"
    })
    required = [config.case_col, config.activity_col, config.timestamp_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{config.name} is missing required columns: {missing}")

    for column in required:
        if df[column].isna().any():
            raise ValueError(f"{config.name}: missing required values in {column}")
    df[config.timestamp_col] = pd.to_datetime(
        df[config.timestamp_col], utc=True, errors="raise", format="mixed"
    )
    if df.empty:
        raise ValueError(f"{config.name}: empty event log")

    # Mergesort preserves the source order when timestamps tie.
    df["_source_order"] = np.arange(len(df))
    df = df.sort_values(
        [config.case_col, config.timestamp_col, "_source_order"], kind="mergesort"
    ).drop(columns="_source_order").reset_index(drop=True)
    grouped_time = df.groupby(config.case_col, sort=False)[config.timestamp_col]
    df["start_duration"] = (
        df[config.timestamp_col] - grouped_time.transform("min")
    ).dt.total_seconds().fillna(0.0) / 86400.0
    df["duration"] = (
        grouped_time.diff().dt.total_seconds().fillna(0.0) / 86400.0
    )
    df["duration"] = df["duration"].clip(lower=0.0)
    df["start_duration"] = df["start_duration"].clip(lower=0.0)
    return df


def _as_key(value: object) -> str:
    """Convert categorical values to stable strings, including NaN handling."""

    if pd.isna(value):
        return "missing:"
    # Prefix ordinary values so literal '[PAD]'/'[UNK]' never alias reserves.
    return "value:" + str(value)


def _safe_quantile_edges(values: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    unique = np.unique(values)
    if unique.size <= 1:
        return None
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size <= 2:
        return None
    # np.digitize uses the internal edges; the minimum and maximum are not
    # needed as boundaries because values are clipped to valid bucket IDs.
    return edges[1:-1].astype(np.float64)


@dataclass
class EncodedTrace:
    case_id: str
    activity: np.ndarray
    attributes: Dict[str, np.ndarray]
    numeric_attributes: Dict[str, np.ndarray]
    attribute_missing: Dict[str, np.ndarray]
    duration_bin: np.ndarray
    elapsed_bin: np.ndarray
    delta: np.ndarray
    elapsed: np.ndarray

    @property
    def length(self) -> int:
        return int(self.activity.shape[0])


class FeatureEncoder:
    """Fit fold-local categorical and quantile-bucket encoders."""

    def __init__(
        self,
        config: DatasetConfig,
        all_activity_values: Sequence[object],
        n_time_bins: int = 20,
    ) -> None:
        self.config = config
        self.n_time_bins = int(n_time_bins)
        if self.n_time_bins < 2:
            raise ValueError("n_time_bins must be at least 2")
        activity_values = sorted({str(value) for value in all_activity_values})
        self.activity_to_id = {value: idx for idx, value in enumerate(activity_values)}
        self.id_to_activity = activity_values
        self.attribute_to_id: Dict[str, Dict[str, int]] = {}
        self.numeric_edges: Dict[str, Optional[np.ndarray]] = {}
        self.numeric_centers: Dict[str, float] = {}
        self.numeric_scales: Dict[str, float] = {}
        self.time_edges: Dict[str, Optional[np.ndarray]] = {}
        self.attribute_cols: List[str] = []
        self.attribute_vocab_sizes: Dict[str, int] = {}
        self.time_scale = 1.0
        self.fitted = False

    @property
    def num_activities(self) -> int:
        return len(self.id_to_activity)

    @property
    def duration_bin_count(self) -> int:
        return self._edge_bucket_count(self.time_edges.get("duration"))

    @property
    def elapsed_bin_count(self) -> int:
        return self._edge_bucket_count(self.time_edges.get("start_duration"))

    def fit(self, train_df: pd.DataFrame) -> "FeatureEncoder":
        if self.fitted:
            raise RuntimeError("Create a fresh FeatureEncoder for each training fold")
        if train_df.empty:
            raise ValueError("Cannot fit an empty training split")
        for column in self.config.attribute_cols:
            if column not in train_df.columns:
                LOGGER.warning("Skipping missing configured attribute %s", column)
                continue
            non_null = train_df[column].dropna()
            if non_null.empty or non_null.nunique(dropna=True) <= 1:
                LOGGER.info("Skipping empty/constant attribute %s", column)
                continue
            self.attribute_cols.append(column)

            if column in self.config.numeric_attribute_cols:
                values = pd.to_numeric(train_df[column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                self.numeric_edges[column] = _safe_quantile_edges(
                    values, self.n_time_bins
                )
                n_buckets = self._edge_bucket_count(self.numeric_edges[column])
                # 0=PAD, 1=UNK, 2.. are valid numeric buckets.
                self.attribute_vocab_sizes[column] = n_buckets + 2
                finite = values[np.isfinite(values)]
                if finite.size:
                    center = float(np.median(finite))
                    q1, q3 = np.quantile(finite, [0.25, 0.75])
                    scale = max(float(q3 - q1), 1e-8)
                else:
                    center, scale = 0.0, 1.0
                self.numeric_centers[column] = center
                self.numeric_scales[column] = scale
            else:
                values = [_as_key(value) for value in train_df[column].tolist()]
                mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
                for value in sorted(set(values)):
                    if value not in mapping:
                        mapping[value] = len(mapping)
                self.attribute_to_id[column] = mapping
                self.attribute_vocab_sizes[column] = len(mapping)

        for column in ("duration", "start_duration"):
            values = train_df[column].to_numpy(dtype=np.float64)
            self.time_edges[column] = _safe_quantile_edges(
                values, self.n_time_bins
            )

        positive = train_df.loc[train_df["duration"] > 0, "duration"].to_numpy(
            dtype=np.float64
        )
        if positive.size:
            self.time_scale = float(np.median(positive))
        self.time_scale = max(self.time_scale, 1e-8)
        self.fitted = True
        return self

    @staticmethod
    def _edge_bucket_count(edges: Optional[np.ndarray]) -> int:
        return int(edges.size + 1) if edges is not None else 1

    def _transform_numeric(self, values: Sequence[object], column: str) -> np.ndarray:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
            dtype=np.float64
        )
        output = np.full(numeric.shape, 1, dtype=np.int64)  # UNK
        finite = np.isfinite(numeric)
        edges = self.numeric_edges.get(column)
        if edges is None:
            output[finite] = 2
        else:
            output[finite] = np.digitize(numeric[finite], edges, right=False) + 2
        return output

    def _transform_numeric_continuous(
        self, values: Sequence[object], column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(numeric)
        center = self.numeric_centers.get(column, 0.0)
        scale = max(self.numeric_scales.get(column, 1.0), 1e-8)
        standardized = np.zeros(numeric.shape, dtype=np.float32)
        standardized[finite] = ((numeric[finite] - center) / scale).astype(np.float32)
        return standardized, (~finite).astype(np.bool_)

    def _transform_time(self, values: Sequence[object], column: str) -> np.ndarray:
        numeric = np.asarray(values, dtype=np.float64)
        edges = self.time_edges.get(column)
        if edges is None:
            return np.zeros(numeric.shape, dtype=np.int64)
        return np.digitize(numeric, edges, right=False).astype(np.int64)

    def transform_frame(self, frame: pd.DataFrame) -> EncodedTrace:
        if not self.fitted:
            raise RuntimeError("FeatureEncoder.fit must be called before transform")
        activity = np.asarray(
            [self.activity_to_id[str(value)] for value in frame[self.config.activity_col]],
            dtype=np.int64,
        )
        attributes: Dict[str, np.ndarray] = {}
        numeric_attributes: Dict[str, np.ndarray] = {}
        attribute_missing: Dict[str, np.ndarray] = {}
        for column in self.attribute_cols:
            values = frame[column].tolist()
            if column in self.config.numeric_attribute_cols:
                attributes[column] = self._transform_numeric(values, column)
                numeric_attributes[column], attribute_missing[column] = (
                    self._transform_numeric_continuous(values, column)
                )
            else:
                mapping = self.attribute_to_id[column]
                attributes[column] = np.asarray(
                    [mapping.get(_as_key(value), 1) for value in values],
                    dtype=np.int64,
                )
                attribute_missing[column] = pd.isna(frame[column]).to_numpy(dtype=np.bool_)

        duration = frame["duration"].to_numpy(dtype=np.float64)
        elapsed = frame["start_duration"].to_numpy(dtype=np.float64)
        delta = np.log1p(np.maximum(duration, 0.0) / self.time_scale).astype(np.float32)
        # ``elapsed`` stays linear: edge descriptors and the Hawkes kernel take
        # differences of it, and a log would turn ``gap`` into a ratio.  The
        # compression to a workable range happens once, in the node encoder,
        # which is the only consumer that needs it as a scale-sensitive value.
        elapsed = (np.maximum(elapsed, 0.0) / self.time_scale).astype(np.float32)
        case_value = frame[self.config.case_col].iloc[0]
        return EncodedTrace(
            case_id=str(case_value),
            activity=activity,
            attributes=attributes,
            numeric_attributes=numeric_attributes,
            attribute_missing=attribute_missing,
            duration_bin=self._transform_time(duration, "duration"),
            elapsed_bin=self._transform_time(
                frame["start_duration"].to_numpy(dtype=np.float64), "start_duration"
            ),
            delta=delta,
            elapsed=elapsed,
        )

    def transform_cases(self, frame: pd.DataFrame) -> List[EncodedTrace]:
        traces: List[EncodedTrace] = []
        for _, group in frame.groupby(self.config.case_col, sort=False):
            if len(group) >= 2:
                traces.append(self.transform_frame(group))
        return traces

    def to_dict(self) -> dict:
        """JSON-serializable preprocessing required to reload a checkpoint."""
        return {
            "dataset_config": asdict(self.config),
            "n_time_bins": self.n_time_bins,
            "activities": self.id_to_activity,
            "attribute_cols": self.attribute_cols,
            "attribute_to_id": self.attribute_to_id,
            "attribute_vocab_sizes": self.attribute_vocab_sizes,
            "numeric_centers": self.numeric_centers,
            "numeric_scales": self.numeric_scales,
            "numeric_edges": {k: None if v is None else v.tolist() for k, v in self.numeric_edges.items()},
            "time_edges": {k: None if v is None else v.tolist() for k, v in self.time_edges.items()},
            "time_scale_days": self.time_scale,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "FeatureEncoder":
        encoder = cls(DatasetConfig(**state["dataset_config"]), state["activities"], state["n_time_bins"])
        encoder.attribute_cols = state["attribute_cols"]
        encoder.attribute_to_id = state["attribute_to_id"]
        encoder.attribute_vocab_sizes = state["attribute_vocab_sizes"]
        encoder.numeric_centers = {
            k: float(v) for k, v in state.get("numeric_centers", {}).items()
        }
        encoder.numeric_scales = {
            k: float(v) for k, v in state.get("numeric_scales", {}).items()
        }
        for field in ("numeric_edges", "time_edges"):
            setattr(encoder, field, {k: None if v is None else np.asarray(v, dtype=np.float64)
                                     for k, v in state[field].items()})
        encoder.time_scale = state["time_scale_days"]
        encoder.fitted = True
        return encoder


class PrefixDataset(Dataset):
    """Lazily slices encoded traces into prefix/next-activity examples."""

    def __init__(
        self,
        traces: Sequence[EncodedTrace],
        min_prefix_len: int = 2,
        max_prefixes: Optional[int] = None,
    ) -> None:
        self.traces = list(traces)
        if min_prefix_len < 1 or (max_prefixes is not None and max_prefixes < 1):
            raise ValueError("Prefix length/limit must be positive")
        self.min_prefix_len = min_prefix_len
        # O(number of cases) index memory, independent of number of prefixes.
        counts = [max(0, trace.length - min_prefix_len) for trace in self.traces]
        self.cumulative_counts = np.cumsum(counts, dtype=np.int64)
        total = int(self.cumulative_counts[-1]) if counts else 0
        self.size = total if max_prefixes is None else min(total, max_prefixes)

    def __len__(self) -> int:
        return self.size

    def locate(self, index: int):
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        trace_idx = int(np.searchsorted(self.cumulative_counts, index, side="right"))
        preceding = int(self.cumulative_counts[trace_idx - 1]) if trace_idx else 0
        prefix_len = self.min_prefix_len + index - preceding
        return trace_idx, prefix_len

    def __getitem__(self, index: int) -> Dict[str, object]:
        trace_idx, prefix_len = self.locate(index)
        trace = self.traces[trace_idx]
        item: Dict[str, object] = {
            "activity": torch.from_numpy(trace.activity[:prefix_len].copy()),
            "duration_bin": torch.from_numpy(trace.duration_bin[:prefix_len].copy()),
            "elapsed_bin": torch.from_numpy(trace.elapsed_bin[:prefix_len].copy()),
            "delta": torch.from_numpy(trace.delta[:prefix_len].copy()),
            "elapsed": torch.from_numpy(trace.elapsed[:prefix_len].copy()),
            "label": torch.tensor(trace.activity[prefix_len], dtype=torch.long),
            "length": prefix_len,
        }
        item["attributes"] = {
            name: torch.from_numpy(values[:prefix_len].copy())
            for name, values in trace.attributes.items()
        }
        item["numeric_attributes"] = {
            name: torch.from_numpy(values[:prefix_len].copy())
            for name, values in trace.numeric_attributes.items()
        }
        item["attribute_missing"] = {
            name: torch.from_numpy(values[:prefix_len].copy())
            for name, values in trace.attribute_missing.items()
        }
        return item


def collate_prefix_batch(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    activity = pad_sequence(
        [item["activity"] for item in batch], batch_first=True, padding_value=0
    )
    duration_bin = pad_sequence(
        [item["duration_bin"] for item in batch], batch_first=True, padding_value=0
    )
    elapsed_bin = pad_sequence(
        [item["elapsed_bin"] for item in batch], batch_first=True, padding_value=0
    )
    delta = pad_sequence(
        [item["delta"] for item in batch], batch_first=True, padding_value=0.0
    )
    elapsed = pad_sequence(
        [item["elapsed"] for item in batch], batch_first=True, padding_value=0.0
    )

    attribute_names = list(batch[0]["attributes"].keys())
    attributes = {
        name: pad_sequence(
            [item["attributes"][name] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        for name in attribute_names
    }
    numeric_attribute_names = list(batch[0]["numeric_attributes"].keys())
    numeric_attributes = {
        name: pad_sequence(
            [item["numeric_attributes"][name] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        for name in numeric_attribute_names
    }
    attribute_missing_names = list(batch[0]["attribute_missing"].keys())
    attribute_missing = {
        name: pad_sequence(
            [item["attribute_missing"][name] for item in batch],
            batch_first=True,
            padding_value=False,
        )
        for name in attribute_missing_names
    }
    max_len = activity.size(1)
    node_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "activity": activity,
        "attributes": attributes,
        "duration_bin": duration_bin,
        "elapsed_bin": elapsed_bin,
        "delta": delta,
        "elapsed": elapsed,
        "lengths": lengths,
        "node_mask": node_mask,
        "numeric_attributes": numeric_attributes,
        "attribute_missing": attribute_missing,
        "labels": torch.stack([item["label"] for item in batch]),
    }
