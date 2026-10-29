"""Prefix-length breakdown of test predictions.

The runner writes ``fold_*/test_predictions.csv`` with one row per test prefix
(``case_id, prefix_length, label_id, prediction_id``).  Because the K folds
partition the cases, pooling the three files covers every case exactly once, so
the buckets below are computed on the whole test population rather than a third
of it.

Prefix length is an *evaluation condition*, not a hyperparameter: retraining
with a different minimum prefix changes both the training distribution and the
task difficulty at once, and the resulting curve cannot be attributed to
either.  Bucketing the predictions costs nothing and answers the same question.

    python scripts/analyze_prefix_length.py --results results/v6_final

Writes ``prefix_length.csv`` next to the per-dataset folders and prints a table.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# Buckets mirror the observed length distributions; the tail is pooled because
# per-length counts fall below ~300 on several logs and become noisy.
BUCKETS = [(2, 2), (3, 3), (4, 5), (6, 7), (8, 10**9)]


def bucket_label(length: int) -> str:
    for low, high in BUCKETS:
        if low <= length <= high:
            return f"{low}" if low == high else f"{low}+" if high > 10**8 else f"{low}-{high}"
    return "?"


def macro_f1(labels, predictions, universe) -> float:
    """Macro F1 over a fixed label universe.

    Fixing the universe keeps the value comparable across buckets: the runner's
    per-split union rule would let a bucket that predicts few classes score
    higher simply by shrinking the average's denominator.
    """

    scores = []
    for label in universe:
        tp = sum(1 for y, p in zip(labels, predictions) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, predictions) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, predictions) if y == label and p != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def collect(dataset_dir: Path):
    rows = []
    for path in sorted(dataset_dir.glob("fold_*/test_predictions.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def analyze(dataset_dir: Path) -> list[dict]:
    rows = collect(dataset_dir)
    if not rows:
        return []
    universe = sorted({int(r["label_id"]) for r in rows} | {int(r["prediction_id"]) for r in rows})
    grouped = defaultdict(lambda: {"labels": [], "predictions": []})
    for row in rows:
        bucket = grouped[bucket_label(int(row["prefix_length"]))]
        bucket["labels"].append(int(row["label_id"]))
        bucket["predictions"].append(int(row["prediction_id"]))
    total = len(rows)
    output = []
    for low, high in BUCKETS:
        name = bucket_label(low)
        if name not in grouped:
            continue
        data = grouped[name]
        n = len(data["labels"])
        correct = sum(1 for y, p in zip(data["labels"], data["predictions"]) if y == p)
        output.append({
            "dataset": dataset_dir.name,
            "bucket": name,
            "prefixes": n,
            "share": n / total,
            "accuracy": correct / n,
            "macro_f1": macro_f1(data["labels"], data["predictions"], universe),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Directory containing one subfolder per dataset")
    args = parser.parse_args()
    records = []
    for dataset_dir in sorted(args.results.iterdir()):
        if dataset_dir.is_dir():
            records.extend(analyze(dataset_dir))
    if not records:
        raise SystemExit(f"No test_predictions.csv found under {args.results}")
    out_path = args.results / "prefix_length.csv"
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    current = None
    for record in records:
        if record["dataset"] != current:
            current = record["dataset"]
            print(f"\n{current}")
            print(f"  {'length':>8s} {'prefixes':>9s} {'share':>7s} {'acc':>7s} {'macroF1':>8s}")
        print(f"  {record['bucket']:>8s} {record['prefixes']:>9d} "
              f"{record['share']:>6.1%} {record['accuracy']:>7.4f} {record['macro_f1']:>8.4f}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
