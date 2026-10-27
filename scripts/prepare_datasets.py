"""Convert and validate the additional local process datasets.

The command is deliberately conservative: source XES files are removed only
when every conversion and read-back validation succeeds.  Helpdesk is written
to a new normalized file and its original CSV is retained.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

try:
    from convert_xes_to_csv import ConversionStats, convert_xes_to_csv, iter_trace_rows
except ImportError:  # Support ``python -m scripts.prepare_datasets``.
    from .convert_xes_to_csv import ConversionStats, convert_xes_to_csv, iter_trace_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

XES_FILES = [
    "BPIC_2020_D.xes",
    "BPIC_2020_I.xes",
    "BPIC_2020_Pe.xes",
    "BPIC_2020_Pr.xes",
    "BPIC_2020_R.xes",
    "Receipt.xes",
]

EXPECTED_COUNTS = {
    "BPIC_2020_D": {"cases": 10500, "events": 56437},
    "BPIC_2020_I": {"cases": 6449, "events": 72151},
    "BPIC_2020_Pe": {"cases": 7065, "events": 86581},
    "BPIC_2020_Pr": {"cases": 2099, "events": 18246},
    "BPIC_2020_R": {"cases": 6886, "events": 36796},
    "Receipt": {"cases": 1434, "events": 8577},
}


def _assert_unique_columns(columns: Sequence[str], path: Path) -> None:
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"{path} has duplicate columns: {duplicates}")


def validate_converted_csv(source: Path, output: Path, stats: ConversionStats) -> dict:
    """Validate schema, counts, timestamps and within-case event order."""

    frame = pd.read_csv(output, encoding="utf-8-sig")
    _assert_unique_columns(list(frame.columns), output)
    required = {"case", "concept:name", "time:timestamp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{output} is missing required columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{output} contains missing required values")
    pd.to_datetime(frame["time:timestamp"], utc=True, errors="raise", format="mixed")

    actual_cases = int(frame["case"].nunique(dropna=False))
    actual_events = int(len(frame))
    actual_activities = sorted(frame["concept:name"].astype(str).unique().tolist())
    if actual_cases != stats.cases or actual_events != stats.events:
        raise ValueError(
            f"{output} count mismatch: XES cases/events={stats.cases}/{stats.events}, "
            f"CSV={actual_cases}/{actual_events}"
        )
    if actual_activities != stats.activities:
        raise ValueError(f"{output} activity set differs from XES source")

    expected_sequences: Dict[str, List[str]] = {}
    for case_id, rows in iter_trace_rows(source):
        expected_sequences[str(case_id)] = [row["concept:name"] for row in rows]
    actual_sequences: Dict[str, List[str]] = {}
    for case_id, group in frame.groupby("case", sort=False):
        actual_sequences[str(case_id)] = group["concept:name"].astype(str).tolist()
    if expected_sequences != actual_sequences:
        raise ValueError(f"{output} does not preserve XES event order")

    return {
        "cases": actual_cases,
        "events": actual_events,
        "activities": actual_activities,
        "columns": list(frame.columns),
        "timestamps_valid": True,
        "order_preserved": True,
    }


def normalize_helpdesk(source: Path, output: Path) -> dict:
    """Write a canonical Helpdesk CSV while preserving its raw source."""

    frame = pd.read_csv(source, encoding="utf-8-sig")
    mapping = {
        "Case ID": "case",
        "Activity": "concept:name",
        "Resource": "org:resource",
        "Complete Timestamp": "time:timestamp",
        "customer": "customer",
        "product": "product",
        "seriousness_2": "seriousness_2",
        "service_level": "service_level",
        "service_type": "service_type",
        "support_section": "support_section",
        "workgroup": "workgroup",
    }
    missing = sorted(set(mapping) - set(frame.columns))
    if missing:
        raise ValueError(f"Helpdesk source is missing columns: {missing}")
    normalized = frame[list(mapping)].rename(columns=mapping)
    _assert_unique_columns(list(normalized.columns), output)
    if normalized[["case", "concept:name", "time:timestamp"]].isna().any().any():
        raise ValueError("Helpdesk normalized output contains missing required values")
    pd.to_datetime(normalized["time:timestamp"], utc=True, errors="raise", format="mixed")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    normalized.to_csv(temp, index=False, encoding="utf-8")
    temp.replace(output)
    return {
        "source": str(source),
        "output": str(output),
        "cases": int(normalized["case"].nunique()),
        "events": int(len(normalized)),
        "activities": sorted(normalized["concept:name"].astype(str).unique().tolist()),
        "columns": list(normalized.columns),
        "timestamps_valid": True,
    }


def prepare(data_dir: Path, delete_xes: bool = False, report_path: Path | None = None) -> dict:
    data_dir = Path(data_dir)
    report: dict = {"xes": {}, "helpdesk": {}, "deleted_sources": []}
    converted: List[Path] = []
    try:
        for filename in XES_FILES:
            source = data_dir / filename
            name = source.stem
            output = data_dir / f"{name}.csv"
            stats = convert_xes_to_csv(source, output)
            if name in EXPECTED_COUNTS:
                expected = EXPECTED_COUNTS[name]
                if stats.cases != expected["cases"] or stats.events != expected["events"]:
                    raise ValueError(
                        f"{source} has unexpected counts {stats.cases}/{stats.events}; "
                        f"expected {expected['cases']}/{expected['events']}"
                    )
            validation = validate_converted_csv(source, output, stats)
            report["xes"][name] = {"conversion": stats.to_dict(), "validation": validation}
            converted.append(source)

        helpdesk_source = data_dir / "Helpdesk.csv"
        helpdesk_output = data_dir / "Helpdesk_normalized.csv"
        report["helpdesk"] = normalize_helpdesk(helpdesk_source, helpdesk_output)
    except Exception:
        # Do not remove any XES if any conversion or validation fails.
        raise

    if delete_xes:
        for source in converted:
            source.unlink()
            report["deleted_sources"].append(str(source))
    if report_path is None:
        report_path = data_dir / "dataset_conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--delete-xes", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = prepare(args.data_dir, delete_xes=args.delete_xes, report_path=args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
