"""Streaming XES to canonical event-log CSV conversion.

The converter intentionally uses only the Python standard library.  It keeps
one trace in memory at a time, preserves the order of events in the XES file,
and emits a canonical ``case``/``concept:name``/``time:timestamp`` schema.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


REQUIRED_EVENT_KEYS = ("concept:name", "time:timestamp")
TYPE_TAGS = {"string", "date", "int", "float", "boolean", "id", "list", "container"}


def local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""

    return tag.rsplit("}", 1)[-1]


def typed_value(element: ET.Element) -> str:
    """Read an XES scalar attribute as a stable CSV string."""

    kind = local_name(element.tag)
    if kind not in TYPE_TAGS:
        raise ValueError(f"Unsupported XES attribute element: {element.tag}")
    if kind in {"list", "container"}:
        raise ValueError("Nested XES list/container attributes are not supported")
    return element.attrib.get("value", "")


def scalar_attributes(element: ET.Element) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for child in list(element):
        key = child.attrib.get("key")
        if not key:
            continue
        if key in values:
            raise ValueError(f"Duplicate XES attribute key {key!r} in {local_name(element.tag)}")
        values[key] = typed_value(child)
    return values


@dataclass
class ConversionStats:
    source: str
    output: str
    cases: int = 0
    events: int = 0
    activities: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def iter_trace_rows(path: Path) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
    """Yield ``(case_id, rows)`` while retaining only one trace in memory."""

    current_trace: Optional[Dict[str, str]] = None
    current_events: Optional[List[Dict[str, str]]] = None
    try:
        context = ET.iterparse(path, events=("start", "end"))
        for event, element in context:
            name = local_name(element.tag)
            if event == "start" and name == "trace":
                if current_trace is not None:
                    raise ValueError(f"Nested trace encountered in {path}")
                current_trace = {}
                current_events = []
                continue
            if event == "end" and name == "event":
                if current_trace is None or current_events is None:
                    raise ValueError(f"Event outside trace in {path}")
                current_events.append(scalar_attributes(element))
                element.clear()
                continue
            if event == "end" and name == "trace":
                if current_trace is None or current_events is None:
                    raise ValueError(f"Unexpected trace end in {path}")
                current_trace.update(scalar_attributes(element))
                case_id = current_trace.get("concept:name")
                if case_id is None or case_id == "":
                    raise ValueError(f"Trace without concept:name in {path}")
                rows: List[Dict[str, str]] = []
                for event_values in current_events:
                    missing = [key for key in REQUIRED_EVENT_KEYS if not event_values.get(key)]
                    if missing:
                        raise ValueError(
                            f"Event in case {case_id!r} lacks required XES fields: {missing}"
                        )
                    row = {"case": case_id}
                    for key, value in current_trace.items():
                        if key != "concept:name":
                            row[f"case:{key}"] = value
                    row.update(event_values)
                    rows.append(row)
                yield case_id, rows
                current_trace = None
                current_events = None
                element.clear()
    except ET.ParseError as error:
        raise ValueError(f"Invalid XES XML in {path}: {error}") from error


def discover_columns(path: Path) -> Tuple[List[str], ConversionStats]:
    """First pass: collect the union of scalar output columns and statistics."""

    columns: List[str] = ["case", *REQUIRED_EVENT_KEYS]
    seen = set(columns)
    activities = set()
    stats = ConversionStats(source=str(path), output="")
    for _, rows in iter_trace_rows(path):
        stats.cases += 1
        stats.events += len(rows)
        for row in rows:
            activity = row["concept:name"]
            activities.add(activity)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    stats.activities = sorted(activities)
    stats.columns = columns
    return columns, stats


def convert_xes_to_csv(source: Path, output: Path) -> ConversionStats:
    """Convert one XES file to CSV using two streaming passes."""

    source = Path(source)
    output = Path(output)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("XES source and CSV output must be different paths")
    columns, stats = discover_columns(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.name + ".tmp")
    try:
        with temp_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            written_cases = 0
            written_events = 0
            for _, rows in iter_trace_rows(source):
                written_cases += 1
                written_events += len(rows)
                writer.writerows(rows)
        if written_cases != stats.cases or written_events != stats.events:
            raise RuntimeError("XES changed during conversion; case/event counts differ between passes")
        temp_output.replace(output)
    finally:
        if temp_output.exists():
            temp_output.unlink()
    stats.output = str(output)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()
    stats = convert_xes_to_csv(args.source, args.output)
    payload = stats.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.stats:
        args.stats.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
