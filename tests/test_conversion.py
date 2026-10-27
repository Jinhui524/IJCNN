"""Tests for the standard-library XES converter and Helpdesk normalization."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.convert_xes_to_csv import convert_xes_to_csv
from scripts.prepare_datasets import normalize_helpdesk, validate_converted_csv


XES = '''<?xml version="1.0" encoding="UTF-8"?>
<log xmlns="http://www.xes-standard.org/">
  <trace>
    <string key="concept:name" value="case-1"/>
    <string key="customer" value="C1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2020-01-01T00:00:00.000+00:00"/>
      <int key="count" value="1"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <date key="time:timestamp" value="2020-01-01T00:00:00.000+00:00"/>
      <boolean key="flag" value="true"/>
    </event>
  </trace>
</log>
'''


class ConversionTests(unittest.TestCase):
    def test_typed_values_trace_mapping_and_tied_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.xes"
            output = Path(tmp) / "sample.csv"
            source.write_text(XES, encoding="utf-8")
            stats = convert_xes_to_csv(source, output)
            frame = pd.read_csv(output)
            self.assertEqual(stats.cases, 1)
            self.assertEqual(stats.events, 2)
            self.assertEqual(frame["concept:name"].tolist(), ["A", "B"])
            self.assertEqual(frame["case:customer"].tolist(), ["C1", "C1"])
            self.assertEqual(frame["count"].iloc[0], 1)
            self.assertTrue(np.isnan(frame["count"].iloc[1]))
            self.assertEqual(frame["flag"].tolist()[1], True)
            self.assertEqual(len(frame.columns), len(set(frame.columns)))
            validated = validate_converted_csv(source, output, stats)
            self.assertTrue(validated["order_preserved"])

    def test_helpdesk_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "Helpdesk.csv"
            output = Path(tmp) / "Helpdesk_normalized.csv"
            pd.DataFrame([
                {"Case ID": "Case 1", "Activity": "A", "Resource": "R1",
                 "Complete Timestamp": "2012/10/09 14:50:17.000",
                 "customer": "C1", "product": "P1", "seriousness_2": "S1",
                 "service_level": "L1", "service_type": "T1", "support_section": "X",
                 "workgroup": "W"},
            ]).to_csv(source, index=False)
            report = normalize_helpdesk(source, output)
            self.assertEqual(report["cases"], 1)
            self.assertEqual(report["events"], 1)
            self.assertTrue(source.exists())
            self.assertEqual(list(pd.read_csv(output).columns), [
                "case", "concept:name", "org:resource", "time:timestamp", "customer",
                "product", "seriousness_2", "service_level", "service_type",
                "support_section", "workgroup",
            ])


if __name__ == "__main__":
    unittest.main()
