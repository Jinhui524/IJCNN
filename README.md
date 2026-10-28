# Hawkes-DyG

Change-aware next-activity prediction over sparse causal history graphs.

## Repository layout

```
hawkes_dyg/          # the method
  config.py          # dataset schemas + experiment hyperparameters
  data.py            # leakage-safe encoding, prefix indexing, collation
  model.py           # node encoder, Hawkes influence, edge-conditioned layer
  runner.py          # case-level K-fold protocol, training, evaluation, CLI
scripts/             # entry points and one-off tooling
  prepare_datasets.py, convert_xes_to_csv.py   # XES -> canonical CSV
  tune_hawkes_dyg.py                            # validation-only random search
  bench_hawkes_dyg.py                           # step-time / variants benchmark
  run_*.py                                      # thin per-dataset wrappers
tests/               # unit + protocol tests
dataset/             # event logs (not versioned; see below)
```

`results/`, `work/`, `tuning/`, `_archive/` are local-only and gitignored.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_hawkes_dyg.txt
python -m unittest discover -s tests -v
```

Python 3.10+ is required (`str.removeprefix`, `X | None`, `dataclass` slots use).
The code has only been exercised on 3.12; the tests are the check.

## Data

`dataset/` holds one canonical CSV per log, with `case`, `concept:name`,
`time:timestamp` and the attributes named in `hawkes_dyg/config.py`. Files are
not committed; regenerate them from the original XES with:

```bash
python scripts/convert_xes_to_csv.py --input <log.xes> --output dataset/<Name>.csv
python scripts/prepare_datasets.py            # normalizes Helpdesk
```

## Running

```bash
# single dataset, full protocol
python -m hawkes_dyg.runner --dataset BPIC_2020_D --output-dir results --device cuda

# or through a wrapper
python scripts/run_helpdesk.py --output-dir results

# fast smoke test on a subset
python -m hawkes_dyg.runner --dataset Helpdesk --max-cases 200 --max-prefixes 1000 \
    --epochs 2 --output-dir /tmp/smoke
```

Each run writes `results/<dataset>/fold_k/{metrics.json,preprocessor.json,
model_config.json,split_cases.json,best_model.pt,history.csv}` and a
`summary.csv` with per-fold rows plus `Average` and `Std`.

### Protocol

Case-level 3-fold cross-validation (`split_seed` fixed), a 15% validation slice
carved out of each training fold for early stopping, and a disjoint test fold.
Preprocessing is fit on the training fold only; activities are enumerated from
the whole log so the label space is stable across folds. Selection defaults to
validation NLL over early stopping with patience 10.

## Notes on the current results

`summary.csv` from the 8-dataset comparison is in the paper workspace; the
headline caveats are:

- Accuracy is dominated by frequent transitions. A first-order Markov chain
  reaches 0.9005 on BPIC_2020_D against 0.9046 for this model, so macro-F1 is
  the discriminating metric, not accuracy.
- `macro_f1` in the baseline harness is computed over the union of true and
  predicted labels, which rewards models that abandon rare classes. Compare on a
  fixed label universe before quoting it in the paper.
- `completed_repeats` is 1. Gaps of 0.001–0.005 between methods on the 2020 logs
  are inside seed-to-seed variation; use >= 5 repeats and a paired test.

## License

Not yet chosen.
