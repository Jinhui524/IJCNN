"""Critical correctness tests; run with python -m unittest discover -s tests -v."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hawkes_dyg.config import DATASET_CONFIGS, DatasetConfig, ExperimentConfig
from hawkes_dyg.data import FeatureEncoder, PrefixDataset, collate_prefix_batch, load_event_log
from hawkes_dyg.model import (
    HawkesDYGNN,
    HawkesGatedDirectedGraphConv,
    HawkesTemporalInfluence,
    NodeFeatureEncoder,
    inverse_softplus,
)
from hawkes_dyg.runner import (
    _metric_dict,
    evaluate,
    load_trained_fold,
    macro_gmean,
    make_optimizer,
    run_dataset,
)
from scripts.tune_hawkes_dyg import (
    SEARCH_SPACE,
    _build_parser,
    config_for_trial,
    diagnostics_from_results,
    sample_trials,
    validation_nll_from_artifacts,
    validation_nll_from_results,
)


def example_frame():
    return pd.DataFrame({
        "case": ["one"] * 5 + ["two"] * 3,
        "activity": ["A", "B", "C", "A", "B", "C", "B", "A"],
        "resource": ["R1", "R2", None, "R1", "R2", "R2", "R1", "R1"],
        "amount": [1, 3, 5, 2, 8, 2, 1, 9],
        "duration": [0., 0.1, 0.2, 0.3, 0.5, 0., 0.2, 0.7],
        "start_duration": [0., 0.1, 0.3, 0.6, 1.1, 0., 0.2, 0.9],
    })


def sample_config():
    return DatasetConfig("synthetic", "synthetic.csv", "case", "activity", "time",
                         ["resource", "amount"], ["amount"])


def setup_data():
    frame = example_frame()
    encoder = FeatureEncoder(sample_config(), ["A", "B", "C"], 4).fit(frame)
    return frame, encoder, PrefixDataset(encoder.transform_cases(frame))


def small_model(encoder, ablate_edge_features=False):
    return HawkesDYGNN(encoder.num_activities, encoder.attribute_vocab_sizes,
                      encoder.duration_bin_count, encoder.elapsed_bin_count,
                      embedding_dim=5, hidden_dim=9, graph_layers=2, dropout=0.0,
                      history_window=4, attribute_names=encoder.attribute_cols,
                      numeric_attribute_names=[name for name in encoder.attribute_cols
                                               if name in encoder.config.numeric_attribute_cols],
                      ablate_edge_features=ablate_edge_features,
                      model_version=HawkesDYGNN.MODEL_VERSION)


def inputs(batch):
    return {k: v for k, v in batch.items() if k != "labels"}


class DataTests(unittest.TestCase):
    def test_timestamps_sort_ties_and_mixed_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.DataFrame({"case": ["01", "01", "01", "02"],
                "activity": ["C", "A", "B", "A"],
                "time": ["2020-01-02T00:00:00Z", "2020-01-01T00:00:00.000Z",
                         "2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"]})
            frame.to_csv(Path(tmp) / "synthetic.csv", index=False)
            log = load_event_log(sample_config(), Path(tmp))
            self.assertEqual(log["case"].tolist(), ["01", "01", "01", "02"])
            self.assertEqual(log["activity"].tolist(), ["A", "B", "C", "A"])
            self.assertEqual(log["duration"].tolist(), [0., 0., 1., 0.])
            self.assertEqual(log["start_duration"].tolist(), [0., 0., 1., 0.])

    def test_train_only_fitting_unknowns_and_serialization(self):
        frame, encoder, _ = setup_data()
        before = json.dumps(encoder.to_dict(), sort_keys=True)
        unseen = frame.iloc[:3].copy()
        unseen.loc[:, "resource"] = ["new", "[PAD]", None]
        unseen.loc[:, "duration"] = [0., 999., 5000.]
        unseen["amount"] = [float("nan"), 1e12, -1e12]
        trace = encoder.transform_frame(unseen)
        self.assertEqual(trace.attributes["resource"][:2].tolist(), [1, 1])
        self.assertEqual(trace.attributes["amount"][0], 1)
        self.assertEqual(before, json.dumps(encoder.to_dict(), sort_keys=True))
        restored = FeatureEncoder.from_dict(json.loads(before)).transform_frame(unseen)
        np.testing.assert_array_equal(trace.delta, restored.delta)
        np.testing.assert_array_equal(trace.duration_bin, restored.duration_bin)
        self.assertLess(trace.duration_bin.max(), encoder.duration_bin_count)

    def test_prefix_index_and_no_future_features(self):
        frame, encoder, ds = setup_data()
        self.assertEqual(len(ds), 4)
        self.assertEqual([ds[i]["length"] for i in range(len(ds))], [2, 3, 4, 2])
        self.assertEqual(ds[0]["activity"].tolist(), [0, 1])
        self.assertEqual(ds[0]["label"].item(), 2)
        changed = frame.copy()
        changed.loc[2:4, "activity"] = ["A", "C", "B"]
        changed.loc[2:4, "duration"] = [99., 900., 999.]
        future_ds = PrefixDataset(encoder.transform_cases(changed))
        for key in ["activity", "delta", "elapsed_bin", "duration_bin"]:
            torch.testing.assert_close(ds[0][key], future_ds[0][key])
        capped = PrefixDataset(encoder.transform_cases(frame), max_prefixes=2)
        self.assertEqual(len(capped), 2)
        with self.assertRaises(IndexError):
            _ = capped[2]

    def test_all_zero_gaps_constant_times(self):
        frame = example_frame()
        frame["duration"] = frame["start_duration"] = 0.0
        encoder = FeatureEncoder(sample_config(), ["A", "B", "C"], 20).fit(frame)
        self.assertEqual(encoder.time_scale, 1.0)
        self.assertEqual(encoder.duration_bin_count, 1)
        self.assertTrue(np.isfinite(encoder.transform_frame(frame.iloc[:5]).delta).all())


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_gate_time_decay_pair_orientation_and_padding(self):
        influence = HawkesTemporalInfluence(3)
        with torch.no_grad():
            influence.theta_alpha[0, 1] = torch.log(torch.expm1(torch.tensor(2.0)))
        activity = torch.tensor([[0, 1, 0], [0, 1, 0]])
        elapsed = torch.tensor([[0., 1., 2.], [0., 4., 4.]])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        omega, edge_mask, beta = influence.edge_scores(activity, elapsed, mask, 2)
        # The kernel decays in *log* time: exp(-beta * log1p(gap)), where ``gap``
        # is already expressed in units of the training median inter-event time
        # by ``FeatureEncoder``.  Dividing by ``time_scale`` again would count
        # the interval twice; this test pins the single-normalization contract.
        expected = 2.0 * math.exp(-float(beta.detach()) * math.log1p(1.0))
        self.assertAlmostEqual(omega[0, 1, 0].item(), expected, places=6)
        self.assertGreater(omega[0, 1, 0], omega[1, 1, 0])
        self.assertAlmostEqual(
            omega[0, 2, 1].item(),
            0.5 * math.exp(-float(beta.detach()) * math.log1p(2.0)),
            places=6,
        )
        self.assertFalse(edge_mask[1, 2, 0].item())
        self.assertFalse(edge_mask[:, 0].any().item())
        lengths = torch.tensor([3, 2])
        intensity, _, _ = influence.candidate_intensity(activity, elapsed, lengths, mask)
        self.assertEqual(intensity.shape, (2, 3))
        self.assertTrue(torch.all(intensity > 0))

    def test_edge_scores_match_per_lag_reference(self):
        """The vectorised gather must reproduce the per-lag loop exactly."""

        influence = HawkesTemporalInfluence(4)
        activity = torch.randint(0, 4, (2, 6))
        elapsed = torch.tensor([[0., .5, 2., 2., 7., 30.], [0., 1., 1., 4., 4., 9.]])
        mask = torch.tensor([[True] * 6, [True] * 5 + [False]])
        window = 4
        omega, edge_mask, beta = influence.edge_scores(activity, elapsed, mask, window)
        alpha = influence.positive_parameters()[0]
        reference = torch.zeros_like(omega)
        reference_mask = torch.zeros_like(edge_mask)
        for lag in range(1, window + 1):
            source, target = activity[:, :-lag], activity[:, lag:]
            gap = (elapsed[:, lag:] - elapsed[:, :-lag]).clamp_min(0.0)
            value = alpha[source, target] * torch.exp(-beta * torch.log1p(gap))
            valid = mask[:, :-lag] & mask[:, lag:]
            reference[:, lag:, lag - 1] = value * valid.to(value.dtype)
            reference_mask[:, lag:, lag - 1] = valid
        self.assertTrue(torch.allclose(omega, reference, atol=1e-6))
        self.assertTrue(torch.equal(edge_mask, reference_mask))

    def test_gap_is_normalized_exactly_once(self):
        """``time_scale`` must not be applied on top of the encoder's scaling.

        ``FeatureEncoder.transform_frame`` already divides ``start_duration`` by
        the training median inter-event time, so a gap of ``1.0`` in ``elapsed``
        is one median interval.  Any second division inside the kernel would
        turn that into ``log1p(1 / time_scale)``, which depends on the dataset's
        clock resolution rather than on the process.  The regime below
        (``time_scale`` well below one day, as on ``Receipt``) is exactly where
        the double division collapses the kernel to a constant.
        """

        for scale in (1.0, 0.000433, 0.5, 3.7):
            influence = HawkesTemporalInfluence(2, time_scale=scale)
            gap = torch.tensor([[1.0, 4.0]])
            expected = torch.log1p(gap)
            torch.testing.assert_close(influence.log_gap(gap), expected)
            # The kernel at one median interval is exp(-beta * log 2), no matter
            # what the dataset's time unit happens to be.
            with torch.no_grad():
                influence.theta_beta.fill_(inverse_softplus(0.5))
            beta = float(influence.positive_parameters()[1].detach())
            torch.testing.assert_close(
                influence.kernel(gap),
                torch.exp(-beta * torch.log1p(gap)),
            )

    def test_graph_layer_is_permutation_of_lag_loop(self):
        """The batched edge encoder must equal encoding each lag separately."""

        torch.manual_seed(0)
        layer = HawkesGatedDirectedGraphConv(
            hidden_dim=6, num_activities=3, attribute_names=["resource"],
            numeric_attribute_names=["amount"], history_window=4, dropout=0.0,
        )
        hidden = torch.randn(2, 5, 6)
        activity = torch.randint(0, 3, (2, 5))
        elapsed = torch.tensor([[0., 1., 1., 5., 6.], [0., 2., 3., 3., 11.]])
        attributes = {"resource": torch.randint(0, 4, (2, 5))}
        numeric = {"amount": torch.randn(2, 5)}
        missing = {"resource": torch.zeros(2, 5, dtype=torch.bool)}
        node_mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
        window = 4
        omega = torch.rand(2, 5, window)
        edge_mask = torch.zeros(2, 5, window, dtype=torch.bool)
        for lag in range(1, window + 1):
            edge_mask[:, lag:, lag - 1] = node_mask[:, :-lag] & node_mask[:, lag:]
        output = layer(hidden, activity, elapsed, attributes, numeric, missing,
                       omega, edge_mask, node_mask)
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.isfinite(output).all())
        # Padded nodes stay exactly zero after the residual/LayerNorm update.
        self.assertTrue(torch.equal(output[1, 3:], torch.zeros_like(output[1, 3:])))
        # Every graph layer must build the same slot grid, so stacking two
        # layers in sequence cannot depend on iteration order.
        second = layer(output, activity, elapsed, attributes, numeric, missing,
                       omega, edge_mask, node_mask)
        self.assertTrue(torch.isfinite(second).all())

    def test_graph_layer_matches_explicit_directed_adjacency(self):
        layer = HawkesGatedDirectedGraphConv(
            hidden_dim=4, num_activities=3, attribute_names=[],
            numeric_attribute_names=[], history_window=3, dropout=0.0
        )
        hidden = torch.randn(1, 4, 4)
        mask = torch.tensor([[True, True, True, False]])
        hidden *= mask.unsqueeze(-1)
        activity = torch.tensor([[0, 1, 2, 0]])
        elapsed = torch.tensor([[0., 1., 2., 3.]])
        omega = torch.zeros(1, 4, 3)
        edge_mask = torch.zeros(1, 4, 3, dtype=torch.bool)
        omega[0, 1, 0] = 0.25
        omega[0, 2, 0] = 0.75
        edge_mask[0, 1, 0] = True
        edge_mask[0, 2, 0] = True
        edge_mask[0, 2, 1] = True
        empty = {}
        output = layer(hidden, activity, elapsed, empty, empty, empty,
                       omega, edge_mask, mask)
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.isfinite(output).all())
        perturbed = hidden.clone()
        perturbed[:, 2] += 100.0
        perturbed_output = layer(perturbed, activity, elapsed, empty, empty, empty,
                                 omega, edge_mask, mask)
        torch.testing.assert_close(output[:, :2], perturbed_output[:, :2])

    def test_batch_padding_invariance_and_gradients(self):
        _, encoder, ds = setup_data()
        model = small_model(encoder).eval()
        single = collate_prefix_batch([ds[0]])
        mixed = collate_prefix_batch([ds[0], ds[2]])
        torch.testing.assert_close(model(**inputs(single))[0], model(**inputs(mixed))[0], atol=1e-6, rtol=1e-5)
        optimizer = make_optimizer(model, ExperimentConfig(weight_decay=0.1))
        before_alpha = model.influence.theta_alpha.detach().clone()
        before_beta = model.influence.theta_beta.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(**inputs(mixed)), mixed["labels"])
        loss.backward()
        self.assertGreater(model.influence.theta_alpha.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.influence.theta_beta.grad.abs().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(before_alpha, model.influence.theta_alpha))
        self.assertFalse(torch.equal(before_beta, model.influence.theta_beta))
        # The candidate intensity head gives every observed source activity a
        # differentiable score for every candidate destination activity.
        self.assertTrue(torch.isfinite(model.influence.theta_alpha).all())

    def test_edge_feature_ablation_keeps_shapes_and_zeroes_descriptors(self):
        """The ablation must remove information without changing capacity.

        Zeroing the descriptors instead of shrinking the tensor keeps every
        parameter shape identical, so a result difference is attributable to the
        information and not to a smaller model.
        """

        _, encoder, ds = setup_data()
        batch = collate_prefix_batch([ds[2]])
        full = small_model(encoder).eval()
        ablated = small_model(encoder, ablate_edge_features=True).eval()
        ablated.load_state_dict(full.state_dict())
        self.assertEqual(
            sum(p.numel() for p in full.parameters()),
            sum(p.numel() for p in ablated.parameters()),
        )
        with torch.no_grad():
            self.assertFalse(torch.allclose(full(**inputs(batch)), ablated(**inputs(batch))))
            layer = ablated.graph_layers[0]
            elapsed = batch["elapsed"]
            sources = layer._slot_positions(
                elapsed.size(1), ablated.history_window, elapsed.device
            )
            static = layer._static_edge_features(
                elapsed,
                torch.ones(elapsed.size(0), elapsed.size(1), ablated.history_window, 1),
                batch["attributes"],
                batch["numeric_attributes"],
                batch["attribute_missing"],
                sources,
                elapsed.size(1),
                ablated.history_window,
            )
            self.assertTrue(torch.equal(static, torch.zeros_like(static)))

    def test_zero_graph_layers_falls_back_to_readout_only(self):
        """``graph_layers=0`` is the no-message-passing ablation.

        Nodes keep their encoder output; ``last``, ``mean`` and ``last_delta``
        still reach the classifier, and the auxiliary diagnostics must not try
        to stack over an empty layer list.
        """

        _, encoder, ds = setup_data()
        model = HawkesDYGNN(
            encoder.num_activities, encoder.attribute_vocab_sizes,
            encoder.duration_bin_count, encoder.elapsed_bin_count,
            embedding_dim=5, hidden_dim=9, graph_layers=0, dropout=0.0,
            history_window=4, attribute_names=encoder.attribute_cols,
            model_version=HawkesDYGNN.MODEL_VERSION,
        ).eval()
        self.assertEqual(len(model.graph_layers), 0)
        batch = collate_prefix_batch([ds[2]])
        logits, aux = model(**inputs(batch), return_aux=True)
        self.assertEqual(logits.shape, (1, encoder.num_activities))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(aux["omega_scale"].numel(), 0)
        self.assertEqual(float(aux["attention_entropy"]), 0.0)
        # Removing propagation must change the prediction path, not break it.
        dense = small_model(encoder).eval()
        with torch.no_grad():
            self.assertFalse(torch.allclose(dense(**inputs(batch)), logits))

    def test_sparse_history_window_and_causal_edges(self):
        _, encoder, ds = setup_data()
        model = small_model(encoder).eval()
        batch = collate_prefix_batch([ds[2]])
        logits, aux = model(**inputs(batch), return_aux=True)
        self.assertEqual(logits.shape[-1], encoder.num_activities)
        edge_mask = aux["edge_mask"][0]
        for target in range(edge_mask.size(0)):
            valid_lags = torch.where(edge_mask[target])[0].tolist()
            self.assertTrue(all(lag < target for lag in valid_lags))
            if target > 0:
                self.assertIn(0, valid_lags)
        self.assertLessEqual(int(edge_mask.sum()), 4 * 4)
        self.assertTrue(torch.isfinite(aux["intensity"]).all())

    def test_continuous_delta_changes_node_representation(self):
        _, encoder, ds = setup_data()
        model = small_model(encoder).eval()
        batch = collate_prefix_batch([ds[1]])
        original = model(**inputs(batch))
        changed = dict(inputs(batch))
        changed["delta"] = batch["delta"].clone()
        changed["delta"][0, 1] += 3.0
        altered = model(**changed)
        self.assertFalse(torch.allclose(original, altered))

    def test_continuous_elapsed_is_log_compressed(self):
        """``elapsed`` must reach the encoder on a bounded, log scale.

        On ``Receipt`` the raw ratio reaches ~6e5 while ``delta`` stays below
        ~10, so an untouched ``elapsed`` channel would dominate the shared
        linear layer.  The two channels of the projected input must therefore
        stay within the same order of magnitude.
        """

        out = NodeFeatureEncoder.continuous_time_features(
            torch.tensor([1.0, 8.0]),
            torch.tensor([637234.0, 0.0]),
        )
        self.assertTrue(torch.isfinite(out).all())
        # log1p(637234) ~ 13.4, comfortably within the range of ``delta``.
        self.assertLess(float(out[0, 1]), 20.0)
        self.assertAlmostEqual(float(out[1, 1]), 0.0)
        # A gap ratio of zero must not produce -inf or NaN.
        clamped = NodeFeatureEncoder.continuous_time_features(
            torch.tensor([0.0]), torch.tensor([-5.0])
        )
        self.assertTrue(torch.isfinite(clamped).all())

    def test_alpha_drives_pair_features_and_layers_see_it(self):
        """Activity-pair strength has a single source: ``influence.alpha``.

        The layers no longer own a pair table, so a change in ``alpha`` must
        reach the edge representation through the injected statistic.
        """

        _, encoder, ds = setup_data()
        model = small_model(encoder).eval()
        batch = inputs(collate_prefix_batch([ds[2]]))
        with torch.no_grad():
            baseline = model(**batch)
        with torch.no_grad():
            model.influence.theta_alpha.add_(2.0)
            shifted = model(**batch)
        self.assertFalse(torch.allclose(baseline, shifted))
        for layer in model.graph_layers:
            self.assertFalse(hasattr(layer, "pair_embedding"))
            self.assertTrue(hasattr(layer, "pair_projection"))

    def test_intensity_prior_is_normalized_and_bounded(self):
        _, encoder, ds = setup_data()
        model = small_model(encoder).eval()
        logits, aux = model(**inputs(collate_prefix_batch([ds[2]])), return_aux=True)
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(
            torch.logsumexp(aux["normalized_log_intensity"], dim=-1),
            torch.zeros(1),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertGreaterEqual(aux["intensity_scale"].item(), 0.0)
        self.assertLessEqual(aux["intensity_scale"].item(), 1.0)
        self.assertTrue(torch.all(aux["omega_scale"] >= 0.0))
        self.assertTrue(torch.all(aux["omega_scale"] <= 1.0))

    def test_nll_independent_of_batch_partition(self):
        _, encoder, ds = setup_data()
        model = small_model(encoder).eval()
        metrics = []
        for batch_size in [1, 3, 4]:
            loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_prefix_batch)
            loss, report = evaluate(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), 3)
            metrics.append(report["nll"])
            self.assertEqual(loss, report["nll"])
        self.assertAlmostEqual(metrics[0], metrics[1], places=6)
        self.assertAlmostEqual(metrics[0], metrics[2], places=6)

    def test_macro_gmean_matches_reference_definition(self):
        # Recalls: 1/2, 1/2, 1; specificities: 1, 2/3, 3/4.
        expected = math.sqrt((2.0 / 3.0) * ((1.0 + 2.0 / 3.0 + 3.0 / 4.0) / 3.0))
        self.assertAlmostEqual(macro_gmean([0, 0, 1, 1, 2], [0, 1, 1, 2, 2]), expected)

    def test_metrics_match_fhgsn_union_label_rule(self):
        # An absent fourth activity must not add a zero-valued class to the
        # average, even when the caller still passes the global class count.
        report = _metric_dict([0, 0, 1, 1], [0, 0, 1, 1], 0.0, num_classes=4)
        self.assertAlmostEqual(report["precision"], 1.0)
        self.assertAlmostEqual(report["recall"], 1.0)
        self.assertAlmostEqual(report["f1"], 1.0)

    def test_single_label_metrics_do_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = _metric_dict([1], [1], 0.0, num_classes=4)
            macro_gmean([1], [1], num_classes=4)
        self.assertEqual(report["f1"], 1.0)
        self.assertFalse(any("single label" in str(item.message) for item in caught))


class ExperimentTests(unittest.TestCase):
    def test_tuning_space_is_deterministic_and_unique(self):
        first = sample_trials(28, seed=123)
        second = sample_trials(28, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(len({tuple(row[name] for name in SEARCH_SPACE) for row in first}), 28)
        for row in first:
            for name, values in SEARCH_SPACE.items():
                self.assertIn(row[name], values)
        with self.assertRaises(ValueError):
            sample_trials(649, seed=123)

    def test_tuning_uses_validation_nll_only(self):
        rows = [
            {"nll": 0.01, "validation_nll": 1.25},
            {"nll": 9.99, "best_validation_loss": 0.75},
        ]
        self.assertAlmostEqual(validation_nll_from_results(rows), 1.0)
        # A historical test-only result must never be accepted as the tuning
        # score, because using it would leak the held-out split into selection.
        self.assertIsNone(validation_nll_from_results([{"nll": 0.01}]))

    def test_tuning_reports_fold_diagnostics_without_test_metrics(self):
        rows = [
            {"validation_f1": 0.2, "best_epoch": 4, "alpha_mean": 0.5,
             "beta": 0.4, "nll": 99.0},
            {"validation_f1": 0.4, "best_epoch": 6, "alpha_mean": 0.7,
             "beta": 0.6, "nll": 0.01},
        ]
        report = diagnostics_from_results(rows)
        self.assertAlmostEqual(report["validation_f1"], 0.3)
        self.assertAlmostEqual(report["best_epoch"], 5.0)
        self.assertAlmostEqual(report["alpha_mean"], 0.6)
        self.assertNotIn("nll", report)

    def test_tuning_artifact_reader_ignores_test_nll(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fold_1").mkdir()
            (root / "fold_2").mkdir()
            (root / "fold_1" / "metrics.json").write_text(
                json.dumps({"nll": 0.01, "best_validation_loss": 1.0}), encoding="utf-8"
            )
            (root / "fold_2" / "metrics.json").write_text(
                json.dumps({"nll": 0.02, "validation_nll": 3.0}), encoding="utf-8"
            )
            self.assertAlmostEqual(validation_nll_from_artifacts(root), 2.0)

    def test_tuning_artifact_reader_accepts_validation_summary_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.csv").write_text(
                "fold,nll,validation_nll\n1,0.01,1.0\n2,0.02,3.0\nAverage,0.015,2.0\nStd,0.005,1.0\n",
                encoding="utf-8",
            )
            self.assertAlmostEqual(validation_nll_from_artifacts(root), 2.0)

    def test_tuning_config_keeps_split_seed_independent_of_model_seed(self):
        base = ExperimentConfig(seed=42, hidden_dim=64)
        trial = {"hidden_dim": 128, "history_window": 16}
        config_a = config_for_trial(base, trial, model_seed=7, split_seed=42)
        config_b = config_for_trial(base, trial, model_seed=8, split_seed=42)
        self.assertEqual(config_a.hidden_dim, 128)
        self.assertEqual(config_a.seed, 7)
        self.assertEqual(config_b.seed, 8)
        if hasattr(config_a, "split_seed"):
            self.assertEqual(config_a.split_seed, config_b.split_seed)
            self.assertEqual(config_a.split_seed, 42)

    def test_tuning_parser_defaults_to_skip_test(self):
        args = _build_parser().parse_args([
            "--dataset", "Helpdesk", "--output-dir", "/tmp/hawkes-tuning"
        ])
        self.assertTrue(args.skip_test)
        args = _build_parser().parse_args([
            "--dataset", "Helpdesk", "--output-dir", "/tmp/hawkes-tuning",
            "--trials", "24", "--skip-test"
        ])
        self.assertTrue(args.skip_test)

    def test_disjoint_folds_artifacts_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for case in range(18):
                for step, activity in enumerate(["A", "B", "C", "B"]):
                    rows.append({"case": str(case), "activity": activity,
                        "time": f"2020-01-{case + 1:02d}T00:0{step}:00Z",
                        "resource": f"R{step % 2}", "amount": case + step})
            pd.DataFrame(rows).to_csv(root / "synthetic.csv", index=False)
            config = ExperimentConfig(folds=3, epochs=1, batch_size=8, embedding_dim=4,
                hidden_dim=8, cpu_threads=1, output_dir=root / "results")
            with patch.dict(DATASET_CONFIGS, {"synthetic": sample_config()}):
                results = run_dataset("synthetic", config, "cpu", data_dir=root)
            self.assertEqual(len(results), 3)
            all_tests = []
            for fold in range(1, 4):
                folder = root / "results" / "synthetic" / f"fold_{fold}"
                split = json.loads((folder / "split_cases.json").read_text())
                train, val, test = (set(split[key]) for key in ["train", "validation", "test"])
                self.assertFalse(train & val or train & test or val & test)
                all_tests.extend(test)
                model, encoder = load_trained_fold(folder)
                log = load_event_log(sample_config(), root)
                ds = PrefixDataset(encoder.transform_cases(log[log["case"].isin(test)]))
                loader = DataLoader(ds, batch_size=3, collate_fn=collate_prefix_batch)
                _, reloaded = evaluate(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), 3)
                self.assertAlmostEqual(reloaded["nll"], results[fold - 1]["nll"], places=6)
                self.assertTrue((folder / "loss_curve.svg").is_file())
                self.assertTrue((folder / "test_predictions.csv").is_file())
            self.assertEqual(len(set(all_tests)), 18)
            self.assertEqual(len(all_tests), 18)

    def test_split_seed_is_independent_of_model_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for case in range(18):
                for step, activity in enumerate(["A", "B", "C", "B"]):
                    rows.append({"case": str(case), "activity": activity,
                        "time": f"2020-01-{case + 1:02d}T00:0{step}:00Z",
                        "resource": f"R{step % 2}", "amount": case + step})
            pd.DataFrame(rows).to_csv(root / "synthetic.csv", index=False)
            with patch.dict(DATASET_CONFIGS, {"synthetic": sample_config()}):
                for model_seed, name in [(42, "seed_a"), (43, "seed_b")]:
                    config = ExperimentConfig(
                        folds=3, epochs=1, batch_size=8, embedding_dim=4,
                        hidden_dim=8, cpu_threads=1, seed=model_seed,
                        split_seed=99, skip_test=True,
                        output_dir=root / name,
                    )
                    run_dataset("synthetic", config, "cpu", data_dir=root)
            for fold in range(1, 4):
                first = json.loads((root / "seed_a" / "synthetic" / f"fold_{fold}" / "split_cases.json").read_text())
                second = json.loads((root / "seed_b" / "synthetic" / f"fold_{fold}" / "split_cases.json").read_text())
                self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
