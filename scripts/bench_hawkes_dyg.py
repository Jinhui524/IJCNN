"""Measure step time and GPU occupancy for the causal history model.

Run this on the training machine before and after a throughput change:

    python -m scripts.bench_hawkes_dyg --device cuda:0 --batch-size 32
    python -m scripts.bench_hawkes_dyg --device cuda:0 --batch-size 256
    python -m scripts.bench_hawkes_dyg --device cuda:0 --compare-lag-loop

``--compare-lag-loop`` runs a reference implementation of the previous
per-lag Python loop next to the vectorised layer, so the speed-up is measured
rather than assumed.  Both variants receive identical inputs.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from hawkes_dyg.model import HawkesDYGNN, HawkesGatedDirectedGraphConv


def make_batch(batch_size: int, sequence_length: int, num_activities: int,
               attribute_names, device) -> Dict[str, object]:
    torch.manual_seed(0)
    lengths = torch.randint(
        max(2, sequence_length // 2), sequence_length + 1, (batch_size,), device=device
    ).clamp(max=sequence_length)
    activity = torch.randint(0, num_activities, (batch_size, sequence_length), device=device)
    elapsed = torch.cumsum(torch.rand(batch_size, sequence_length, device=device) * 3.0, dim=1)
    node_mask = torch.arange(sequence_length, device=device)[None, :] < lengths[:, None]
    attributes = {
        name: torch.randint(0, 8, (batch_size, sequence_length), device=device)
        for name in attribute_names
    }
    return {
        "activity": activity,
        "attributes": attributes,
        "duration_bin": torch.randint(0, 5, (batch_size, sequence_length), device=device),
        "elapsed_bin": torch.randint(0, 5, (batch_size, sequence_length), device=device),
        "delta": torch.rand(batch_size, sequence_length, device=device),
        "elapsed": elapsed,
        "lengths": lengths,
        "node_mask": node_mask,
        "numeric_attributes": {},
        "attribute_missing": {name: torch.zeros_like(values, dtype=torch.bool)
                              for name, values in attributes.items()},
    }


def build_model(num_activities: int, attribute_names, device, hidden_dim=64,
                history_window=32) -> HawkesDYGNN:
    model = HawkesDYGNN(
        num_activities=num_activities,
        attribute_vocab_sizes={name: 8 for name in attribute_names},
        duration_bin_count=5,
        elapsed_bin_count=5,
        attribute_names=list(attribute_names),
        embedding_dim=32,
        hidden_dim=hidden_dim,
        graph_layers=2,
        history_window=history_window,
        dropout=0.2,
        time_scale=1.0,
    ).to(device)
    return model


def time_steps(model, batch, device, steps=20, warmup=5) -> Tuple[float, Dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    labels = torch.randint(0, model.num_activities, (int(batch["activity"].size(0)),), device=device)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(**batch), labels)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(**batch), labels)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_step = (time.perf_counter() - started) / steps
    memory = {}
    if device.type == "cuda":
        memory["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        memory["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2 ** 20
    return per_step, memory


class LagLoopLayer(HawkesGatedDirectedGraphConv):
    """The previous per-lag Python loop, kept only for the benchmark."""

    def forward(self, hidden, activity, elapsed, attributes, numeric_attributes,
                attribute_missing, omega, edge_mask, node_mask, alpha=None):
        batch_size, sequence_length, _ = hidden.shape
        window = omega.size(-1)
        score_slots = hidden.new_zeros((batch_size, sequence_length, window))
        value_slots = hidden.new_zeros((batch_size, sequence_length, window, self.hidden_dim))
        seq_score = hidden.new_zeros((batch_size, sequence_length))
        seq_values = hidden.new_zeros((batch_size, sequence_length, self.hidden_dim))
        seq_valid = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=hidden.device)
        query_all = self.query_projection(hidden)
        for lag in range(1, window + 1):
            source_hidden = hidden[:, :-lag]
            target_hidden = hidden[:, lag:]
            # Reference implementation: same pair view as the batched layer.
            if alpha is None:
                pair_statistic = hidden.new_ones(
                    (batch_size, sequence_length - lag, 1)
                )
            else:
                pair_statistic = alpha.to(hidden.dtype)[
                    activity[:, :-lag], activity[:, lag:]
                ].unsqueeze(-1)
            pair = self.pair_projection(
                torch.cat([pair_statistic,
                           torch.log(pair_statistic.clamp_min(1e-6))], dim=-1)
            )
            lag_ids = torch.full((batch_size, sequence_length - lag), lag,
                                 device=hidden.device, dtype=torch.long)
            lag_emb = self.lag_embedding(lag_ids)
            gap = (elapsed[:, lag:] - elapsed[:, :-lag]).clamp_min(0.0)
            parts = [gap.unsqueeze(-1), torch.log1p(gap).unsqueeze(-1)]
            for name in self.attribute_names:
                values = attributes[name]
                same = (values[:, :-lag] == values[:, lag:]).to(gap.dtype)
                missing = attribute_missing[name]
                change = (missing[:, :-lag] != missing[:, lag:]).to(gap.dtype)
                parts.extend([same.unsqueeze(-1), change.unsqueeze(-1)])
            parts.append(torch.log(pair_statistic.clamp_min(1e-6)))
            static = torch.cat(parts, dim=-1)
            edge_input = torch.cat([source_hidden, target_hidden, target_hidden - source_hidden,
                                    pair, lag_emb, static], dim=-1)
            edge_repr = self.edge_encoder(edge_input)
            score = ((query_all[:, lag:] * self.key_projection(source_hidden)).sum(dim=-1)
                     / math.sqrt(self.hidden_dim)
                     + self.edge_score(edge_repr).squeeze(-1)
                     + self.omega_scale_value() * torch.log1p(omega[:, lag:, lag - 1]))
            values_out = (self.value_projection(source_hidden)
                          + self.edge_value_projection(edge_repr))
            if lag == 1:
                seq_score[:, lag:] = score
                seq_values[:, lag:] = values_out
                seq_valid[:, lag:] = edge_mask[:, lag:, lag - 1]
            else:
                score_slots[:, lag:, lag - 1] = score
                value_slots[:, lag:, lag - 1] = values_out
        hist_mask = edge_mask.clone()
        hist_mask[:, :, 0] = False
        masked = score_slots.masked_fill(~hist_mask, float("-inf"))
        weights = torch.where(hist_mask, torch.softmax(masked, dim=-1),
                              torch.zeros_like(score_slots))
        hist_message = (weights.unsqueeze(-1) * value_slots).sum(dim=2)
        seq_gate = torch.sigmoid(seq_score).unsqueeze(-1)
        seq_message = seq_values * seq_gate * seq_valid.unsqueeze(-1).to(hidden.dtype)
        message = self.self_projection(hidden) + seq_message + hist_message
        output = self.norm(hidden + self.dropout(F.relu(message)))
        return output * node_mask.unsqueeze(-1).to(output.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--num-activities", type=int, default=51)
    parser.add_argument("--history-window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--compare-lag-loop", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    attributes = ["org:resource", "org:role"]
    batch = make_batch(args.batch_size, args.sequence_length, args.num_activities, attributes, device)
    model = build_model(args.num_activities, attributes, device, history_window=args.history_window)
    per_step, memory = time_steps(model, batch, device, steps=args.steps)
    throughput = args.batch_size / per_step
    print(f"vectorised: {per_step * 1000:.2f} ms/step  {throughput:.0f} samples/s  "
          f"{memory.get('peak_allocated_mb', 0):.0f} MiB peak")
    if args.compare_lag_loop:
        reference = build_model(args.num_activities, attributes, device,
                                history_window=args.history_window)
        for layer in reference.graph_layers:
            layer.__class__ = LagLoopLayer
        ref_step, ref_memory = time_steps(reference, batch, device, steps=args.steps)
        print(f"lag loop  : {ref_step * 1000:.2f} ms/step  {args.batch_size / ref_step:.0f} samples/s  "
              f"{ref_memory.get('peak_allocated_mb', 0):.0f} MiB peak")
        print(f"speed-up  : {ref_step / per_step:.2f}x")


if __name__ == "__main__":
    main()
