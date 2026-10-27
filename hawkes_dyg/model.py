"""Sparse causal history graph model for next-activity prediction."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(value_tensor)))


def inverse_sigmoid(value: float, maximum: float = 1.0) -> float:
    """Return a logit initializing ``maximum * sigmoid(raw)`` to ``value``."""

    if maximum <= 0.0:
        raise ValueError("maximum must be positive")
    ratio = min(max(float(value) / float(maximum), 1e-5), 1.0 - 1e-5)
    return float(math.log(ratio / (1.0 - ratio)))


def sinusoidal_position_encoding(
    length: int, dimension: int, device: torch.device
) -> torch.Tensor:
    """Return the standard fixed sinusoidal encoding for positions 0..length-1."""

    if length == 0:
        return torch.empty((0, dimension), device=device)
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / max(dimension, 1))
    )
    encoding = torch.zeros((length, dimension), device=device)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(
        positions * div_term[: encoding[:, 1::2].shape[1]]
    )
    return encoding


class NodeFeatureEncoder(nn.Module):
    """Embed activities, attributes, time buckets, continuous time and position."""

    def __init__(
        self,
        num_activities: int,
        attribute_vocab_sizes: Mapping[str, int],
        duration_bin_count: int,
        elapsed_bin_count: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activity_embedding = nn.Embedding(num_activities, embedding_dim)
        self.attribute_names = list(attribute_vocab_sizes.keys())
        self.attribute_embeddings = nn.ModuleDict(
            {
                f"attr_{idx}": nn.Embedding(size, embedding_dim, padding_idx=0)
                for idx, size in enumerate(attribute_vocab_sizes.values())
            }
        )
        self.attribute_embedding_keys = {
            name: f"attr_{idx}" for idx, name in enumerate(self.attribute_names)
        }
        self.duration_embedding = nn.Embedding(duration_bin_count, embedding_dim)
        self.elapsed_embedding = nn.Embedding(elapsed_bin_count, embedding_dim)
        # ``delta`` and ``elapsed`` are kept as continuous signals in addition
        # to their quantized embeddings.  The projection has its own small
        # bottleneck so the scale of raw time cannot dominate categorical
        # embeddings while still allowing the model to use fine-grained gaps.
        self.continuous_time_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        input_dim = embedding_dim * (3 + len(self.attribute_names)) + 2 * hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def continuous_time_features(
        delta: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor:
        """Return ``[..., 2]`` log-scaled ``(delta, elapsed)`` channels.

        ``delta`` is already ``log1p`` of a scaled interval.  ``elapsed`` is the
        raw offset from the case start, which on sub-hourly logs reaches six
        figures while ``delta`` stays single digits; stacking them unconverted
        lets one channel dominate the shared linear layer.  Both are therefore
        expressed in nats before projection.
        """

        log_elapsed = torch.log1p(elapsed.clamp_min(0.0))
        stacked = torch.stack([delta, log_elapsed], dim=-1)
        return torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        activity: torch.Tensor,
        attributes: Mapping[str, torch.Tensor],
        duration_bin: torch.Tensor,
        elapsed_bin: torch.Tensor,
        node_mask: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
        elapsed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces = [self.activity_embedding(activity)]
        for name in self.attribute_names:
            pieces.append(
                self.attribute_embeddings[self.attribute_embedding_keys[name]](
                    attributes[name]
                )
            )
        pieces.extend(
            [self.duration_embedding(duration_bin), self.elapsed_embedding(elapsed_bin)]
        )
        batch_size, sequence_length = activity.shape
        position = sinusoidal_position_encoding(
            sequence_length, self.hidden_dim, activity.device
        )
        position = position.unsqueeze(0).expand(batch_size, -1, -1)
        if delta is None:
            delta = torch.zeros_like(position[..., 0])
        if elapsed is None:
            elapsed = torch.zeros_like(position[..., 0])
        # ``delta`` is already log-scaled upstream; ``elapsed`` is not, and its
        # range spans several orders of magnitude between logs (one
        # ``time_scale`` is 37 s on Receipt but three days on Helpdesk).  Log
        # both before they share a linear layer, otherwise the wider channel
        # dominates the projection regardless of the next LayerNorm, which
        # normalizes the hidden dimension rather than these two inputs.
        continuous_time = self.continuous_time_features(delta, elapsed)
        continuous_time = self.continuous_time_projection(continuous_time)
        output = self.projection(torch.cat(pieces + [position, continuous_time], dim=-1))
        return output * node_mask.unsqueeze(-1).to(output.dtype)


class HawkesTemporalInfluence(nn.Module):
    """Hawkes-inspired temporal excitation and candidate intensity prior."""

    def __init__(
        self,
        num_activities: int,
        initial_alpha: float = 0.5,
        initial_beta: float = 0.5,
        initial_mu: float = 0.1,
        epsilon: float = 1e-6,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.num_activities = num_activities
        # ``time_scale`` is the median positive inter-event time of the training
        # split (in days).  ``FeatureEncoder`` already divides ``elapsed`` by it,
        # so gaps reaching this module are dimensionless multiples of one median
        # inter-event time.  The buffer is retained purely for checkpoint
        # introspection and must not be applied to gaps a second time.
        self.register_buffer(
            "time_scale",
            torch.tensor(max(float(time_scale), epsilon), dtype=torch.float32),
        )
        self.theta_alpha = nn.Parameter(
            torch.full(
                (num_activities, num_activities),
                inverse_softplus(initial_alpha),
            )
        )
        self.theta_beta = nn.Parameter(torch.tensor(inverse_softplus(initial_beta)))
        self.theta_mu = nn.Parameter(
            torch.full((num_activities,), inverse_softplus(initial_mu))
        )

    def positive_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.theta_alpha)
        beta = F.softplus(self.theta_beta) + self.epsilon
        mu = F.softplus(self.theta_mu) + self.epsilon
        return alpha, beta, mu

    def log_gap(self, gap: torch.Tensor) -> torch.Tensor:
        """Return the log-scaled time interval used by the paper's delta_ji.

        ``gap`` arrives already divided by ``time_scale``: ``FeatureEncoder``
        stores ``elapsed`` as ``days / time_scale``.  Dividing again here would
        evaluate ``log1p(raw_gap / time_scale**2)``, which is not the intended
        dimensionless interval -- on a log with sub-hourly events (``Receipt``,
        ``time_scale`` ~ 37 s) it inflates the argument by more than three
        orders of magnitude and saturates the kernel to a constant.  The buffer
        is kept for checkpoint introspection only.
        """

        return torch.log1p(gap)

    def kernel(self, gap: torch.Tensor) -> torch.Tensor:
        """Hawkes decay evaluated on the log time scale: exp(-beta * delta_ji).

        Scaling the decay in log time is what keeps the kernel informative
        across several orders of magnitude of inter-event gaps.  With a linear
        gap a median of a few days already drives exp(-beta * gap) to zero, so
        both the prior and its gradient vanish.
        """

        beta = self.positive_parameters()[1]
        return torch.exp(-beta * self.log_gap(gap))

    def edge_scores(
        self,
        activity: torch.Tensor,
        elapsed: torch.Tensor,
        node_mask: torch.Tensor,
        history_window: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return lag-aligned omega [B,L,H], edge mask and beta."""

        if history_window < 1:
            raise ValueError("history_window must be at least one")
        batch_size, sequence_length = activity.shape
        window = min(history_window, max(sequence_length - 1, 1))
        alpha, beta, _ = self.positive_parameters()
        # One gather over the whole window replaces the per-lag Python loop.
        # ``positions``/``lags`` are ``[1, L, H]`` so the result keeps the
        # ``[B, L, H]`` layout that ``omega`` and ``edge_mask`` use.
        lags = torch.arange(1, window + 1, device=activity.device).view(1, 1, window)
        positions = torch.arange(sequence_length, device=activity.device).view(1, sequence_length, 1)
        target_positions = positions.expand(batch_size, sequence_length, window)
        source_positions = (positions - lags).clamp_min(0).expand(batch_size, sequence_length, window)
        # An edge is usable only when the source is inside the sequence and both
        # endpoints are real events.  The source mask is gathered at the clamped
        # source position; slots whose source would be negative are invalidated
        # by the first term below, so the clamp never admits a phantom edge.
        # ``int8`` avoids relying on bool support in ``gather``.
        source_mask = torch.gather(
            node_mask.to(torch.int8), 1, source_positions.reshape(batch_size, -1)
        ).view(batch_size, sequence_length, window).bool()
        valid = (positions - lags >= 0) & node_mask.unsqueeze(-1) & source_mask
        gather_target = target_positions.reshape(batch_size, -1)
        gather_source = source_positions.reshape(batch_size, -1)
        target_activity = torch.gather(activity, 1, gather_target).view(batch_size, sequence_length, window)
        source_activity = torch.gather(activity, 1, gather_source).view(batch_size, sequence_length, window)
        target_time = torch.gather(elapsed, 1, gather_target).view(batch_size, sequence_length, window)
        source_time = torch.gather(elapsed, 1, gather_source).view(batch_size, sequence_length, window)
        gap = (target_time - source_time).clamp_min(0.0)
        pair_alpha = alpha[source_activity, target_activity]
        pair_omega = pair_alpha * self.kernel(gap)
        omega = torch.where(valid, pair_omega, torch.zeros_like(pair_omega))
        return omega, valid, beta

    def candidate_intensity(
        self,
        activity: torch.Tensor,
        elapsed: torch.Tensor,
        lengths: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all-candidate intensity at the end of each prefix."""

        alpha, beta, mu = self.positive_parameters()
        batch_index = torch.arange(activity.size(0), device=activity.device)
        last_index = lengths.to(activity.device) - 1
        last_time = elapsed[batch_index, last_index]
        gap = (last_time.unsqueeze(1) - elapsed).clamp_min(0.0)
        kernel = self.kernel(gap) * node_mask.to(elapsed.dtype)
        source_alpha = alpha[activity]
        intensity = mu.unsqueeze(0) + (source_alpha * kernel.unsqueeze(-1)).sum(dim=1)
        return intensity, beta, mu


class HawkesGatedDirectedGraphConv(nn.Module):
    """Edge-conditioned causal attention over a fixed recent-history window."""

    def __init__(
        self,
        hidden_dim: int,
        num_activities: int,
        attribute_names: Sequence[str],
        numeric_attribute_names: Sequence[str],
        history_window: int,
        dropout: float,
        pair_dim: int = 16,
        lag_dim: int = 8,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if history_window < 1:
            raise ValueError("history_window must be at least one")
        self.hidden_dim = hidden_dim
        self.history_window = history_window
        # Mirrors ``HawkesTemporalInfluence.time_scale`` for checkpoint
        # introspection.  Gaps are normalized upstream by ``FeatureEncoder``, so
        # this buffer is not applied when building edge descriptors.
        self.register_buffer(
            "time_scale", torch.tensor(max(float(time_scale), 1e-6), dtype=torch.float32)
        )
        self.attribute_names = list(attribute_names)
        self.numeric_attribute_names = list(numeric_attribute_names)
        self.num_activities = num_activities
        # Activity-pair strength is learned once, as ``alpha`` in
        # ``HawkesTemporalInfluence``.  This layer used to own an independent
        # ``N x N`` embedding for the same thing, so two parameterizations
        # competed to explain the transition structure and neither was
        # identifiable.  The projection below is a cheap nonlinear view of the
        # prior's own matrix: it adds expressiveness but cannot encode a second,
        # contradictory pair statistic.
        self.pair_projection = nn.Sequential(
            nn.Linear(2, pair_dim),
            nn.ReLU(),
        )
        self.lag_embedding = nn.Embedding(history_window + 1, lag_dim)
        # ``static_dim`` counts gap, log-gap, log pair excitation, then a
        # (same, missing_change) pair per categorical attribute and a
        # (difference, |difference|) pair per numeric one.  ``log1p(omega)``
        # used to be appended here as well, which duplicated the attention bias
        # below: the encoder's first layer is linear and could cancel it, making
        # ``omega_scale`` uninterpretable.  The bias is now the only path from
        # the Hawkes prior to attention; ``log alpha`` below is the prior's
        # activity-pair strength on its own, so the product ``omega = alpha *
        # kernel(gap)`` is exposed as two factors rather than one.
        static_dim = 3 + 2 * len(self.attribute_names) + 2 * len(
            self.numeric_attribute_names
        )
        edge_input_dim = 3 * hidden_dim + pair_dim + lag_dim + static_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_score = nn.Linear(hidden_dim, 1)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # A non-negative, bounded Hawkes bias prevents the temporal prior from
        # reversing its intended meaning or overwhelming learned graph scores.
        self.omega_scale_max = 1.0
        self.theta_omega_scale = nn.Parameter(
            torch.tensor(inverse_sigmoid(0.1, self.omega_scale_max))
        )
        self.last_attention_entropy = torch.tensor(0.0)
        self.last_seq_gate_mean = torch.tensor(0.0)
        # Flat source index per (L, window, device); rebuilt only when the batch
        # shape changes.  Plain attribute so it never enters the state dict.
        self._slot_cache: dict = {}

    def omega_scale_value(self) -> torch.Tensor:
        return self.omega_scale_max * torch.sigmoid(self.theta_omega_scale)

    @property
    def omega_scale(self) -> torch.Tensor:
        """Bounded non-negative value used in edge attention."""

        return self.omega_scale_value()

    def _slot_positions(
        self,
        sequence_length: int,
        window: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the cached ``[window * L]`` source index of every edge slot.

        Flat slot ``w * L + i`` is the edge from event ``i - (w + 1)`` to event
        ``i``, so the source position is simply ``clamp(i - w - 1, 0)``.  The
        target position is the identity, which callers exploit by broadcasting
        instead of gathering.

        The tensor is cached per shape because ``window`` and ``L`` only change
        between batches, not between layers or steps.  It is a 1-D contiguous
        index so ``index_select`` can use its fast path; the previous version
        gathered with an expanded stride-0 index of shape ``[B, L*W, d]``, which
        forced a slow generic kernel.
        """

        key = (sequence_length, window, str(device))
        cached = self._slot_cache.get(key)
        if cached is not None:
            return cached
        positions = torch.arange(sequence_length, device=device).view(1, sequence_length)
        lags = torch.arange(1, window + 1, device=device).view(window, 1)
        sources = (positions - lags).clamp_min(0).reshape(-1).contiguous()
        self._slot_cache[key] = sources
        return sources

    @staticmethod
    def _gather_slots(
        values: torch.Tensor,
        index: torch.Tensor,
        sequence_length: int,
        window: int,
    ) -> torch.Tensor:
        """Gather ``[B, L, ...]`` into ``[B, L, window, ...]`` along the event axis.

        ``index`` is the flat ``[window * L]`` source index.  Boolean inputs are
        gathered as ``int8`` because several torch builds do not implement
        ``gather`` for ``bool``; the result is cast back so callers keep the
        original dtype.
        """

        batch = values.size(0)
        is_bool = values.dtype == torch.bool
        source = values.to(torch.int8) if is_bool else values
        gathered = torch.index_select(source, 1, index)
        if source.dim() == 2:
            gathered = gathered.view(batch, window, sequence_length).transpose(1, 2)
        else:
            gathered = gathered.view(
                batch, window, sequence_length, source.size(-1)
            ).transpose(1, 2)
        return gathered.bool() if is_bool else gathered

    def _lag_embeddings(
        self, sequence_length: int, window: int, device: torch.device, batch_size: int
    ) -> torch.Tensor:
        """Return ``[B, L, W, lag_dim]`` lag embeddings.

        Only the integer id grid is cached, never the embedding output: that
        carries a slot into the parameter graph and would point at a freed graph
        on the next backward pass (``Trying to backward through the graph a
        second time``).  Re-running the lookup keeps autograd correct and costs
        one gather over ``L * W`` ids.

        A separate direct/history embedding used to sit beside this one, but its
        id was ``lag == 1`` and therefore a deterministic function of the lag
        already embedded here.  The direct-predecessor edge is gated separately
        in ``forward``, so the extra table only duplicated existing capacity.
        """

        key = ("lag_ids", sequence_length, window, str(device))
        lags = self._slot_cache.get(key)
        if lags is None:
            lags = torch.arange(1, window + 1, device=device).view(1, window, 1)
            lags = lags.expand(1, window, sequence_length).transpose(1, 2).contiguous()
            self._slot_cache[key] = lags
        return self.lag_embedding(lags).expand(batch_size, -1, -1, -1)

    def _static_edge_features(
        self,
        elapsed: torch.Tensor,
        pair_alpha: torch.Tensor,
        attributes: Mapping[str, torch.Tensor],
        numeric_attributes: Mapping[str, torch.Tensor],
        attribute_missing: Mapping[str, torch.Tensor],
        sources: torch.Tensor,
        sequence_length: int,
        window: int,
    ) -> torch.Tensor:
        """Return ``[B, L, W, static_dim]`` change descriptors for every slot.

        ``sources`` is the flat ``[W * L]`` source index; the target side is the
        identity, so those terms are broadcast with ``unsqueeze`` instead of
        gathered.  Each remaining gather produces the whole window at once.

        ``pair_alpha`` is the Hawkes excitation strength of each slot's own
        activity pair, already gathered by the caller.  The layer used to keep
        an independent ``N x N`` embedding of the same statistic while the prior
        kept a free matrix, so two parameterizations competed to explain the
        transition structure and neither was identifiable.  Feeding the prior's
        value in keeps a single source of truth.
        """

        source_elapsed = self._gather_slots(elapsed, sources, sequence_length, window)
        gap = (elapsed.unsqueeze(2) - source_elapsed).clamp_min(0.0)
        # Same log time scale as the Hawkes kernel, so the descriptor exposes the
        # exact quantity delta_ji that the prior is computed from.  ``elapsed``
        # is already in units of ``time_scale`` (see ``HawkesTemporalInfluence.
        # log_gap``), so no further division is applied here.
        log_gap = torch.log1p(gap)
        # ``pair_alpha`` arrives as ``[B, L, W, 1]``; keep it that way so every
        # descriptor piece stays a single channel.
        pieces = [gap.unsqueeze(-1), log_gap.unsqueeze(-1)]
        pieces.append(torch.log(pair_alpha.clamp_min(1e-6)))
        for name in self.attribute_names:
            values = attributes[name]
            source_values = self._gather_slots(values, sources, sequence_length, window)
            same = (values.unsqueeze(2) == source_values).to(gap.dtype)
            missing = attribute_missing[name]
            missing_change = (
                self._gather_slots(missing, sources, sequence_length, window)
                != missing.unsqueeze(2)
            ).to(gap.dtype)
            pieces.extend([same.unsqueeze(-1), missing_change.unsqueeze(-1)])
        for name in self.numeric_attribute_names:
            values = numeric_attributes[name]
            difference = values.unsqueeze(2) - self._gather_slots(
                values, sources, sequence_length, window
            )
            pieces.extend([difference.unsqueeze(-1), difference.abs().unsqueeze(-1)])
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        activity: torch.Tensor,
        elapsed: torch.Tensor,
        attributes: Mapping[str, torch.Tensor],
        numeric_attributes: Mapping[str, torch.Tensor],
        attribute_missing: Mapping[str, torch.Tensor],
        omega: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden.shape
        window = omega.size(-1)
        sources = self._slot_positions(sequence_length, window, hidden.device)
        # Gather the source side once for the whole window; the target side is
        # the identity mapping, so it is broadcast instead of gathered.  Every
        # tensor below has shape ``[B, L, W, ...]`` and comes from one batched
        # kernel, which is what removes the per-lag Python loop.
        source_hidden = self._gather_slots(hidden, sources, sequence_length, window)
        target_hidden = hidden.unsqueeze(2).expand(
            batch_size, sequence_length, window, self.hidden_dim
        )
        source_activity = self._gather_slots(activity, sources, sequence_length, window)
        target_activity = activity.unsqueeze(2)
        # The pair view is built from ``alpha`` itself (handed in by the caller)
        # rather than from a private table.  ``alpha`` and ``log alpha`` give the
        # projection both a linear and a log-scaled handle on pair strength.
        if alpha is None:
            pair_statistic = torch.ones_like(omega).unsqueeze(-1)
        else:
            pair_statistic = alpha.to(omega.dtype)[
                source_activity, target_activity
            ].unsqueeze(-1)
        pair = self.pair_projection(
            torch.cat([pair_statistic, torch.log(pair_statistic.clamp_min(1e-6))], dim=-1)
        )
        lag_emb = self._lag_embeddings(
            sequence_length, window, hidden.device, batch_size
        )
        static = self._static_edge_features(
            elapsed,
            pair_statistic,
            attributes,
            numeric_attributes,
            attribute_missing,
            sources,
            sequence_length,
            window,
        )
        edge_input = torch.cat(
            [
                source_hidden,
                target_hidden,
                target_hidden - source_hidden,
                pair,
                lag_emb,
                static,
            ],
            dim=-1,
        )
        edge_repr = self.edge_encoder(edge_input)
        # Every slot of a target shares that target's query vector, so broadcast
        # the [B, L, d] query along the window instead of gathering it.
        query = self.query_projection(hidden).unsqueeze(2)
        score = (
            (query * self.key_projection(source_hidden)).sum(dim=-1)
            / math.sqrt(self.hidden_dim)
            + self.edge_score(edge_repr).squeeze(-1)
            + self.omega_scale_value() * torch.log1p(omega)
        )
        values = self.value_projection(source_hidden) + self.edge_value_projection(
            edge_repr
        )

        # ``-inf`` is applied only immediately before softmax below.  Keeping
        # invalid slots at zero while constructing edge representations avoids
        # propagating NaNs from padded examples.
        seq_score = score[:, :, 0]
        seq_values = values[:, :, 0]
        seq_valid = edge_mask[:, :, 0]

        # Explicit masked softmax over historical (lag >= 2) edges.  A row
        # with no valid history is returned as all zeros, including the first
        # event and padded positions.
        hist_mask = edge_mask.clone()
        hist_mask[:, :, 0] = False
        masked_scores = score.masked_fill(~hist_mask, float("-inf"))
        weights = torch.softmax(masked_scores, dim=-1)
        weights = torch.where(hist_mask, weights, torch.zeros_like(weights))
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        hist_message = (weights.unsqueeze(-1) * values).sum(dim=2)

        # Use a bounded sigmoid gate only for the direct predecessor.  This
        # retains a stable scale while guaranteeing that a valid seq edge can
        # contribute independently of the number of history edges.
        seq_gate = torch.sigmoid(seq_score).unsqueeze(-1)
        seq_message = seq_values * seq_gate * seq_valid.unsqueeze(-1).to(hidden.dtype)
        message = self.self_projection(hidden) + seq_message + hist_message
        output = self.norm(hidden + self.dropout(F.relu(message)))
        output = output * node_mask.unsqueeze(-1).to(output.dtype)

        valid_targets = node_mask & (hist_mask.any(dim=-1))
        entropy = -(
            weights * torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        ).sum(dim=-1)
        if valid_targets.any():
            self.last_attention_entropy = entropy.masked_select(valid_targets).mean().detach()
        else:
            self.last_attention_entropy = hidden.new_zeros(())
        self.last_seq_gate_mean = seq_gate.squeeze(-1).masked_select(seq_valid).mean().detach() if seq_valid.any() else hidden.new_zeros(())
        return output


class HawkesDYGNN(nn.Module):
    """Next-activity predictor over sparse causal history graphs."""

    MODEL_VERSION = "causal_history_v5"

    def __init__(
        self,
        num_activities: int,
        attribute_vocab_sizes: Mapping[str, int],
        duration_bin_count: int,
        elapsed_bin_count: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        graph_layers: int = 2,
        dropout: float = 0.2,
        history_window: int = 32,
        attribute_names: Optional[Sequence[str]] = None,
        numeric_attribute_names: Optional[Sequence[str]] = None,
        initial_alpha: float = 0.5,
        initial_beta: float = 0.5,
        initial_mu: float = 0.1,
        intensity_scale_init: float = 0.1,
        intensity_scale_max: float = 1.0,
        omega_scale_init: float = 0.1,
        omega_scale_max: float = 1.0,
        time_scale: float = 1.0,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()
        if model_version not in (None, self.MODEL_VERSION):
            raise ValueError(
                f"Incompatible model version {model_version!r}; "
                f"expected {self.MODEL_VERSION!r}"
            )
        self.num_activities = num_activities
        self.history_window = int(history_window)
        if graph_layers < 1:
            raise ValueError("graph_layers must be at least one")
        if self.history_window < 1:
            raise ValueError("history_window must be at least one")
        resolved_attribute_names = list(attribute_names or attribute_vocab_sizes.keys())
        resolved_numeric_names = list(numeric_attribute_names or [])
        self.node_encoder = NodeFeatureEncoder(
            num_activities=num_activities,
            attribute_vocab_sizes=attribute_vocab_sizes,
            duration_bin_count=duration_bin_count,
            elapsed_bin_count=elapsed_bin_count,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.influence = HawkesTemporalInfluence(
            num_activities,
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            initial_mu=initial_mu,
            time_scale=time_scale,
        )
        # Activity-pair strength lives only in ``self.influence.theta_alpha``.
        # The graph layers receive the resulting matrix and build their pair
        # view from it, so there is exactly one place that models transitions.
        self.graph_layers = nn.ModuleList(
            [
                HawkesGatedDirectedGraphConv(
                    hidden_dim=hidden_dim,
                    num_activities=num_activities,
                    attribute_names=resolved_attribute_names,
                    numeric_attribute_names=resolved_numeric_names,
                    history_window=self.history_window,
                    dropout=dropout,
                    time_scale=time_scale,
                )
                for _ in range(graph_layers)
            ]
        )
        if intensity_scale_max <= 0.0:
            raise ValueError("intensity_scale_max must be positive")
        if not 0.0 < intensity_scale_init <= intensity_scale_max:
            raise ValueError("intensity_scale_init must lie in (0, intensity_scale_max]")
        self.intensity_scale_max = float(intensity_scale_max)
        self.theta_intensity_scale = nn.Parameter(
            torch.tensor(inverse_sigmoid(intensity_scale_init, self.intensity_scale_max))
        )
        # All graph layers share the same bound; retain the constructor args
        # for checkpoint/config introspection while initializing each layer.
        if not 0.0 < omega_scale_init <= omega_scale_max:
            raise ValueError("omega_scale_init must lie in (0, omega_scale_max]")
        for layer in self.graph_layers:
            layer.omega_scale_max = float(omega_scale_max)
            with torch.no_grad():
                layer.theta_omega_scale.fill_(
                    inverse_sigmoid(omega_scale_init, omega_scale_max)
                )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, num_activities),
        )
        # Auxiliary regression on the final change vector.  ``last_delta`` is
        # the only readout term that carries the "change" claim, but nothing in
        # the task loss constrains what it means: ``classifier`` may ignore it
        # and still fit the activity distribution.  Predicting the log gap of
        # the last transition forces it to encode temporal change, and the
        # target is the prefix's own last interval, so no future information
        # enters.  Trained only as an auxiliary term; inference is unchanged.
        self.change_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        activity: torch.Tensor,
        attributes: Mapping[str, torch.Tensor],
        duration_bin: torch.Tensor,
        elapsed_bin: torch.Tensor,
        delta: torch.Tensor,
        elapsed: torch.Tensor,
        lengths: torch.Tensor,
        node_mask: torch.Tensor,
        numeric_attributes: Mapping[str, torch.Tensor],
        attribute_missing: Mapping[str, torch.Tensor],
        return_aux: bool = False,
    ):
        hidden = self.node_encoder(
            activity,
            attributes,
            duration_bin,
            elapsed_bin,
            node_mask,
            delta=delta,
            elapsed=elapsed,
        )
        omega, edge_mask, beta = self.influence.edge_scores(
            activity, elapsed, node_mask, self.history_window
        )
        alpha, _, _ = self.influence.positive_parameters()
        for layer in self.graph_layers:
            hidden = layer(
                hidden,
                activity,
                elapsed,
                attributes,
                numeric_attributes,
                attribute_missing,
                omega,
                edge_mask,
                node_mask,
                alpha=alpha,
            )
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        last_index = lengths.to(hidden.device) - 1
        last = hidden[batch_index, last_index]
        previous_index = (last_index - 1).clamp_min(0)
        previous = hidden[batch_index, previous_index]
        last_delta = last - previous
        mask = node_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / lengths.clamp_min(1).to(
            hidden.dtype
        ).unsqueeze(-1)
        graph_representation = torch.cat([last, mean, last_delta], dim=-1)
        logits = self.classifier(graph_representation)
        change_prediction = self.change_head(last_delta).squeeze(-1)
        intensity, beta, mu = self.influence.candidate_intensity(
            activity, elapsed, lengths, node_mask
        )
        intensity_scale = self.intensity_scale_max * torch.sigmoid(self.theta_intensity_scale)
        log_intensity = torch.log(intensity.clamp_min(self.influence.epsilon))
        # Normalize the prior across candidate activities so absolute Hawkes
        # scale and prefix length cannot arbitrarily shift all logits.
        normalized_log_intensity = log_intensity - torch.logsumexp(
            log_intensity, dim=-1, keepdim=True
        )
        logits = logits + intensity_scale * normalized_log_intensity
        if return_aux:
            alpha, _, _ = self.influence.positive_parameters()
            attention_entropy = torch.stack(
                [layer.last_attention_entropy for layer in self.graph_layers]
            ).mean()
            seq_gate_mean = torch.stack(
                [layer.last_seq_gate_mean for layer in self.graph_layers]
            ).mean()
            return logits, {
                "omega": omega,
                "edge_mask": edge_mask,
                "alpha": alpha,
                "beta": beta,
                "mu": mu,
                "intensity": intensity,
                "intensity_scale": intensity_scale,
                "normalized_log_intensity": normalized_log_intensity,
                "omega_scale": torch.stack(
                    [layer.omega_scale_value() for layer in self.graph_layers]
                ),
                "attention_entropy": attention_entropy,
                "seq_gate_mean": seq_gate_mean,
                "change_prediction": change_prediction,
                # The target of the auxiliary head: the log gap that produced
                # the last event of the prefix, already inside the prefix.
                "change_target": delta[batch_index, last_index].to(change_prediction.dtype),
            }
        return logits

