"""Time/session components; tested building blocks, not a trained SDM upgrade."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def build_sequence_features(items, times, request_time, actions=None, *, gap_seconds=1800, max_history=100):
    if len(items) != len(times) or (actions is not None and len(actions) != len(items)):
        raise ValueError('items, times and actions must align')
    if not math.isfinite(request_time) or not math.isfinite(gap_seconds) or gap_seconds <= 0 or type(max_history) is not int or max_history < 1:
        raise ValueError('invalid request time, gap or max_history')
    if any(not math.isfinite(t) or t >= request_time for t in times):
        raise ValueError('every event must be finite and strictly before the request')
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError('history must be chronologically ordered')
    if any(type(i) is not int or i < 1 for i in items):
        raise ValueError('item IDs must be positive integers')
    actions = [1] * len(items) if actions is None else actions
    if any(type(a) is not int or a < 1 for a in actions):
        raise ValueError('observed action IDs must be positive integers')
    session_ids, session = [], 0
    for index, timestamp in enumerate(times):
        if index == 0 or timestamp - times[index - 1] > gap_seconds:
            session += 1
        session_ids.append(session)
    keep = slice(max(0, len(items) - max_history), None)
    kept_times = times[keep]
    current_session = session if times and request_time - times[-1] <= gap_seconds else session + 1
    return dict(items=list(items[keep]), ages=[request_time - t for t in kept_times], current_session=current_session,
                actions=list(actions[keep]), sessions=session_ids[keep], history_truncated=len(items) > max_history,
                note='session IDs computed before truncation; gap is a hypothesis, not latent-intent truth')


def masked_softmax(scores, valid):
    # Zero mass, not uniform pseudo-observations, for an entirely absent branch.
    weights = scores.masked_fill(~valid, -1e9).softmax(-1) * valid
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


class TimeAwareSessionEncoder(nn.Module):
    """Actual session masks, action/time tokens, recency bias and availability-aware gate.

    Explicit delta from the existing fixed-window SessionGateRetriever. This
    component is not yet integrated with its trainer or evaluated for business lift.
    """

    def __init__(self, dim=32, action_count=8, time_scale_seconds=3600):
        super().__init__()
        if dim < 1 or action_count < 1 or time_scale_seconds <= 0 or not math.isfinite(time_scale_seconds):
            raise ValueError('dimensions and time scale must be positive')
        self.time_scale = time_scale_seconds
        self.action = nn.Embedding(action_count + 1, 8, padding_idx=0)
        self.token = nn.Linear(dim + 9, dim)
        self.short_cell = nn.GRUCell(dim, dim)
        self.short_key = nn.Linear(dim, dim, bias=False)
        self.long_key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.raw_decay = nn.Parameter(torch.tensor(-2.0))
        self.gate = nn.Linear(3 * dim + 2, dim)

    def forward(self, item_vectors, user_vector, ages, actions, sessions, valid, *, current_sessions=None):
        if item_vectors.ndim != 3 or item_vectors.shape[1] < 1:
            raise ValueError('use at least one padded time step for empty histories')
        shape = item_vectors.shape[:2]
        if any(x.shape != shape for x in (ages, actions, sessions, valid)) or user_vector.shape != (shape[0], item_vectors.shape[-1]):
            raise ValueError('sequence tensors must align')
        if valid.dtype != torch.bool or not torch.isfinite(ages).all() or (ages[valid] < 0).any():
            raise ValueError('valid must be boolean and observed ages finite/nonnegative')
        if (sessions[valid] <= 0).any():
            raise ValueError('observed session IDs must be positive')
        for row in range(shape[0]):
            if (sessions[row][valid[row]].diff() < 0).any() or (ages[row][valid[row]].diff() > 0).any():
                raise ValueError('session IDs and ages must follow chronological order')
        age = ages.clamp_min(0) / self.time_scale
        tokens = self.token(torch.cat([item_vectors, self.action(actions), torch.log1p(age).unsqueeze(-1)], -1))
        last_session = sessions.masked_fill(~valid, 0).max(-1).values
        if current_sessions is not None:
            if current_sessions.shape != (shape[0],) or (current_sessions < last_session).any():
                raise ValueError('current session IDs must align and not precede observed sessions')
            last_session = current_sessions
        short_mask = valid & (sessions == last_session[:, None])
        long_mask = valid & ~short_mask
        state = user_vector.new_zeros(user_vector.shape)
        states = []
        for step in range(shape[1]):
            proposal = self.short_cell(tokens[:, step], state)
            state = torch.where(short_mask[:, step, None], proposal, state)
            states.append(state)
        short_values = torch.stack(states, dim=1)
        q = self.query(user_vector)
        short_logits = (self.short_key(short_values) * q[:, None]).sum(-1) / math.sqrt(q.shape[-1])
        long_logits = (self.long_key(tokens) * q[:, None]).sum(-1) / math.sqrt(q.shape[-1])
        # This is a recency inductive bias, NOT display-position debiasing.
        penalty = F.softplus(self.raw_decay) * age
        short_weights = masked_softmax(short_logits - penalty, short_mask)
        long_weights = masked_softmax(long_logits - penalty, long_mask)
        short = (short_weights[..., None] * short_values).sum(1)
        long = (long_weights[..., None] * tokens).sum(1)
        counts = torch.stack([short_mask.sum(-1), long_mask.sum(-1)], -1).float().log1p()
        gate = torch.sigmoid(self.gate(torch.cat([short, long, user_vector, counts], -1)))
        short_exists, long_exists = short_mask.any(-1), long_mask.any(-1)
        gate = torch.where((short_exists & ~long_exists)[:, None], torch.ones_like(gate), gate)
        gate = torch.where((long_exists & ~short_exists)[:, None], torch.zeros_like(gate), gate)
        output = gate * short + (1 - gate) * long
        output = torch.where(valid.any(-1, keepdim=True), output, user_vector)
        return F.normalize(output, dim=-1), dict(gate=gate, short_weights=short_weights, long_weights=long_weights,
                                                short_mask=short_mask, long_mask=long_mask)
