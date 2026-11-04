"""Runnable reference designs, NOT exact reproductions or production models."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    model: str
    users: int
    items: int
    categories: int
    dim: int = 32
    interests: int = 4
    max_history: int = 32
    short_window: int = 8
    position: bool = True
    side_features: bool = False


MODEL_NAMES = ('mf', 'pooled', 'session_gate', 'mind_ref', 'comirec_sa_ref', 'comirec_dr_ref')


def safe_attention(query, keys, mask, attention):
    # Add a zero sentinel, accessible only when the entire history is empty.
    sentinel = keys.new_zeros(keys.shape[0], 1, keys.shape[-1])
    extended = torch.cat([keys, sentinel], dim=1)
    valid = torch.cat([mask, ~mask.any(dim=1, keepdim=True)], dim=1)
    output, _ = attention(query, extended, extended, key_padding_mask=~valid, need_weights=False)
    return output


class ItemTower(nn.Module):
    def __init__(self, config, item_categories):
        super().__init__()
        self.embedding = nn.Embedding(config.items + 1, config.dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        self.register_buffer('categories', torch.tensor(item_categories, dtype=torch.long))
        self.side = config.side_features
        if self.side:
            self.category_embedding = nn.Embedding(config.categories + 1, config.dim, padding_idx=0)
            self.fusion = nn.Linear(2 * config.dim, config.dim)

    def forward(self, ids):
        value = self.embedding(ids)
        if self.side:
            value = self.fusion(torch.cat([value, self.category_embedding(self.categories[ids])], dim=-1))
        return value * (ids != 0).unsqueeze(-1)


class Retriever(nn.Module):
    def __init__(self, config, item_categories):
        super().__init__()
        self.config = config
        self.items = ItemTower(config, item_categories)
        self.users = nn.Embedding(config.users + 1, config.dim)
        nn.init.normal_(self.users.weight, std=0.1)

    def encode_user(self, user_ids, history):
        raise NotImplementedError

    def encode_items(self):
        ids = torch.arange(self.config.items + 1, device=self.users.weight.device)
        return F.normalize(self.items(ids), dim=-1)

    def loss(self, user_ids, history, target, temperature=0.1):
        interests, valid = self.encode_user(user_ids, history)  # [B,K,d]
        catalog = self.encode_items()                         # [V+1,d]
        target_vectors = catalog[target]
        # Label-aware hard readout ONLY during supervised training.
        choice = torch.einsum('bkd,bd->bk', interests, target_vectors).masked_fill(~valid, -torch.inf).argmax(-1)
        selected = interests[torch.arange(len(target), device=target.device), choice]
        logits = selected @ catalog.T / temperature
        logits[:, 0] = -torch.inf
        return F.cross_entropy(logits, target)  # Full softmax: small-catalog harness.

    def scores(self, user_ids, history, catalog=None):
        interests, valid = self.encode_user(user_ids, history)
        catalog = self.encode_items() if catalog is None else catalog
        scores = torch.einsum('bkd,vd->bkv', interests, catalog)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -torch.inf).max(dim=1).values
        scores[:, 0] = -torch.inf
        return scores

    def finish(self, vectors):
        vectors = F.normalize(vectors, dim=-1)
        return vectors, torch.ones(vectors.shape[:2], dtype=torch.bool, device=vectors.device)


class MatrixFactorization(Retriever):
    def encode_user(self, user_ids, history):
        return self.finish(self.users(user_ids).unsqueeze(1))


class PooledTwoTower(Retriever):
    def __init__(self, config, item_categories):
        super().__init__(config, item_categories)
        d = config.dim
        self.network = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def encode_user(self, user_ids, history):
        mask = history != 0
        pooled = self.items(history).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.finish(self.network(torch.cat([pooled, self.users(user_ids)], dim=-1)).unsqueeze(1))


class SessionGateRetriever(Retriever):
    """SDM-inspired: fixed recent window, not an original SDM reproduction."""

    def __init__(self, config, item_categories):
        super().__init__(config, item_categories)
        d = config.dim
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.short_self = nn.MultiheadAttention(d, 4, batch_first=True)
        self.short_pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.long_pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.gate = nn.Linear(3 * d, d)

    def encode_user(self, user_ids, history):
        mask = history != 0
        lengths = mask.sum(1)
        width = min(self.config.short_window, history.shape[1])
        offsets = torch.arange(width, device=history.device).unsqueeze(0)
        start = (lengths - width).clamp_min(0).unsqueeze(1)
        indices = (start + offsets).clamp_max(history.shape[1] - 1)
        short_ids = history.gather(1, indices)
        short_mask = offsets < lengths.clamp_max(width).unsqueeze(1)
        short_ids = short_ids * short_mask
        short, _ = self.lstm(self.items(short_ids))
        short = safe_attention(short, short, short_mask, self.short_self)
        user = self.users(user_ids).unsqueeze(1)
        s = safe_attention(user, short, short_mask, self.short_pool).squeeze(1)
        long_mask = mask & (torch.arange(history.shape[1], device=history.device).unsqueeze(0) < start)
        long = safe_attention(user, self.items(history), long_mask, self.long_pool).squeeze(1)
        g = torch.sigmoid(self.gate(torch.cat([s, long, user.squeeze(1)], dim=-1)))
        fused = g * s + (1 - g) * long
        fused = torch.where(mask.any(1, keepdim=True), fused, user.squeeze(1))
        return self.finish(fused.unsqueeze(1))


class MultiInterestBase(Retriever):
    def __init__(self, config, item_categories):
        super().__init__(config, item_categories)
        self.user_fusion = nn.Linear(2 * config.dim, config.dim)

    def finish_multi(self, vectors, user_ids, lengths, dynamic=False):
        user = self.users(user_ids).unsqueeze(1).expand_as(vectors)
        output = self.user_fusion(torch.cat([vectors, user], dim=-1))
        output = torch.where((lengths > 0)[:, None, None], output, user)
        valid = torch.ones(output.shape[:2], dtype=torch.bool, device=output.device)
        if dynamic:
            count = torch.log2(lengths.float() + 1).floor().long().clamp(1, self.config.interests)
            valid = torch.arange(self.config.interests, device=output.device)[None, :] < count[:, None]
        return F.normalize(output, dim=-1), valid


class ComiRecSAReference(MultiInterestBase):
    def __init__(self, config, item_categories):
        super().__init__(config, item_categories)
        d = config.dim
        self.attention = nn.Sequential(nn.Linear(d, 4 * d), nn.Tanh(), nn.Linear(4 * d, config.interests))
        self.positions = nn.Parameter(torch.randn(config.max_history, d) * 0.01) if config.position else None

    def encode_user(self, user_ids, history):
        mask = history != 0
        values = self.items(history)
        keys = values if self.positions is None else values + self.positions[:history.shape[1]]
        logits = self.attention(keys).transpose(1, 2).masked_fill(~mask[:, None, :], -1e9)
        weights = logits.softmax(-1) * mask[:, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        vectors = weights @ values
        return self.finish_multi(vectors, user_ids, mask.sum(1))


class RoutingReference(MultiInterestBase):
    def __init__(self, config, item_categories, position_specific):
        super().__init__(config, item_categories)
        d, k = config.dim, config.interests
        self.position_specific = position_specific
        if position_specific:
            self.projection = nn.Parameter(torch.randn(config.max_history, k, d, d) / d ** 0.5)
        else:
            self.projection = nn.Parameter(torch.randn(d, d) / d ** 0.5)
        # Deterministic after checkpointing, unlike per-call random initialization.
        self.register_buffer('routing_initial', torch.randn(config.max_history, k) * 0.01)

    @staticmethod
    def squash(value):
        norm2 = value.square().sum(-1, keepdim=True)
        return norm2 / (1 + norm2) * value / (norm2 + 1e-9).sqrt()

    def encode_user(self, user_ids, history):
        mask = history != 0
        values = self.items(history)
        length = history.shape[1]
        if self.position_specific:
            votes = torch.einsum('bld,lkde->blke', values, self.projection[:length])
            logits = values.new_zeros(len(history), length, self.config.interests)
        else:
            votes = (values @ self.projection).unsqueeze(2).expand(-1, -1, self.config.interests, -1)
            logits = self.routing_initial[:length].unsqueeze(0).expand(len(history), -1, -1)
        for step in range(3):
            weights = logits.softmax(-1) * mask.unsqueeze(-1)
            current = votes if step == 2 else votes.detach()
            vectors = self.squash((weights.unsqueeze(-1) * current).sum(1))
            if step < 2:
                logits = logits + (votes.detach() * vectors.detach().unsqueeze(1)).sum(-1)
        return self.finish_multi(vectors, user_ids, mask.sum(1), dynamic=not self.position_specific)


class MINDReference(RoutingReference):
    def __init__(self, config, item_categories):
        super().__init__(config, item_categories, position_specific=False)


class ComiRecDRReference(RoutingReference):
    def __init__(self, config, item_categories):
        super().__init__(config, item_categories, position_specific=True)


def build_model(config, item_categories):
    if config.dim < 4 or config.dim % 4:
        raise ValueError('dim must be a positive multiple of four')
    if min(config.users, config.items, config.categories, config.interests, config.max_history, config.short_window) < 1:
        raise ValueError('counts and windows must be positive')
    if config.model == 'mf' and config.side_features:
        raise ValueError('pure MF comparison does not enable side features')
    classes = dict(zip(MODEL_NAMES, (MatrixFactorization, PooledTwoTower, SessionGateRetriever,
                                    MINDReference, ComiRecSAReference, ComiRecDRReference)))
    if config.model not in classes:
        raise ValueError(f'unknown model: {config.model}')
    return classes[config.model](config, item_categories)
