"""Examination-only positive-loss IPW helpers, not arbitrary-policy OPE."""

import math


def examination_weights(propensities, *, min_propensity=0.01, max_weight=20.0):
    if not 0 < min_propensity <= 1 or not math.isfinite(min_propensity):
        raise ValueError('min_propensity must be in (0,1]')
    if max_weight is not None and (max_weight < 1 or not math.isfinite(max_weight)):
        raise ValueError('max_weight must be finite and >=1, or None')
    if not propensities or any(not math.isfinite(p) or not 0 < p <= 1 for p in propensities):
        raise ValueError('known propensities must lie in (0,1]; zero support is not recoverable')
    weights = [1 / max(p, min_propensity) for p in propensities]
    return [min(w, max_weight) if max_weight is not None else w for w in weights]


def weight_diagnostics(weights):
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('weights must be finite and nonnegative')
    total, square = sum(weights), sum(w * w for w in weights)
    return dict(count=len(weights), sum=total, max=max(weights, default=0),
                effective_sample_size=total * total / square if square else 0)


def positive_weighted_softmax(logits, targets, propensities, *, clicks=None, reduction='snips',
                              min_propensity=0.01, max_weight=20.0):
    """Correct examination selection in a positive log-softmax objective.

    With ALL impressions, clicks/e and reduction='ips' estimates the corresponding
    positive relevance objective under PBM assumptions, absent clipping. With
    clicked-only rows use SNIPS as a normalized reweighted positive objective.
    It does not turn unclicked impressions into observed irrelevance labels.
    """
    import torch
    from torch.nn import functional as F

    if logits.ndim != 2 or logits.shape[0] < 1 or targets.shape != (logits.shape[0],):
        raise ValueError('expected [B,V] logits and [B] targets')
    p_tensor = torch.as_tensor(propensities)
    if p_tensor.ndim != 1:
        raise ValueError('examination propensities must be a one-dimensional vector')
    p = p_tensor.detach().cpu().tolist()
    if len(p) != len(targets):
        raise ValueError('one examination propensity per observed event is required')
    weights = logits.new_tensor(examination_weights(p, min_propensity=min_propensity, max_weight=max_weight))
    click = torch.ones_like(weights) if clicks is None else torch.as_tensor(clicks, device=logits.device, dtype=logits.dtype).detach()
    if click.shape != weights.shape or not torch.isfinite(click).all() or not ((click == 0) | (click == 1)).all():
        raise ValueError('clicks must be an aligned binary mask')
    loss = F.cross_entropy(logits, targets, reduction='none')
    weighted = loss * weights * click
    if reduction == 'ips':
        return weighted.mean()
    if reduction == 'snips':
        return weighted.sum() / (weights * click).sum().clamp_min(1e-8)
    raise ValueError('reduction must be ips or snips')
