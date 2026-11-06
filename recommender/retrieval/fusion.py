"""Deterministic rank-based fusion component; no online orchestration or learned calibration."""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Candidate:
    item_id: int
    score: float | None = None


@dataclass(frozen=True)
class ChannelResult:
    name: str
    candidates: list[Candidate] = field(default_factory=list)
    available: bool = True
    min_quota: int = 0
    max_quota: int | None = None
    weight: float = 1.0


def fuse_candidates(channels, budget, *, excluded=(), eligible=None, rrf_k=60):
    """RRF with best-effort minimum and hard OWNERSHIP maximum quotas.

    Every selected item has exactly one selected_by owner, but retains all source
    provenance. max_quota caps owner counts, NOT appearances in the provenance
    sets. A production policy that caps all source coverage needs a different
    constrained selector. Eligibility errors propagate (no fail-open fallback).
    """
    if type(budget) is not int or budget < 1 or not math.isfinite(rrf_k) or rrf_k < 1:
        raise ValueError('budget must be positive integer and rrf_k finite >=1')
    names = [c.name for c in channels]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError('channel names must be nonempty and unique')
    for channel in channels:
        if type(channel.min_quota) is not int or channel.min_quota < 0 or (
            channel.max_quota is not None and (type(channel.max_quota) is not int or channel.max_quota < channel.min_quota)
        ) or not math.isfinite(channel.weight) or channel.weight <= 0:
            raise ValueError('invalid channel quota or weight')
    if sum(c.min_quota for c in channels) > budget:
        raise ValueError('declared minimum quotas exceed budget')
    excluded = set(excluded)
    records, channel_ids, diagnostics, counts, eligibility_cache = {}, {}, {}, {}, {}
    for channel in channels:
        counts[channel.name] = 0
        channel_ids[channel.name] = []
        diagnostics[channel.name] = dict(available=channel.available, input_count=len(channel.candidates),
                                         duplicate_within_channel=0, excluded=0, eligible_count=0)
        if not channel.available:
            continue
        seen = set()
        for candidate in channel.candidates:
            if type(candidate.item_id) is not int or candidate.item_id <= 0 or (
                candidate.score is not None and not math.isfinite(candidate.score)
            ):
                raise ValueError('candidate IDs must be positive and diagnostic scores finite')
            if candidate.item_id in seen:
                diagnostics[channel.name]['duplicate_within_channel'] += 1
                continue
            seen.add(candidate.item_id)
            if candidate.item_id not in eligibility_cache:
                eligibility_cache[candidate.item_id] = candidate.item_id not in excluded and (
                    eligible is None or bool(eligible(candidate.item_id)))
            if not eligibility_cache[candidate.item_id]:
                diagnostics[channel.name]['excluded'] += 1
                continue
            # Use input order after filtering/dedup, never compare raw model scores.
            rank = len(channel_ids[channel.name]) + 1
            channel_ids[channel.name].append(candidate.item_id)
            contribution = channel.weight / (rrf_k + rank)
            record = records.setdefault(candidate.item_id, dict(item_id=candidate.item_id, fusion_score=0.0, sources=[]))
            record['fusion_score'] += contribution
            record['sources'].append(dict(channel=channel.name, rank=rank, raw_score=candidate.score,
                                          contribution=contribution))
        diagnostics[channel.name]['eligible_count'] = len(channel_ids[channel.name])
    caps = {c.name: c.max_quota if c.max_quota is not None else budget for c in channels}
    selected, selected_ids = [], set()

    def take(item_id, owner):
        if item_id in selected_ids or counts[owner] >= caps[owner] or len(selected) >= budget:
            return False
        selected.append(dict(records[item_id], selected_by=owner))
        selected_ids.add(item_id)
        counts[owner] += 1
        return True

    # Round-robin protects configured minimums without pretending overlapping or
    # empty channels can always meet them.
    for level in range(max((c.min_quota for c in channels), default=0)):
        for channel in channels:
            if channel.available and level < channel.min_quota:
                for item_id in channel_ids[channel.name]:
                    if take(item_id, channel.name):
                        break
    for record in sorted(records.values(), key=lambda r: (-r['fusion_score'], r['item_id'])):
        if len(selected) >= budget:
            break
        owners = sorted(record['sources'], key=lambda s: (-s['contribution'], names.index(s['channel'])))
        for source in owners:
            if take(record['item_id'], source['channel']):
                break
    for channel in channels:
        diagnostics[channel.name]['selected_as_owner'] = counts[channel.name]
        diagnostics[channel.name]['unmet_minimum'] = max(0, channel.min_quota - counts[channel.name])
    return dict(candidates=selected, diagnostics=diagnostics, requested=budget, returned=len(selected),
                shortfall=budget - len(selected),
                policy='rank-based RRF; minimums best-effort; maximums constrain selected_by ownership only')
