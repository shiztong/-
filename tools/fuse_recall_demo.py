"""Fuse actually exported synthetic-model queries; not a production effect experiment."""

import argparse
import json
from pathlib import Path

from recommender.retrieval.fusion import Candidate, ChannelResult, fuse_candidates


def run(checkpoint_root):
    from recommender.retrieval.experiment import query
    channels = []
    for name in ('pooled', 'session_gate', 'mind_ref'):
        folder = Path(checkpoint_root) / name
        result = query(folder / 'model.pt', folder / 'index.pt', 1, [1, 2, 3], 20)
        channels.append(ChannelResult(name, [Candidate(r['item_id'], r['score']) for r in result], min_quota=2, max_quota=5))
    fused = fuse_candidates(channels, budget=10, excluded=[1, 2, 3])
    return dict(warning=__doc__, user_id=1, history=[1, 2, 3], per_channel_top_k=20, result=fused)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root', required=True)
    args = p.parse_args()
    print(json.dumps(run(args.checkpoint_root), ensure_ascii=False, indent=2))
