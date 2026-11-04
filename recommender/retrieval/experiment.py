"""Small-catalog CPU training/evaluation/export CLI. No ANN or production service."""

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .models import MODEL_NAMES, ModelConfig, build_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def generate(path, users=96, items=240, steps=32, seed=17):
    path = Path(path)
    if path.exists():
        raise ValueError(f'refusing to overwrite data: {path}')
    if min(users, items) < 1 or not 4 <= steps <= items or items < 12:
        raise ValueError('require users>=1, items>=12, and 4<=steps<=items')
    rng = random.Random(seed)
    categories = 6
    item_categories = [0] + [1 + (i - 1) % categories for i in range(1, items + 1)]
    interactions = []
    for user in range(1, users + 1):
        primary, secondary = rng.sample(range(1, categories + 1), 2)
        seen = set()
        for step in range(steps):
            topic = primary if step < steps * 0.65 else secondary
            available = [i for i in range(1, items + 1) if i not in seen]
            preferred = [i for i in available if item_categories[i] == topic]
            item = rng.choice(preferred if preferred and rng.random() < 0.8 else available)
            seen.add(item)
            interactions.append([user, item, step])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(schema_version=1, synthetic=True, seed=seed, users=users, items=items,
                          categories=categories, item_categories=item_categories, interactions=interactions,
                          notice='Synthetic preference/session-shift data, not real users or business evidence.'))


def load_data(path):
    data = json.loads(Path(path).read_text())
    for key in ('users', 'items', 'categories'):
        if type(data[key]) is not int or data[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if data['items'] > 20000:
        raise ValueError('full-softmax/exact-search harness limited to 20,000 items; use sampled loss/ANN for scale')
    if len(data['item_categories']) != data['items'] + 1 or data['item_categories'][0] != 0:
        raise ValueError('item_categories must have a zero padding entry and one entry per item')
    if any(type(c) is not int or c < 0 or c > data['categories'] for c in data['item_categories']):
        raise ValueError('category ID out of range')
    sequences = {}
    for user, item, timestamp in data['interactions']:
        if type(user) is not int or type(item) is not int or not 1 <= user <= data['users'] or not 1 <= item <= data['items']:
            raise ValueError('interaction ID out of range')
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError('timestamp must be finite numeric')
        sequences.setdefault(user, []).append((timestamp, item))
    if not sequences:
        raise ValueError('no interactions')
    for user, rows in sequences.items():
        rows.sort()
        if len(rows) < 4 or len({t for t, _ in rows}) != len(rows):
            raise ValueError('each observed user needs >=4 events and unique event times')
    return data, sequences


def make_examples(sequences, split):
    result = []
    for user, rows in sorted(sequences.items()):
        ids = [item for _, item in rows]
        indices = range(1, len(ids) - 2) if split == 'train' else [len(ids) - (2 if split == 'valid' else 1)]
        for index in indices:
            result.append((user, ids[:index], ids[index], rows[index][0]))
    return result


def require_global_time_split(sequences):
    # Per-user leave-two-out can otherwise train on other users' future events.
    train_max = max(rows[-3][0] for rows in sequences.values())
    valid_min = min(rows[-2][0] for rows in sequences.values())
    valid_max = max(rows[-2][0] for rows in sequences.values())
    test_min = min(rows[-1][0] for rows in sequences.values())
    if not train_max < valid_min or not valid_max < test_min:
        raise ValueError('leave-two-out is not globally chronological for this input; build explicit time-window splits first')
    return dict(train_max_time=train_max, valid_min_time=valid_min, valid_max_time=valid_max, test_min_time=test_min)


def tensor_batch(examples, max_history):
    users = torch.tensor([row[0] for row in examples], dtype=torch.long)
    history = torch.zeros(len(examples), max_history, dtype=torch.long)
    for j, row in enumerate(examples):
        values = row[1][-max_history:]
        history[j, :len(values)] = torch.tensor(values, dtype=torch.long)
    target = torch.tensor([row[2] for row in examples], dtype=torch.long)
    return users, history, target


@torch.no_grad()
def evaluate(model, examples, top_k=10):
    model.eval()
    # Novel-item evaluation. Report repeated targets rather than silently changing masks.
    unique = [row for row in examples if row[2] not in row[1]]
    if not unique:
        raise ValueError('no novel-item evaluation targets; implement separate repeat-consumption evaluation')
    catalog = model.encode_items()
    hits, ndcg, item_coverage, topic_coverage = 0, 0.0, set(), []
    started = time.perf_counter()
    for offset in range(0, len(unique), 64):
        rows = unique[offset:offset + 64]
        user, history, _ = tensor_batch(rows, model.config.max_history)
        scores = model.scores(user, history, catalog)
        for j, row in enumerate(rows):
            scores[j, row[1]] = -torch.inf
            candidates = min(top_k, torch.isfinite(scores[j]).sum().item())
            if candidates == 0:
                raise ValueError('no eligible candidates')
            ranking = scores[j].topk(candidates).indices.tolist()
            item_coverage.update(ranking)
            topic_coverage.append(len({int(model.items.categories[i]) for i in ranking}))
            if row[2] in ranking:
                hits += 1
                ndcg += 1 / math.log2(ranking.index(row[2]) + 2)
    count = len(unique)
    return dict(top_k=top_k, evaluated_queries=count, skipped_repeated_targets=len(examples) - count,
                recall=hits / count, hit_rate=hits / count, ndcg=ndcg / count,
                unique_recommended_items=len(item_coverage), mean_categories_in_top_k=sum(topic_coverage) / count,
                exact_batched_evaluation_seconds=time.perf_counter() - started,
                protocol='single novel next-item target; full eligible catalog; exact max-over-interest scores; seen excluded')


def train(data_path, output, names=MODEL_NAMES, epochs=12, seed=17, dim=32, interests=4,
          max_history=32, top_k=10, side_features=False, position=True):
    if min(epochs, max_history, interests, top_k) < 1:
        raise ValueError('epochs, history, interests and top-k must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, sequences = load_data(data_path)
    split_contract = require_global_time_split(sequences)
    train_rows, valid_rows, test_rows = [make_examples(sequences, s) for s in ('train', 'valid', 'test')]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, seed=seed, data_sha256=digest(data_path), synthetic=data.get('synthetic', False),
                  warning='Reference-model smoke benchmark. Not paper reproduction or expected business uplift.',
                  python=platform.python_version(), torch=torch.__version__, hardware=platform.machine(),
                  processor=platform.processor(), device='cpu', torch_threads=1, split_contract=split_contract,
                  train_examples=len(train_rows), validation_queries=len(valid_rows), test_queries=len(test_rows),
                  selection='best validation recall; first epoch wins ties; test used only after selection', models=[])
    for name in names:
        torch.manual_seed(seed)
        config = ModelConfig(name, data['users'], data['items'], data['categories'], dim=dim,
                             interests=interests, max_history=max_history, position=position, side_features=side_features)
        model = build_model(config, data['item_categories'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        train_log, best_state, best_metric, best_epoch = [], None, -1.0, None
        shuffle = torch.Generator().manual_seed(seed)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_rows), generator=shuffle).tolist()
            total_loss = 0.0
            for offset in range(0, len(order), 64):
                rows = [train_rows[i] for i in order[offset:offset + 64]]
                user, history, target = tensor_batch(rows, max_history)
                optimizer.zero_grad()
                loss = model.loss(user, history, target)
                if not torch.isfinite(loss):
                    raise ValueError(f'nonfinite loss: {name}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
            valid = evaluate(model, valid_rows, top_k)
            train_log.append(dict(epoch=epoch, train_loss=total_loss / len(train_rows), valid_recall=valid['recall']))
            if valid['recall'] > best_metric:
                best_metric, best_epoch, best_state = valid['recall'], epoch, copy.deepcopy(model.state_dict())
        train_seconds = time.perf_counter() - started
        model.load_state_dict(best_state)
        model.eval()
        model_dir = output / name
        model_dir.mkdir()
        checkpoint = model_dir / 'model.pt'
        torch.save(dict(config=asdict(config), state_dict=best_state, item_categories=data['item_categories'],
                        data_sha256=report['data_sha256'], best_epoch=best_epoch), checkpoint)
        with torch.no_grad():
            vectors = model.encode_items().detach().cpu()
        torch.save(dict(vectors=vectors, checkpoint_sha256=digest(checkpoint),
                        retrieval='exact inner product; max over valid user interests; padding ID zero'), model_dir / 'index.pt')
        result = dict(model=name, config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
                      selected_epoch=best_epoch, train_seconds=train_seconds, train_log=train_log,
                      test=evaluate(model, test_rows, top_k), checkpoint_sha256=digest(checkpoint))
        write_json(model_dir / 'metrics.json', result)
        report['models'].append(result)
        print(json.dumps(dict(model=name, selected_epoch=best_epoch, test_recall=result['test']['recall'])), flush=True)
    write_json(output / 'report.json', report)
    return report


def load_export(checkpoint, index):
    stored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    indexed = torch.load(index, map_location='cpu', weights_only=True)
    if indexed['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('index/checkpoint version mismatch')
    model = build_model(ModelConfig(**stored['config']), stored['item_categories'])
    model.load_state_dict(stored['state_dict'])
    model.eval()
    with torch.no_grad():
        if not torch.allclose(indexed['vectors'], model.encode_items(), atol=1e-7):
            raise ValueError('index vectors differ from checkpoint')
    return model, indexed['vectors']


@torch.no_grad()
def query(checkpoint, index, user_id, history_ids, top_k):
    model, catalog = load_export(checkpoint, index)
    if not 1 <= user_id <= model.config.users or top_k < 1:
        raise ValueError('user ID out of vocabulary or nonpositive top-k')
    if any(not 1 <= i <= model.config.items for i in history_ids):
        raise ValueError('history item out of vocabulary')
    users, history, _ = tensor_batch([(user_id, history_ids, 1, 0)], model.config.max_history)
    scores = model.scores(users, history, catalog)[0]
    scores[history_ids] = -torch.inf
    count = min(top_k, int(torch.isfinite(scores).sum()))
    ids = scores.topk(count).indices.tolist()
    return [dict(item_id=i, score=float(scores[i])) for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    gen = subs.add_parser('generate')
    gen.add_argument('--output', required=True)
    gen.add_argument('--seed', type=int, default=17)
    fit = subs.add_parser('train')
    fit.add_argument('--data', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--model', choices=['all', *MODEL_NAMES], default='all')
    fit.add_argument('--epochs', type=int, default=12)
    fit.add_argument('--seed', type=int, default=17)
    fit.add_argument('--dim', type=int, default=32)
    fit.add_argument('--interests', type=int, default=4)
    fit.add_argument('--max-history', type=int, default=32)
    fit.add_argument('--top-k', type=int, default=10)
    fit.add_argument('--side-features', action='store_true')
    fit.add_argument('--no-position', action='store_true')
    search = subs.add_parser('query')
    search.add_argument('--checkpoint', required=True)
    search.add_argument('--index', required=True)
    search.add_argument('--user-id', required=True, type=int)
    search.add_argument('--history', default='')
    search.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    try:
        if args.action == 'generate':
            generate(args.output, seed=args.seed)
        elif args.action == 'train':
            if args.side_features and args.model in ('all', 'mf'):
                raise ValueError('MF is ID-only; compare side features in separately named runs')
            train(args.data, args.output, names=MODEL_NAMES if args.model == 'all' else [args.model],
                  epochs=args.epochs, seed=args.seed, dim=args.dim, interests=args.interests,
                  max_history=args.max_history, top_k=args.top_k, side_features=args.side_features,
                  position=not args.no_position)
        else:
            history = [int(x) for x in args.history.split(',') if x.strip()]
            print(json.dumps(query(args.checkpoint, args.index, args.user_id, history, args.top_k), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
