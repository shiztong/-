"""Run the actual synthetic CPU recall suite and aggregate training-seed variation."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def aggregate(paths, source_root=None):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    if not reports or len({r['data_sha256'] for r in reports}) != 1:
        raise ValueError('all reports must use the same data snapshot')
    if len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('training seeds must be distinct')
    model_sets = [{r['model'] for r in report['models']} for report in reports]
    if any(s != model_sets[0] for s in model_sets):
        raise ValueError('model sets must match')
    rows = []
    for result in reports[0]['models']:
        name = result['model']
        values = [next(m for m in report['models'] if m['model'] == name) for report in reports]
        if any(v['config'] != values[0]['config'] or v['test']['protocol'] != values[0]['test']['protocol']
               or v['test']['top_k'] != values[0]['test']['top_k'] for v in values):
            raise ValueError('model configuration and evaluation protocol must match across seeds')
        recalls = [v['test']['recall'] for v in values]
        ndcgs = [v['test']['ndcg'] for v in values]
        rows.append(dict(model=name, parameters=result['parameters'], recall_mean=statistics.mean(recalls),
                         recall_training_seed_sample_std=statistics.stdev(recalls) if len(recalls) > 1 else None,
                         ndcg_mean=statistics.mean(ndcgs), seeds=[r['seed'] for r in reports],
                         per_seed_recall=recalls, selected_epochs=[v['selected_epoch'] for v in values],
                         evaluated_queries_per_seed=[v['test']['evaluated_queries'] for v in values]))
    hashes = {}
    if source_root:
        for filename in ('recommender/retrieval/models.py', 'recommender/retrieval/experiment.py'):
            hashes[filename] = hashlib.sha256((Path(source_root) / filename).read_bytes()).hexdigest()
    return dict(warning='Synthetic training smoke results only. Seed spread is not a confidence interval or business forecast.',
                data_sha256=reports[0]['data_sha256'], synthetic=reports[0]['synthetic'], summary=rows,
                source_file_sha256=hashes, run_reports=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/recall/cpu-smoke.json')
    parser.add_argument('--output', required=True, help='new directory; existing files are preserved')
    args = parser.parse_args()
    from recommender.retrieval.experiment import generate, train, write_json
    settings = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    data = root / 'data.json'
    generate(data, **settings['data'])
    reports = []
    for seed in settings['training_seeds']:
        output = root / f'seed{seed}'
        train(data, output, names=settings['models'], seed=seed,
              **{k: settings[k] for k in ('epochs', 'dim', 'interests', 'max_history', 'top_k')})
        reports.append(output / 'report.json')
    summary = aggregate(reports, Path.cwd())
    summary['suite_config'] = settings
    summary['suite_config_sha256'] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    write_json(root / 'summary.json', summary)


if __name__ == '__main__':
    main()
