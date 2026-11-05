"""Deterministic PBM simulation, not a real UI/propensity measurement."""

import argparse
import json
import random

from recommender.retrieval.debias import examination_weights, weight_diagnostics


def simulate(seed=17, impressions=200000):
    if impressions < 1:
        raise ValueError('impressions must be positive')
    rng = random.Random(seed)
    relevance = {'A': 0.8, 'B': 0.4}
    exam = [1.0, 0.25]
    # Independent calibration sample with item randomized over both positions.
    calibration_n, calibration_c = [0, 0], [0, 0]
    for _ in range(impressions):
        item, pos = rng.choice(['A', 'B']), rng.randrange(2)
        calibration_n[pos] += 1
        calibration_c[pos] += int(rng.random() < exam[pos] and rng.random() < relevance[item])
    ctr = [c / n for c, n in zip(calibration_c, calibration_n)]
    estimated_relative_exam = [1.0, ctr[1] / ctr[0]]
    if not 0 < estimated_relative_exam[1] <= 1:
        raise ValueError('calibration failed; increase sample size, do not silently cap estimated propensities')
    accum = {i: dict(n=0, clicks=0, known_ipw_sum=0.0, estimated_ipw_sum=0.0, positive_weights=[]) for i in relevance}
    for _ in range(impressions):
        item = rng.choice(['A', 'B'])
        pos = 1 if (rng.random() < (0.9 if item == 'A' else 0.1)) else 0
        click = int(rng.random() < exam[pos] and rng.random() < relevance[item])
        row = accum[item]
        row['n'] += 1
        row['clicks'] += click
        row['known_ipw_sum'] += click / exam[pos]
        row['estimated_ipw_sum'] += click / estimated_relative_exam[pos]
        if click:
            row['positive_weights'].append(examination_weights([estimated_relative_exam[pos]])[0])
    results = {}
    for item, row in accum.items():
        results[item] = dict(latent_relevance=relevance[item], impressions=row['n'], observed_ctr=row['clicks'] / row['n'],
                             known_propensity_ht=row['known_ipw_sum'] / row['n'],
                             calibrated_propensity_ht=row['estimated_ipw_sum'] / row['n'],
                             positive_weight_diagnostics=weight_diagnostics(row['positive_weights']))
    return dict(warning=__doc__, seed=seed, calibration_impressions=impressions, evaluation_impressions=impressions,
                assumptions='C=E*R, E independent of R given position; randomized calibration; e(first)=1 known in simulator',
                true_examination=exam, calibrated_relative_examination=estimated_relative_exam, results=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--impressions', type=int, default=200000)
    args = p.parse_args()
    print(json.dumps(simulate(args.seed, args.impressions), indent=2))


if __name__ == '__main__':
    main()
