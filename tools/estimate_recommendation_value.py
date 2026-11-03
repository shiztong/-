"""Scenario planning, not a forecast or a substitute for randomized experiments."""

import argparse
import json
import math
from statistics import NormalDist


def positive(value, name, allow_zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def evaluate(config):
    dau = positive(config['dau'], 'dau')
    minutes = positive(config['baseline_minutes_per_user_day'], 'baseline minutes')
    value = positive(config['contribution_per_incremental_minute'], 'unit contribution')
    costs = config['incremental_costs']
    daily_cost = sum(positive(costs[key], key, True) for key in (
        'serving_per_day', 'training_per_day', 'data_storage_per_day',
    )) + positive(costs['engineering_one_time'], 'engineering', True) / positive(
        costs['engineering_amortization_days'], 'amortization days',
    )
    scenarios = []
    for scenario in config['scenarios']:
        uplift = scenario['relative_uplift']
        positive(abs(uplift), 'absolute uplift', True)
        if uplift < -1:
            raise ValueError('relative uplift cannot be below -1')
        incremental_minutes = dau * minutes * uplift
        contribution = incremental_minutes * value
        scenarios.append({
            'name': scenario['name'], 'relative_uplift': uplift,
            'incremental_minutes_per_day': incremental_minutes,
            'incremental_contribution_per_day': contribution,
            'incremental_cost_per_day': daily_cost,
            'net_per_day': contribution - daily_cost,
        })
    exp = config['experiment']
    sigma = positive(exp['user_metric_stddev'], 'user-level stddev')
    delta = positive(exp['absolute_mde'], 'absolute MDE')
    alpha, power = exp['alpha_two_sided'], exp['power']
    if not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError('require alpha in (0,1), power in (0.5,1)')
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    n = math.ceil(2 * z * z * sigma * sigma / (delta * delta))
    return {
        'warning': 'Hypothetical scenarios; not measured lifts. Currency is the input currency.',
        'break_even_relative_uplift': daily_cost / (dau * minutes * value),
        'scenarios': scenarios,
        'experiment': {
            'approx_unique_users_per_arm': n,
            'unit': 'independent user-level fixed-window metric',
            'warning': 'Normal approximation, equal allocation/variance; repeated days are not new users. No calendar duration inferred.',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='JSON scenario configuration')
    args = parser.parse_args()
    try:
        with open(args.config, encoding='utf-8') as stream:
            output = evaluate(json.load(stream))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
