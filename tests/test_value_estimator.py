import copy
import json
import unittest
from pathlib import Path

from tools.estimate_recommendation_value import evaluate


class ValueEstimatorTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'docs/recommendation/evidence/value-scenarios.example.json'
        self.config = json.loads(path.read_text())

    def test_break_even_matches_zero_net(self):
        result = evaluate(self.config)
        self.assertAlmostEqual(result['break_even_relative_uplift'], 1400 / 60000)
        self.config['scenarios'] = [{'name': 'break_even', 'relative_uplift': result['break_even_relative_uplift']}]
        self.assertAlmostEqual(evaluate(self.config)['scenarios'][0]['net_per_day'], 0)

    def test_no_effect_still_pays_cost(self):
        results = {s['name']: s for s in evaluate(self.config)['scenarios']}
        self.assertEqual(results['no_effect']['net_per_day'], -1400)
        self.assertEqual(results['larger_positive']['net_per_day'], 400)

    def test_sample_size_scales_with_variance_and_inverse_mde_squared(self):
        n = evaluate(self.config)['experiment']['approx_unique_users_per_arm']
        cfg = copy.deepcopy(self.config)
        cfg['experiment']['absolute_mde'] /= 2
        n2 = evaluate(cfg)['experiment']['approx_unique_users_per_arm']
        self.assertLessEqual(abs(n2 - 4 * n), 3)

    def test_invalid_assumptions_rejected(self):
        for field, bad in [('dau', 0), ('dau', float('nan')), ('contribution_per_incremental_minute', -1)]:
            cfg = copy.deepcopy(self.config)
            cfg[field] = bad
            with self.assertRaises(ValueError):
                evaluate(cfg)
