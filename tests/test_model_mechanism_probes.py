import math
import unittest

from tools.model_mechanism_probes import attention_pool, ple_indices, run_probes


class MechanismProbeTests(unittest.TestCase):
    def test_attention_without_position_is_permutation_invariant(self):
        events = [[0.2, 0.8], [1.0, -0.3], [0.0, 0.5]]
        for a, b in zip(attention_pool(events), attention_pool(list(reversed(events)))):
            self.assertAlmostEqual(a, b)

    def test_fixed_positions_can_break_invariance(self):
        events = [[1.0, 0.0], [0.0, 1.0]]
        a = attention_pool(events, [0.5, -0.5])
        b = attention_pool(list(reversed(events)), [0.5, -0.5])
        self.assertGreater(abs(a[0] - b[0]), 0.1)

    def test_gate_derivative_matches_finite_difference(self):
        result = run_probes()['mmoe_gate_derivative_of_scalar_output']
        self.assertLess(result['max_absolute_error'], 1e-8)
        self.assertAlmostEqual(sum(result['analytic']), 0)

    def test_index_issue_is_hidden_only_in_equal_count_example(self):
        a, b = ple_indices(2, 2)
        self.assertEqual(a, b)
        a, b = ple_indices(3, 2)
        self.assertNotEqual(a, b)
        self.assertEqual(b[-1], [6, 7])

    def test_parameter_counts_and_materialized_tensor_size(self):
        p = run_probes()['fixed_source_parameter_arithmetic']
        self.assertEqual(p['comirec_dr_projection_only'], 4194304)
        self.assertEqual(p['mind_shared_projection_only'], 4096)
        self.assertEqual(p['comirec_sa_attention_and_positions_excluding_bias'], 33792)
        self.assertEqual(p['literal_dr_fp32_broadcast_gib_if_materialized'], 2)

    def test_distribution_utility_differs_despite_same_mean(self):
        p = run_probes()['expectation_vs_transform']
        self.assertLess(p['sqrt_utility_0_or_20_equal_probability'], p['sqrt_utility_constant_10'])
        self.assertAlmostEqual(p['sqrt_utility_constant_10'], math.sqrt(10))
