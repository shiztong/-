import unittest

from recommender.retrieval.debias import examination_weights, weight_diagnostics
from tools.position_bias_demo import simulate

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from recommender.retrieval.sequence import build_sequence_features, TimeAwareSessionEncoder
    from recommender.retrieval.debias import positive_weighted_softmax


class PositionWeightTests(unittest.TestCase):
    def test_weights_clipping_and_support(self):
        self.assertEqual(examination_weights([1, 0.5, 0.001]), [1, 2, 20])
        for p in (0, -0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                examination_weights([p])

    def test_effective_sample_size(self):
        self.assertEqual(weight_diagnostics([2, 2, 2])['effective_sample_size'], 3)
        self.assertLess(weight_diagnostics([1, 1, 100])['effective_sample_size'], 1.05)
        self.assertEqual(weight_diagnostics([])['effective_sample_size'], 0)

    def test_simulated_position_bias_reverses_naive_order(self):
        result = simulate(impressions=20000)['results']
        self.assertLess(result['A']['observed_ctr'], result['B']['observed_ctr'])
        self.assertGreater(result['A']['calibrated_propensity_ht'], result['B']['calibrated_propensity_ht'])
        for row in result.values():
            self.assertLess(abs(row['known_propensity_ht'] - row['latent_relevance']), 0.04)


@unittest.skipIf(torch is None, 'optional PyTorch environment is required')
class SequenceAndLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_actual_sessions_computed_before_truncation(self):
        f = build_sequence_features([1, 2, 3, 4], [0, 20, 4000, 4010], 4100, max_history=3)
        self.assertEqual(f['sessions'], [1, 2, 2])
        self.assertEqual(f['ages'], [4080, 100, 90])
        self.assertTrue(f['history_truncated'])

    def test_future_events_and_unsorted_history_rejected(self):
        for times in ([1, 10], [4, 2]):
            with self.assertRaises(ValueError):
                build_sequence_features([1, 2], times, 10)

    def test_return_after_long_gap_has_empty_current_session(self):
        f = build_sequence_features([1, 2], [1, 2], 10000)
        self.assertEqual(f['current_session'], 2)
        encoder = TimeAwareSessionEncoder(dim=8)
        _, debug = encoder(torch.randn(1, 2, 8), torch.randn(1, 8), torch.tensor([[9999., 9998.]]),
                           torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long),
                           torch.ones(1, 2, dtype=torch.bool), current_sessions=torch.tensor([2]))
        self.assertFalse(debug['short_mask'].any())
        self.assertTrue(debug['long_mask'].all())
        self.assertTrue(torch.equal(debug['gate'], torch.zeros(1, 8)))

    def test_session_encoder_fallback_availability_masks_and_gradients(self):
        encoder = TimeAwareSessionEncoder(dim=8)
        vectors, user = torch.randn(3, 4, 8), torch.randn(3, 8)
        valid = torch.tensor([[False]*4, [True, True, False, False], [True, True, True, False]])
        sessions = torch.tensor([[0]*4, [1, 1, 0, 0], [1, 1, 2, 0]])
        ages = torch.tensor([[0., 0, 0, 0], [100., 10, 0, 0], [5000., 4000, 10, 0]])
        actions = valid.long()
        value, debug = encoder(vectors, user, ages, actions, sessions, valid)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.allclose(value[0], torch.nn.functional.normalize(user[0], dim=-1)))
        self.assertEqual(float(debug['short_weights'][0].sum()), 0)
        self.assertTrue(torch.equal(debug['gate'][1], torch.ones(8)))
        self.assertEqual(debug['short_mask'][2].tolist(), [False, False, True, False])
        self.assertEqual(debug['long_mask'][2].tolist(), [True, True, False, False])
        (value * torch.randn_like(value)).sum().backward()
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_ips_and_normalized_positive_loss(self):
        logits = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        targets = torch.tensor([0, 0])
        raw = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        self.assertTrue(torch.allclose(positive_weighted_softmax(logits, targets, [1., 1.]), raw.mean()))
        weighted = positive_weighted_softmax(logits, targets, [1., 0.5])
        self.assertTrue(torch.allclose(weighted, (raw[0] + 2*raw[1]) / 3))
        ips = positive_weighted_softmax(logits, targets, [1., 0.5], clicks=[1, 0], reduction='ips')
        self.assertTrue(torch.allclose(ips, raw[0] / 2))
        ips.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(2)))

    def test_zero_click_batch_is_finite_zero_loss(self):
        logits = torch.randn(2, 3, requires_grad=True)
        loss = positive_weighted_softmax(logits, torch.tensor([0, 1]), [.5, .2], clicks=[0, 0])
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
