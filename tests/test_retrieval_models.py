import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from recommender.retrieval.models import MODEL_NAMES, ModelConfig, build_model
    from recommender.retrieval.experiment import generate, train, load_export, query, require_global_time_split
    from tools.run_recall_suite import aggregate, main as suite_main


@unittest.skipIf(torch is None, 'optional PyTorch recall environment is not installed')
class RetrievalModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.categories = [0] + [i % 3 + 1 for i in range(12)]

    def test_all_six_models_handle_empty_history_and_backpropagate(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(ModelConfig(name, 4, 12, 3, dim=8, max_history=8), self.categories)
                history = torch.tensor([[0] * 8, [1, 2, 3, 0, 0, 0, 0, 0]])
                users = torch.tensor([1, 2])
                loss = model.loss(users, history, torch.tensor([4, 5]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
                scores = model.scores(users, history)
                self.assertEqual(tuple(scores.shape), (2, 13))
                self.assertTrue(torch.isfinite(scores[:, 1:]).all())

    def test_padding_does_not_change_any_model_output(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(ModelConfig(name, 4, 12, 3, dim=8, max_history=8), self.categories).eval()
                user = torch.tensor([1])
                a, _ = model.encode_user(user, torch.tensor([[1, 2, 3, 0]]))
                b, _ = model.encode_user(user, torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]))
                self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_sa_without_position_is_order_invariant(self):
        config = ModelConfig('comirec_sa_ref', 4, 12, 3, dim=8, max_history=8, position=False)
        model = build_model(config, self.categories).eval()
        user = torch.tensor([1])
        a, _ = model.encode_user(user, torch.tensor([[1, 2, 3, 0]]))
        b, _ = model.encode_user(user, torch.tensor([[3, 1, 2, 0]]))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_mind_active_interest_counts(self):
        model = build_model(ModelConfig('mind_ref', 4, 12, 3, dim=8, max_history=8), self.categories)
        history = torch.tensor([[0] * 8, [1, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 0]])
        _, valid = model.encode_user(torch.tensor([1, 2, 3, 4]), history)
        self.assertEqual(valid.sum(-1).tolist(), [1, 1, 2, 3])

    def test_global_future_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'globally chronological'):
            require_global_time_split({1: [(1, 1), (2, 2), (3, 3), (4, 4)],
                                       2: [(10, 1), (20, 2), (30, 3), (40, 4)]})

    def test_training_export_query_and_version_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / 'data.json', Path(tmp) / 'run'
            generate(data, users=8, items=24, steps=8)
            report = train(data, out, names=['pooled', 'comirec_sa_ref'], epochs=2, dim=8, max_history=8)
            self.assertEqual(len(report['models']), 2)
            for result in report['models']:
                model_dir = out / result['model']
                checkpoint, index = model_dir / 'model.pt', model_dir / 'index.pt'
                model, vectors = load_export(checkpoint, index)
                self.assertTrue(torch.allclose(model.encode_items(), vectors))
                recommendations = query(checkpoint, index, 1, [1, 2], 5)
                self.assertEqual(len(recommendations), 5)
                self.assertFalse({r['item_id'] for r in recommendations} & {0, 1, 2})
                self.assertEqual(result['test']['evaluated_queries'], 8)
            changed = torch.load(index, weights_only=True)
            changed['checkpoint_sha256'] = 'wrong'
            torch.save(changed, index)
            with self.assertRaisesRegex(ValueError, 'version mismatch'):
                load_export(checkpoint, index)

    def test_config_driven_suite_and_aggregation_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = dict(data=dict(users=8, items=24, steps=8, seed=17), models=['pooled'],
                            training_seeds=[17, 18], epochs=1, dim=8, interests=2, max_history=8, top_k=5)
            config, output = Path(tmp) / 'config.json', Path(tmp) / 'suite'
            config.write_text(json.dumps(settings))
            with patch('sys.argv', ['suite', '--config', str(config), '--output', str(output)]):
                suite_main()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['suite_config'], settings)
            self.assertEqual(summary['summary'][0]['seeds'], [17, 18])
            first = output / 'seed17/report.json'
            with self.assertRaisesRegex(ValueError, 'distinct'):
                aggregate([first, first])
            changed = json.loads((output / 'seed18/report.json').read_text())
            changed['models'][0]['test']['top_k'] = 10
            bad = Path(tmp) / 'incompatible.json'
            bad.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                aggregate([first, bad])
