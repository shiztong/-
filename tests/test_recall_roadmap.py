import json
import tempfile
import unittest
from pathlib import Path

from tools.check_recall_roadmap import load_and_validate


class RoadmapContractTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'configs/recall').mkdir(parents=True)
        (root / 'doc.md').write_text('Design and scope')
        (root / 'core.py').write_text('def component():\n    return 1\n')
        return dict(items=[dict(id='DATA', status='P', priority='P0', doc='doc.md', scope='missing real data',
                               exit_gate='as-of replay', evidence_scope='design', adoption_requires=[], core_code=[]),
                           dict(id='COMP', status='C', priority='P1', doc='doc.md', scope='unit tested only',
                                exit_gate='integrated experiment', evidence_scope='component_tests',
                                adoption_requires=['DATA'], core_code=[dict(path='core.py', symbol='component')])])

    def test_component_can_exist_while_adoption_data_gate_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            (root / 'configs/recall/roadmap.json').write_text(json.dumps(data))
            self.assertEqual(load_and_validate(root)['items'][1]['status'], 'C')

    def test_cycles_and_evidence_promotion_are_rejected(self):
        for defect in ('cycle', 'promotion', 'missing_code'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = self.fixture(root)
                if defect == 'cycle':
                    data['items'][0]['adoption_requires'] = ['COMP']
                elif defect == 'promotion':
                    data['items'][0]['status'] = 'L'
                else:
                    data['items'][1]['core_code'][0]['symbol'] = 'not_implemented'
                (root / 'configs/recall/roadmap.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_and_validate(root)
