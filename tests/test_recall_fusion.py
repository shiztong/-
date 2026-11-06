import unittest

from recommender.retrieval.fusion import Candidate, ChannelResult, fuse_candidates


class RecallFusionTests(unittest.TestCase):
    def test_dedup_retains_provenance_and_filters_before_quota(self):
        result = fuse_candidates([
            ChannelResult('a', [Candidate(1), Candidate(2), Candidate(2), Candidate(3)], min_quota=1),
            ChannelResult('b', [Candidate(2), Candidate(4)], min_quota=1)], 3, excluded=[1])
        ids = [r['item_id'] for r in result['candidates']]
        self.assertEqual(len(set(ids)), 3)
        self.assertNotIn(1, ids)
        self.assertEqual({s['channel'] for s in next(r for r in result['candidates'] if r['item_id'] == 2)['sources']}, {'a', 'b'})
        self.assertEqual(result['diagnostics']['a']['duplicate_within_channel'], 1)

    def test_raw_score_scale_does_not_affect_selection(self):
        a = [ChannelResult('x', [Candidate(1, .01), Candidate(2, .001)]), ChannelResult('y', [Candidate(3, 9000), Candidate(2, 8000)])]
        b = [ChannelResult('x', [Candidate(1, 100), Candidate(2, 50)]), ChannelResult('y', [Candidate(3, -1), Candidate(2, -2)])]
        self.assertEqual([c['item_id'] for c in fuse_candidates(a, 3)['candidates']],
                         [c['item_id'] for c in fuse_candidates(b, 3)['candidates']])

    def test_owner_caps_and_degraded_channel(self):
        result = fuse_candidates([ChannelResult('a', [Candidate(i) for i in range(1, 9)], max_quota=2),
                                  ChannelResult('b', [Candidate(9)], available=False, min_quota=1)], 4)
        self.assertEqual(result['returned'], 2)
        self.assertEqual(result['shortfall'], 2)
        self.assertEqual(result['diagnostics']['b']['unmet_minimum'], 1)

    def test_owner_cap_does_not_claim_provenance_cap(self):
        result = fuse_candidates([ChannelResult('a', [Candidate(1), Candidate(2)], max_quota=0),
                                  ChannelResult('b', [Candidate(1), Candidate(2)])], 2)
        self.assertTrue(all(c['selected_by'] == 'b' for c in result['candidates']))
        self.assertTrue(all('a' in {s['channel'] for s in c['sources']} for c in result['candidates']))

    def test_eligibility_error_does_not_fail_open(self):
        def unavailable(_):
            raise RuntimeError('eligibility service unavailable')
        with self.assertRaises(RuntimeError):
            fuse_candidates([ChannelResult('a', [Candidate(1)])], 1, eligible=unavailable)

    def test_same_item_uses_one_eligibility_snapshot(self):
        calls = []
        def eligible(item):
            calls.append(item)
            return True
        fuse_candidates([ChannelResult('a', [Candidate(1)]), ChannelResult('b', [Candidate(1)])], 1, eligible=eligible)
        self.assertEqual(calls, [1])

    def test_impossible_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_candidates([ChannelResult('a', [], min_quota=2)], 1)
        with self.assertRaises(ValueError):
            fuse_candidates([ChannelResult('a'), ChannelResult('a')], 1)
