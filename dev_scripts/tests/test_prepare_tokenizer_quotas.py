import unittest

from dev_scripts.prepare_tokenizer_corpus import quotas


class TokenizerFileQuotaTests(unittest.TestCase):
    def test_100k_is_exact_file_budget_and_corpus_mix(self):
        result = quotas(100000)
        self.assertEqual(sum(v['pypi'] for v in result.values()), 90000)
        self.assertEqual(sum(v['csn'] for v in result.values()), 10000)
        counts = [sum(v.values()) for v in result.values()]
        self.assertEqual(sum(counts), 100000)
        self.assertEqual(max(counts) - min(counts), 1)

    def test_only_backbone_versions_are_allocated(self):
        self.assertEqual(list(quotas(100000)), [f'3.{n}' for n in range(6, 15)])

    def test_smoke_has_same_exact_mix_per_version(self):
        self.assertTrue(all(v == {'pypi': 9, 'csn': 1} for v in quotas(90).values()))

    def test_inexact_corpus_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            quotas(99)


if __name__ == '__main__':
    unittest.main()
