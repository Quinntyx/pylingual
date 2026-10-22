import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from dev_scripts.sample_nas_sources import catalog, read_candidate
from dev_scripts.prepare_tokenizer_corpus import quotas


class NasSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root/'catalog.json.gz'

    def tearDown(self):
        self.temp.cleanup()

    def source(self, snapshot, split, corpus, identity):
        directory = self.root/snapshot/'code'/split/(corpus+'-'+identity)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'source.py').write_text('x = 1\n'*30)
        return directory

    def index(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return catalog(self.root, self.cache)

    def test_global_heldout_exclusion(self):
        identity = 'a'*128
        self.source('py314-pylingual-v4', 'train', 'pypi', identity)
        self.source('py36-pylingual-v1', 'test', 'pypi', identity)
        result = self.index()
        self.assertEqual(result['candidates'], [])
        self.assertIn(identity, result['heldout_identities'])

    def test_315_never_contributes_training_sources(self):
        self.source('py315-pylingual-v2', 'train', 'pypi', 'a'*128)
        self.source('py314-pylingual-v4', 'train', 'csn', 'b'*128)
        result = self.index()
        self.assertEqual([row[0] for row in result['candidates']], ['b'*128])

    def test_315_heldout_sources_are_reserved(self):
        self.source('py315-pylingual-v2', 'valid', 'pypi', 'a'*128)
        self.source('py314-pylingual-v4', 'train', 'pypi', 'a'*128)
        self.assertEqual(self.index()['candidates'], [])

    def test_identity_dedup_and_newest_source_preference(self):
        self.source('py36-pylingual-v1', 'train', 'pypi', 'a'*128)
        newest = self.source('py314-pylingual-v4', 'train', 'pypi', 'a'*128)
        result = self.index()
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['candidates'][0][1], str(newest))

    def test_original_and_transport_hash_are_distinct(self):
        directory = self.source('py314-pylingual-v4', 'train', 'pypi', 'a'*128)
        record = read_candidate(['a'*128, str(directory), 'pypi', '3.14'])
        self.assertEqual(record['source_sha512'], 'a'*128)
        self.assertEqual(record['payload_sha512'], hashlib.sha512((directory/'source.py').read_bytes()).hexdigest())

    def test_numeric_csn_identifiers_are_indexed(self):
        self.source('py314-pylingual-v4', 'train', 'csn', '5241')
        self.assertEqual(self.index()['candidates'][0][0], '5241')

    def test_numeric_csn_uses_actual_payload_sha512(self):
        directory = self.source('py314-pylingual-v4', 'train', 'csn', '5241')
        record = read_candidate(['5241', str(directory), 'csn', '3.14'])
        self.assertEqual(record['nas_identity'], 'csn-5241')
        self.assertEqual(record['source_sha512'], record['payload_sha512'])
        self.assertEqual(len(record['source_sha512']), 128)

    def test_cache_roundtrip(self):
        self.source('py314-pylingual-v4', 'train', 'pypi', 'a'*128)
        result = self.index()
        self.assertEqual(self.index(), result)
        with gzip.open(self.cache, 'rt') as stream:
            self.assertTrue(json.load(stream)['complete'])

    def test_exact_100k_90_10_uniform_quotas(self):
        result = quotas(100000)
        self.assertEqual(sum(row['pypi'] for row in result.values()), 90000)
        self.assertEqual(sum(row['csn'] for row in result.values()), 10000)
        self.assertLessEqual(max(map(lambda row: sum(row.values()), result.values()))-min(map(lambda row: sum(row.values()), result.values())), 1)
        self.assertNotIn('3.15', result)

    def test_ambiguous_source_folder_rejected(self):
        directory = self.source('py314-pylingual-v4', 'train', 'pypi', 'a'*128)
        (directory/'second.py').write_text('x = 2\n'*30)
        self.assertEqual(read_candidate(['a'*128, str(directory), 'pypi', '3.14'])['failure'], 'ambiguous-source-file')


if __name__ == '__main__':
    unittest.main()
