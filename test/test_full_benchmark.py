"""Benchmark comparability, source-overlap rejection, and durable resume checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from dev_scripts.benchmark_full_ablation import arm_paths, config_for, save_consistent, selection_for
from dev_scripts.evaluate_shared import file_metrics


class FullBenchmarkChecks(unittest.TestCase):
    def test_each_arm_routes_only_its_declared_intervention(self):
        models = Path('/models')
        expected = {
            'control': ('control314/statement', 'control314/segmentation', False, False),
            'no_adapter': ('universal/statement', 'universal/segmentation', False, False),
            'statement_adapter': ('universal/statement', 'control314/segmentation', True, False),
            'adapted': ('universal/statement', 'universal/segmentation', True, True),
        }
        for arm, (statement, segmentation, stmt_adapter, seg_adapter) in expected.items():
            with self.subTest(arm=arm):
                paths = arm_paths(models, arm, '3.14')
                self.assertEqual(paths['statement'], models / statement)
                self.assertEqual(paths['segmentation'], models / segmentation)
                configuration = config_for(models, arm, '3.14')['SHARED_BACKBONES']
                self.assertEqual(configuration['USE_STATEMENT_ADAPTERS'], stmt_adapter)
                self.assertEqual(configuration['USE_SEGMENTATION_ADAPTERS'], seg_adapter)

    def test_selection_is_complete_and_rejects_training_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dataset, data = root / 'dataset', root / 'data'
            (dataset / 'pylingualv2' / 'fixture').mkdir(parents=True)
            (dataset / 'pylingualv2/fixture/file.pyc').write_bytes(b'pyc bytes')
            (dataset / 'pylingualv2/314-pyc-list.txt').write_text('pylingualv2/fixture/file.pyc\n')
            data.mkdir()
            (data / 'manifest.json').write_text('{}')
            for subset in ('backbone', 'adapter315'):
                (data / subset).mkdir()
                (data / subset / 'manifest.json').write_text(json.dumps({'files': []}))
            selected = selection_for(dataset, data, '3.14')
            self.assertEqual(len(selected['files']), 1)
            (data / 'backbone/manifest.json').write_text(json.dumps({'files': [{'source_id': 'fixture'}]}))
            with self.assertRaisesRegex(ValueError, 'overlaps sampled sources'):
                selection_for(dataset, data, '3.14')

    def test_resume_input_mutation_is_not_silent(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'inputs.json'
            save_consistent(path, {'seed': 1})
            save_consistent(path, {'seed': 1})
            with self.assertRaisesRegex(ValueError, 'changed experiment inputs'):
                save_consistent(path, {'seed': 2})

    def test_file_resume_preserves_finished_cases_and_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            relative = ['one.pyc', 'two.pyc']
            initial = {'version':'3.14', 'relative':relative[0], 'success':True, 'code_objects':1, 'equal_code_objects':1, 'category':'Equal', 'seconds':0.1}
            (root/'file_results.jsonl').write_text(json.dumps(initial)+'\n')
            args = SimpleNamespace(output=root, dataset=root, config=None, timeout=2, versions=['3.14'], resume=True)
            selection = {'files':[{'version':'3.14', 'relative':name} for name in relative]}
            model = SimpleNamespace(parameters=lambda: [])
            pair = (SimpleNamespace(model=model), SimpleNamespace(model=model))
            answer = SimpleNamespace(equivalence_results=[SimpleNamespace(success=True, bc_a=object())])
            with mock.patch('dev_scripts.evaluate_shared.load_models', return_value=pair) as load, mock.patch('dev_scripts.evaluate_shared.decompile', return_value=answer) as decompile, mock.patch('dev_scripts.evaluate_shared.torch.cuda.is_available', return_value=False):
                file_metrics(args, selection)
                first = (root/'file_results.jsonl').read_text()
                file_metrics(args, selection)
                self.assertEqual((root/'file_results.jsonl').read_text(), first)
                self.assertEqual(decompile.call_count, 1)
                self.assertEqual(load.call_count, 1)
            summary = json.loads((root/'summary.json').read_text())['3.14']
            self.assertEqual((summary['files'], summary['equal_files']), (2,2))
