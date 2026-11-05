"""Additional frozen-backbone versions retain the original benchmark defaults and scorer."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev_scripts import benchmark_shared_v1 as benchmark


class SharedBenchmarkVersionsTests(unittest.TestCase):
    def invoke(self, arguments, target):
        with patch('sys.argv', ['benchmark', *arguments]), patch.object(benchmark, target) as called:
            benchmark.main()
        return called

    def test_older_unit_uses_the_existing_scorer(self):
        called = self.invoke(['unit', '--release', 'v1', '--version', '3.6', '--arm', 'no_adapter',
                              '--compilers', '/unused/compilers.json'], 'unit')
        called.assert_called_once()
        self.assertEqual(called.call_args.args[0].version, '3.6')
        self.assertEqual(called.call_args.args[0].timeout, 300)

    def test_untrained_adapter_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(['unit', '--release', 'v1', '--version', '3.6', '--arm', 'statement_adapter',
                         '--compilers', '/unused/compilers.json'], 'unit')
        self.assertEqual(error.exception.code, 2)

    def test_prepare_and_report_accept_explicit_scope(self):
        for command in ('prepare', 'report'):
            called = self.invoke([command, '--versions', '3.6', '3.10', '--arms', 'no_adapter'], command)
            self.assertEqual(called.call_args.args[0].versions, ['3.6', '3.10'])
        self.assertEqual(benchmark.VERSIONS, ('3.14', '3.15'))
        self.assertEqual(benchmark.ARMS, ('no_adapter', 'statement_adapter'))

    def test_checkout_only_materializes_requested_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commit = benchmark.RELEASES['v1'][1]
            commands = []

            def run(command, **options):
                commands.append(command)
                if command[0:2] == ['git', 'clone']:
                    Path(command[-1]).mkdir(parents=True)
                return SimpleNamespace(stdout=commit + '\n')

            with patch.object(benchmark.subprocess, 'run', side_effect=run):
                benchmark.checkout(root, 'v1', versions=['3.6', '3.7'])
            sparse = next(command for command in commands if 'sparse-checkout' in command)
            self.assertEqual(sparse[-2:], ['python-3.6', 'python-3.7'])

    def test_base_only_report_uses_requested_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'results/v1/no_adapter/3.6'
            output.mkdir(parents=True)
            (output / 'complete.json').write_text('{"complete":true}')
            metrics = {'files': 2, 'equal_files': 1, 'file_accuracy': .5,
                       'non_overlapping_source_files': {'files': 2, 'equal_files': 1, 'file_accuracy': .5}}
            (output / 'summary.json').write_text(json.dumps({'3.6': metrics}))
            benchmark.report(SimpleNamespace(root=root, versions=['3.6'], arms=['no_adapter']))
            report = json.loads((root / 'results/report.json').read_text())
            self.assertEqual(set(report['releases']['v1']['versions']), {'3.6'})
            self.assertEqual(report['releases']['v1']['versions']['3.6'], {'no_adapter': metrics})


if __name__ == '__main__':
    unittest.main()
