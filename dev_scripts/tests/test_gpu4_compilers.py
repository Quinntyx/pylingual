import unittest
from pathlib import Path

from dev_scripts.run_gpu4_tokenizer_preparation import compiler_command, verify_runtime


class GPU4CompilerTests(unittest.TestCase):
    def test_container_uses_local_stdio_and_never_pulls(self):
        argv = compiler_command(Path('/experiment/repo'), 'sha256:pinned', 'owner', 123, 456)
        self.assertEqual(argv[:3], ['docker', 'run', '--rm'])
        self.assertIn('--pull=never', argv)
        self.assertIn('--network=none', argv)
        self.assertIn('--read-only', argv)
        self.assertIn('123:456', argv)
        self.assertEqual(argv[-2:], ['sha256:pinned', 'python'])
        self.assertNotIn('ssh', argv)
        self.assertNotIn('--privileged', argv)

    def test_identical_native_runtime_is_accepted(self):
        expected = {'version': '3.14.6 host-build', 'magic': 'test', 'opcodes': ['LOAD_CONST', 'RETURN_VALUE']}
        actual = {**expected, 'version': '3.14.6 container-build', 'opcodes': list(reversed(expected['opcodes']))}
        verify_runtime('3.14', expected, actual)

    def test_runtime_drift_is_rejected(self):
        expected = {'version': '3.14.6 build', 'magic': 'test', 'opcodes': ['LOAD_CONST']}
        for change in ({'version': '3.14.7 build'}, {'magic': 'changed'}, {'opcodes': ['NEW_OPCODE']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_runtime('3.14', expected, {**expected, **change})


if __name__ == '__main__':
    unittest.main()
