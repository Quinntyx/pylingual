"""Real local TCP compilation gateway checks; no model or Hub downloads."""
import importlib.util
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest import mock

from dev_scripts.run_remote_ablation import Compiler, Gateway, Handler
from pylingual.utils.generate_bytecode import CompileError, compile_version
from pylingual.utils.version import PythonVersion


class CompileGatewayTests(unittest.TestCase):
    def setUp(self):
        script = Path(__file__).resolve().parents[1] / 'dev_scripts/compile_source_server.py'
        self.compiler = Compiler(sys.executable, str(script))
        pool = queue.Queue()
        pool.put(self.compiler)
        self.gateway = Gateway(('127.0.0.1', 0), Handler)
        self.gateway.compiler_pools = {f'{sys.version_info.major}.{sys.version_info.minor}': pool}
        self.thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.thread.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.env = mock.patch.dict(os.environ, {'PYLINGUAL_COMPILE_GATEWAY_PORT': str(self.gateway.server_address[1])})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.gateway.shutdown()
        self.gateway.server_close()
        self.thread.join(timeout=5)
        self.compiler.close()
        self.temporary.cleanup()

    def test_actual_compile_roundtrip(self):
        source, output = self.root / 'source.py', self.root / 'output.pyc'
        source.write_text('def add(a, b):\n    return a + b\n')
        compile_version(source, output, PythonVersion(sys.version_info[:2]))
        self.assertEqual(output.read_bytes()[:4], importlib.util.MAGIC_NUMBER)
        self.assertGreater(len(output.read_bytes()), 16)

    def test_compile_error_keeps_line_diagnostic(self):
        source, output = self.root / 'source.py', self.root / 'output.pyc'
        source.write_text('def invalid(:\n    pass\n')
        with self.assertRaises(CompileError) as caught:
            compile_version(source, output, PythonVersion(sys.version_info[:2]))
        self.assertIn('line 1', str(caught.exception))
        self.assertIn('SyntaxError', str(caught.exception))
        self.assertFalse(output.exists())

    def test_gateway_can_compile_non_utf8_source_file(self):
        source, output = self.root / 'source.py', self.root / 'output.pyc'
        source.write_bytes(b'# coding: latin-1\nvalue = "caf\xe9"\n')
        compile_version(source, output, PythonVersion(sys.version_info[:2]))
        self.assertTrue(output.is_file())
