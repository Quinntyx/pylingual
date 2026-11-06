"""Persistent JSON-line compile server, compatible with Python 3.6+.

Compilation only: no imports or execution of any sampled package code.
"""
import base64
import importlib.util
import io
import json
import marshal
import opcode
import struct
import sys
import warnings
import traceback


def main():
    print(json.dumps({'ready': True, 'version': sys.version, 'magic': importlib.util.MAGIC_NUMBER.hex(), 'opcodes': sorted(name for name, value in opcode.opmap.items() if 0 <= value < 256)}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            source = request['source']
            if len(source.encode('utf-8')) > 3000000:
                raise ValueError('Compilation request too large')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                code = compile(source, request.get('filename', '<sample>'), 'exec', dont_inherit=True, optimize=0)
            if sys.version_info >= (3, 7):
                header = importlib.util.MAGIC_NUMBER + struct.pack('<III', 0, 0, len(source.encode('utf-8')))
            else:
                header = importlib.util.MAGIC_NUMBER + struct.pack('<II', 0, len(source.encode('utf-8')))
            response = {'success': True, 'pyc_b64': base64.b64encode(header + marshal.dumps(code)).decode('ascii')}
        except Exception as exc:
            response = {'success': False, 'error': ''.join(traceback.format_exception_only(type(exc), exc))[:2000]}
        print(json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
