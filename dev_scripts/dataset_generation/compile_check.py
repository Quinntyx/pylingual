"""Run by the dataset's Python version for collect_sources.py: compiles each JSON-encoded source read from stdin and
prints "ok" or the error type, one line each."""

import json
import sys
import warnings

warnings.simplefilter("ignore")
for line in sys.stdin:
    try:
        compile(json.loads(line), "<source>", "exec")
        print("ok", flush=True)
    except BaseException as e:
        print(type(e).__name__, flush=True)
