"""Read-only remote source sampler. Never executes package code."""
import argparse
import base64
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reserved', required=True)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--resume-offset', type=int, default=0)
    args = parser.parse_args()
    reserved = set(json.loads(Path(args.reserved).read_text()))
    rng = random.Random(args.seed + args.resume_offset)
    pools = {}
    for corpus in ('pypi', 'csn'):
        paths = Path('/data2/data2_' + corpus + '_py_list_2023-06-03.txt').read_text().splitlines()
        pools[corpus] = [p.strip() for p in paths if p.strip().endswith('.py')]
        rng.shuffle(pools[corpus])
    seen = set()
    failures = Counter()
    for line in sys.stdin:
        request = json.loads(line)
        corpus, wanted = request['corpus'], int(request['count'])
        seen.update(request.get('exclude_sha512', ()))
        if corpus not in pools or not 1 <= wanted <= 2000:
            raise ValueError('Invalid sampling request')
        emitted = 0
        while pools[corpus] and emitted < wanted:
            candidate = Path(pools[corpus].pop())
            source_id = candidate.parent.name
            if source_id in reserved or candidate.stem in reserved:
                failures['reserved_source'] += 1
                continue
            try:
                resolved = candidate.resolve()
                if not str(resolved).startswith('/data2/'):
                    raise ValueError('Source escaped corpus root')
                size = resolved.stat().st_size
                if not 100 <= size <= 100000:
                    failures['file_size'] += 1
                    continue
                raw = resolved.read_bytes()
                digest = hashlib.sha512(raw).hexdigest()
                if digest in reserved or digest in seen:
                    failures['duplicate_or_reserved_hash'] += 1
                    continue
                seen.add(digest)
            except (OSError, ValueError):
                failures['read'] += 1
                continue
            print(json.dumps({'corpus': corpus, 'source_id': source_id, 'source_path': str(candidate), 'source_sha512': digest, 'raw_b64': base64.b64encode(raw).decode('ascii')}), flush=True)
            emitted += 1
        print(json.dumps({'batch_complete': True, 'emitted': emitted, 'remaining': len(pools[corpus]), 'failures': dict(failures)}), flush=True)


if __name__ == '__main__':
    main()
