"""Select a version-balanced subset of the tokenizer corpus for backbone training.

Within each (version, corpus) cell, files are ordered by sha256(f"{seed}:{canonical_sha256}") and
the first quotas(files) of them are kept, so the subset keeps the exact 90/10 PyPI/CSN mix and the
uniform 3.6-3.14 balance of prepare_tokenizer_corpus. The output is a data directory (journal plus
manifest) that prepare_full_training_data reads like the full corpus; chunks are not copied.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from dev_scripts.prepare_tokenizer_corpus import quotas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Complete tokenizer corpus data directory")
    parser.add_argument("--hub-record", type=Path, required=True, help="hub.json of the published corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    source = json.loads((args.corpus/"manifest.json").read_text())
    entries = [json.loads(line) for line in (args.corpus/"accepted.jsonl").read_text().splitlines()]
    if not source.get("complete") or len(entries) != source["files"]:
        raise ValueError("Source corpus is incomplete")
    wanted = quotas(args.files)
    cells = defaultdict(list)
    for entry in entries:
        cells[(entry["version"], entry["corpus"])].append(entry)
    chosen = []
    for version, counts in wanted.items():
        for corpus, count in counts.items():
            ranked = sorted(cells[(version, corpus)],
                            key=lambda entry: hashlib.sha256(f"{args.seed}:{entry['canonical_sha256']}".encode()).hexdigest())
            if len(ranked) < count:
                raise ValueError(f"Only {len(ranked)} {corpus} files for {version}, need {count}")
            chosen.extend(ranked[:count])
    args.output.mkdir(parents=True, exist_ok=False)
    journal = "".join(json.dumps(entry)+"\n" for entry in chosen)
    (args.output/"accepted.jsonl").write_text(journal)
    manifest = {"complete": True, "files": len(chosen), "versions": list(wanted), "counts": wanted,
                "seed": args.seed, "mix_unit": "unique source files", "heldout_315": True,
                "subset_of": {**json.loads(args.hub_record.read_text()), "files": source["files"],
                              "journal_sha256": hashlib.sha256((args.corpus/"accepted.jsonl").read_bytes()).hexdigest()},
                "selection": "per (version, corpus) cell, lowest sha256(f'{seed}:{canonical_sha256}') first, quotas(files)",
                "exclusions_sha256": source["exclusions_sha256"],
                "statement_rows": sum(entry["statement_rows"] for entry in chosen),
                "segmentation_rows": sum(entry["segmentation_rows"] for entry in chosen)}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({key: manifest[key] for key in ("files", "statement_rows", "segmentation_rows")}), flush=True)


if __name__ == "__main__":
    main()
