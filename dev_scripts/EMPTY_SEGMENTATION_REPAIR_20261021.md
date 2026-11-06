# Empty-code-object inference repair — 2026-10-21

During the requested universal statement + universal segmentation Python 3.14
full-file benchmark, valid nested code objects with zero modeled instructions
were serialized as an empty string and incorrectly sent to the strict opcode
binder. This caused `ValueError: Unrecognized instruction field: ''` before
any decompilation-quality check. These errors must not be presented as model
accuracy failures.

## Reproduction and fix

- Official eval v2.0.1 input: `python-3.14/3307d26213e4317bebc32410f9a048bc5ba10d27fb56933d22ef64131dfa255ae1508643bac6a517b26e31850d157117c9e96eb50546d1fd3c7600e1eae7ce69/task_base_model.pyc`.
- SHA-256: `983d00decefd3bd0b2078fe9f55e8da7eab3b7c8bda4181a728dd558742ef6c0`.
- Real input inspection found 43 code objects, including 11 empty requests.
- Three regression cases failed before the repair: mixed empty/nonempty
  objects, all-empty objects, and no code objects. Malformed nonempty requests
  remained correctly rejected.
- `pylingual/decompiler.py::run_segmentation` now sends only nonempty code
  objects to the model, restores the original code-object positions afterward,
  and preserves empty boundary lists. Strict opcode parsing is unchanged.
- The four new regression tests, 14 opcode-registry tests and eight benchmark
  tests pass on syssec-gpu4 (26 total).
- No tokenizer, weights, training process, or training data changed.

## Benchmark integrity

The first partial attempt at `base-base-3.14-20261021` was explicitly stopped
and retained as an invalid infrastructure attempt. Its preliminary counts are
not benchmark results. The corrected run will use a separate output directory,
new code fingerprints, the same complete official file lists and the same
300-second per-file timeout. Do not combine old and corrected rows.

Real GPU replay and corrected-run acceptance are recorded in the experiment
workspace's `base-base-3.14.job.json`; this note alone does not establish their
completion.

## Runtime acceptance

The real GPU replay completed in 155.84 seconds with all 43 code-object
positions aligned, including 11 zero-instruction objects. Verification ran
(34/44 checks equal; the file as a whole was not equal). This demonstrates the
infrastructure repair, not perfect decompilation quality.

Both complete official 3.14 benchmark lists were restarted in the separate
`base-base-3.14-20261021-emptyfix` directory on GPUs 2 and 3.
The original partial attempt remains excluded.
