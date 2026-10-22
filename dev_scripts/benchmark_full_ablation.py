"""Full, resumable 3.14/3.15 ablation benchmark on the training GPU.

Validation components are diagnostic; the independent complete file benchmark
is the generalization measurement. No benchmark input selects model checkpoints.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

ARMS = ('control', 'no_adapter', 'statement_adapter', 'adapted')
VERSIONS = ('3.14', '3.15')


def arm_paths(models, arm, version):
    control = models / ('control' + version.replace('.', ''))
    statement = control / 'statement' if arm == 'control' else models / 'universal/statement'
    segmentation = control / 'segmentation' if arm in ('control', 'statement_adapter') else models / 'universal/segmentation'
    return {'statement': statement, 'segmentation': segmentation}


def config_for(models, arm, version):
    paths = arm_paths(models, arm, version)
    return {'SHARED_BACKBONES': {
        'STATEMENT_MODEL': str(paths['statement'].resolve()),
        'SEGMENTATION_MODEL': str(paths['segmentation'].resolve()),
        'USE_ADAPTERS': arm not in ('control', 'no_adapter'),
        'USE_STATEMENT_ADAPTERS': arm in ('adapted', 'statement_adapter'),
        'USE_SEGMENTATION_ADAPTERS': arm == 'adapted',
    }}


def selection_for(dataset, data, version):
    manifests = [data / name / 'manifest.json' for name in ('backbone', 'adapter315')]
    excluded = {row[key] for path in manifests for row in json.loads(path.read_text())['files'] for key in ('source_sha512', 'source_id') if key in row}
    listing = dataset / 'pylingualv2' / (version.replace('.', '') + '-pyc-list.txt')
    paths = list(dict.fromkeys(line.strip() for line in listing.read_text().splitlines() if line.strip()))
    selected = []
    for relative in paths:
        file = (dataset / relative).resolve()
        if not file.is_relative_to(dataset.resolve()) or not file.is_file():
            raise ValueError('Missing or unsafe benchmark path: ' + relative)
        if file.parent.name in excluded:
            raise ValueError('Benchmark overlaps sampled sources: ' + relative)
        selected.append({'version': version, 'relative': relative, 'source_sha512': file.parent.name, 'pyc_sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    return {'versions': [version], 'files_per_version': len(selected), 'full_listing': str(listing), 'source_overlap_with_training': 0, 'training_manifest_sha256': hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(), 'files': selected}


def save_consistent(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Refusing to resume with changed experiment inputs: ' + str(path))
    path.write_text(json.dumps(value, indent=2) + '\n')


def experiment_fingerprints(args, versions):
    repo = Path(__file__).resolve().parents[1]
    runtime = sorted((repo / 'pylingual').rglob('*.py'))
    runtime.extend(repo / 'dev_scripts' / name for name in ('benchmark_full_ablation.py', 'evaluate_shared.py', 'shared_training.py'))
    artifacts = {}
    for version in versions:
        for family in arm_paths(args.models, args.arm, version).values():
            candidates = [family / 'family.json']
            for child in ('backbone', 'tokenizer', 'adapters'):
                candidates.extend(path for path in (family / child).rglob('*') if path.is_file())
            for path in candidates:
                relative = str(path.relative_to(args.models))
                if relative not in artifacts:
                    with path.open('rb') as stream:
                        artifacts[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'runtime_sha256': {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime}, 'model_artifact_sha256': artifacts, 'data_manifest_sha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(), 'compiler_provenance_sha256': hashlib.sha256((args.models.parent / 'benchmark-compiler-provenance.json').read_bytes()).hexdigest()}


def run_unit(args):
    import torch
    from dev_scripts.evaluate_shared import component_metrics, file_metrics
    from pylingual.models import load_models
    from pylingual.utils.version import PythonVersion
    torch.set_num_threads(4)
    output = args.output / args.arm / args.version
    output.mkdir(parents=True, exist_ok=True)
    selection = selection_for(args.dataset, args.data, args.version)
    save_consistent(output / 'selection.json', selection)
    config = output / 'runtime.json'
    save_consistent(config, config_for(args.models, args.arm, args.version))
    paths = arm_paths(args.models, args.arm, args.version)
    provenance = {'arm': args.arm, 'version': args.version, 'timeout_seconds': args.timeout, 'validation_component_rows': 500, 'independent_file_benchmark_cases': len(selection['files']), 'families': {task: hashlib.sha256((path / 'family.json').read_bytes()).hexdigest() for task, path in paths.items()}}
    provenance['fingerprints'] = experiment_fingerprints(args, [args.version])
    save_consistent(output / 'provenance.json', provenance)
    families = output / 'families'
    families.mkdir(exist_ok=True)
    for task, path in paths.items():
        link = families / task
        if not link.exists():
            link.symlink_to(path.resolve(), target_is_directory=True)
        if link.resolve() != path.resolve():
            raise ValueError('Family link changed')
    settings = SimpleNamespace(output=output, versions=[args.version], families=families, data=args.data / 'backbone', adapter_data=args.data / 'adapter315', disable_adapters=args.arm in ('control', 'no_adapter'), disable_statement_adapters=args.arm not in ('adapted', 'statement_adapter'), disable_segmentation_adapters=args.arm != 'adapted', component_rows=500, config=config, dataset=args.dataset, timeout=args.timeout, seed=20261016, resume=True)
    if not (output / 'component_metrics.json').exists():
        component_metrics(settings)
    if not (output / 'summary.json').exists():
        load_models(config_file=config, version=PythonVersion(args.version))
        file_metrics(settings, selection)
    summary = json.loads((output / 'summary.json').read_text())
    results = [json.loads(line) for line in (output / 'file_results.jsonl').read_text().splitlines()]
    if len(results) != len(selection['files']) or len({row['relative'] for row in results}) != len(selection['files']):
        raise RuntimeError('Incomplete file benchmark')
    print(json.dumps({'unit_complete': True, 'arm': args.arm, 'version': args.version, 'summary': summary}), flush=True)


def performance(args):
    import torch
    from dev_scripts.shared_training import collate, encode_row, read_rows
    from pylingual.models import load_models
    from pylingual.shared_models import load_family
    from pylingual.utils.version import PythonVersion
    torch.set_num_threads(4)
    output = args.output / args.arm
    output.mkdir(parents=True, exist_ok=True)
    save_consistent(output / 'performance-provenance.json', experiment_fingerprints(args, VERSIONS))
    if (output / 'performance.json').exists():
        return
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    retained = []
    loads = {}
    for version in VERSIONS:
        config = output / ('performance-' + version + '.json')
        save_consistent(config, config_for(args.models, args.arm, version))
        torch.cuda.synchronize()
        started = time.perf_counter()
        pair = load_models(config_file=config, version=PythonVersion(version))
        torch.cuda.synchronize()
        loads[version] = time.perf_counter() - started
        retained.append(pair)
    unique_parameters = {}
    unique_buffers = {}
    for segmenter, translator in retained:
        for model in (segmenter.model, translator.model):
            for value in model.parameters():
                unique_parameters[value.data_ptr()] = value.numel() * value.element_size()
            for value in model.buffers():
                unique_buffers[value.data_ptr()] = value.numel() * value.element_size()
    gc.collect()
    torch.cuda.empty_cache()
    resident = {'cuda_allocated_bytes': torch.cuda.memory_allocated(), 'cuda_reserved_bytes': torch.cuda.memory_reserved(), 'unique_parameter_bytes': sum(unique_parameters.values()), 'unique_buffer_bytes': sum(unique_buffers.values()), 'distinct_parameter_tensors': len(unique_parameters), 'loaded_versions': list(VERSIONS)}
    timing = {}
    for version in VERSIONS:
        timing[version] = {}
        for task, path in arm_paths(args.models, args.arm, version).items():
            bank, tokenizer, metadata, _ = load_family(path, task, torch.device('cuda:0'))
            enabled = args.arm in ('adapted', 'statement_adapter') if task == 'statement' else args.arm == 'adapted'
            rows = read_rows(args.data / ('control' + version.replace('.', '')) / (task + '-validation.jsonl'))
            examples = [(row, encode_row(task, tokenizer, row)) for row in rows]
            examples = [(row, encoded) for row, encoded in examples if encoded is not None][:8]
            if len(examples) != 8:
                raise RuntimeError('Not enough performance inputs')
            batch = collate([encoded for _, encoded in examples], tokenizer, 'cuda')
            elapsed = []
            generated_tokens = []
            torch.cuda.reset_peak_memory_stats()
            with bank.use(version if enabled else None), torch.inference_mode():
                for iteration in range(12):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    if task == 'statement':
                        result = bank.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=128)
                    else:
                        result = bank.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits.argmax(-1)
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - started
                    token_count = int((result[:, 1:] != tokenizer.pad_token_id).sum().item()) if task == 'statement' else None
                    del result
                    if iteration >= 2:
                        elapsed.append(duration)
                        if token_count is not None:
                            generated_tokens.append(token_count)
            ordered = sorted(elapsed)
            timing[version][task] = {'batch_size': 8, 'warmup_batches': 2, 'measured_batches': 10, 'input_ids': [row['canonical_sha256'] for row, _ in examples], 'input_tokens': batch['attention_mask'].sum().item(), 'input_shape': list(batch['input_ids'].shape), 'batch_seconds_median': statistics.median(elapsed), 'batch_seconds_p95': ordered[min(len(ordered)-1, int(len(ordered)*0.95))], 'examples_per_second': 8 / statistics.mean(elapsed), 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            if generated_tokens:
                timing[version][task].update(generated_tokens_per_batch=generated_tokens, generated_tokens_per_second=sum(generated_tokens) / sum(elapsed))
            del batch
    report = {'arm': args.arm, 'process_cold_load_seconds': loads, 'disk_cache_controlled': False, 'resident_both_versions': resident, 'warm_inference': timing, 'hardware': subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip(), 'torch_version': torch.__version__, 'measurement_note': 'Isolated sequential process; model generation bypasses translation cache. FP32 inference, 8-example batches, 2 warmups then 10 timed repeats. Peaks include both resident version configurations.'}
    (output / 'performance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'performance_complete': True, 'arm': args.arm, 'resident': resident}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--version', choices=VERSIONS)
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()
    if args.performance:
        performance(args)
    elif args.version:
        run_unit(args)
    else:
        parser.error('--version is required for a file benchmark unit')


if __name__ == '__main__':
    main()
