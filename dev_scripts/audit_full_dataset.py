"""Audit generated sample provenance and paired-task splits, without model inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reserved', type=Path, required=True)
    args = parser.parse_args()
    reserved = set(json.loads(args.reserved.read_text()))
    root = json.loads((args.data/'manifest.json').read_text())
    versions = ['3.'+str(v) for v in range(6,16)]
    assert root['versions'] == versions
    assert root['complete'] and root['examples'] == 50000
    all_rows, files, fingerprints = [], [], {}
    for subset in ('backbone','adapter315','control314','control315'):
        base = args.data/subset
        manifest = json.loads((base/'manifest.json').read_text())
        metadata = {row['canonical_sha256']: row for row in manifest['files']}
        assert len(metadata) == len(manifest['files']) == manifest['examples']
        rows_by_task = {}
        for task in ('statement','segmentation'):
            rows = []
            for split in ('train','validation'):
                path = base/(task+'-'+split+'.jsonl')
                fingerprints[str(path.relative_to(args.data))] = hashlib.sha256(path.read_bytes()).hexdigest()
                for line in path.open():
                    row = json.loads(line)
                    assert metadata[row['canonical_sha256']]['split'] == split
                    assert (int(row['canonical_sha256'][:8],16)%10 == 0) == (split == 'validation')
                    assert row['version'] in manifest['versions']
                    assert row['source_sha512'] not in reserved and row['source_id'] not in reserved
                    assert row['corpus'] in ('pypi','csn')
                    rows.append(row)
            assert len(rows) == len(metadata)
            assert len({row['canonical_sha256'] for row in rows}) == len(rows)
            rows_by_task[task] = {row['canonical_sha256']:row for row in rows}
        assert rows_by_task['statement'].keys() == rows_by_task['segmentation'].keys()
        for identity in rows_by_task['statement']:
            for field in ('canonical_sha256','source_sha512','version','corpus'):
                assert rows_by_task['statement'][identity][field] == rows_by_task['segmentation'][identity][field]
        if subset in ('backbone','adapter315'):
            all_rows.extend(rows_by_task['statement'].values())
            files.extend(manifest['files'])
        else:
            expected = {row['canonical_sha256']:row for row in all_rows if row['version'] == manifest['versions'][0]}
            assert rows_by_task['statement'] == expected
    assert len(all_rows) == len({row['canonical_sha256'] for row in all_rows}) == len({row['source_sha512'] for row in all_rows}) == 50000
    splits = {row['canonical_sha256']:row['split'] for row in files}
    for version in versions:
        for split, expected in [('train',{'pypi':4050,'csn':450}),('validation',{'pypi':450,'csn':50})]:
            actual = dict(Counter(row['corpus'] for row in all_rows if row['version']==version and splits[row['canonical_sha256']]==split))
            assert actual == expected, (version,split,actual)
    assert set(json.loads((args.data/'backbone/manifest.json').read_text())['versions']) == set(versions[:-1])
    report = {'passed':True,'unique_canonical_sources':50000,'unique_raw_sources':50000,'versions':versions,'source_overlap_with_reserved_benchmark_ids':0,'train_validation_overlap':0,'exact_per_version_mix':True,'paired_task_identity':True,'controls_match_version_data':True,'backbone_has_315':False,'data_sha256':fingerprints}
    (args.data/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='data_sha256'}),flush=True)


if __name__ == '__main__':
    main()
