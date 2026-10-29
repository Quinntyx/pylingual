import json
import sys
from pathlib import Path

from dataset_generation.DatasetDescription import get_dataset_description_from_arg_json
from dataset_generation.collect_sources import collect_sources
from pylingual.utils.get_logger import get_logger

VERSION = list(sys.version_info[:2])  # compiled with this interpreter


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def source(n: int) -> str:
    return f"def f{n}(x):\n    return x + {n}\n"


def fake_nas(tmp_path: Path) -> tuple[Path, list[str]]:
    """A NAS with 12 PyPI and 4 CSN train sources, a held-out one, and an eval checkout; see test_collect."""
    nas, ids = tmp_path / "nas", [f"{n:0128x}" for n in range(12)]
    texts = {n: source(n) for n in range(12)} | {1: "x = 1\n", 2: "def f(:\n    pass\n", 3: source(4), 6: source(106)}
    for n, identity in enumerate(ids):
        write(nas / "py312-pylingual-v1" / "code" / "train" / f"pypi-{identity}" / "m.py", texts[n])
    write(nas / "py311-pylingual-v1" / "code" / "test" / f"pypi-{ids[0]}" / "m.py", texts[0])  # held out
    for n in range(4):
        write(nas / "py39-pylingual-v1" / "code" / "train" / f"csn-{n}" / "c.py", source(50 + n))
    return nas, ids


def describe(tmp_path: Path, nas: Path, name: str) -> Path:
    path = tmp_path / f"{name}.json"
    collect = {"nas_root": str(nas), "index": str(tmp_path / "index.json.gz"), "exclude": [{"repo": "unused", "ref": "v1"}], "seed": 7, "min_bytes": 10}
    requests = [
        {"name": "pypi", "corpus": "pypi", "source_path": str(tmp_path / name / "pypi"), "num_train": 2, "num_test": 1, "num_valid": 1},
        {"name": "csn", "corpus": "csn", "source_path": str(tmp_path / name / "csn"), "num_train": 1, "num_test": 0, "num_valid": 1},
    ]
    path.write_text(json.dumps({"name": name, "version": VERSION, "save_to_dir": str(tmp_path / "datasets"), "huggingface_user": "u", "collect": collect, "data_requests": requests}))
    # the eval release: one source by NAS identity, one by content
    checkout = tmp_path / "datasets" / name / "collect" / "eval-checkouts" / "v1" / "python-3.12"
    write(checkout / f"{5:0128x}" / "a.py", source(5))
    write(checkout / "77" / "b.py", source(106))
    return path


def collected(tmp_path: Path, name: str) -> dict[str, list[str]]:
    collect_dir = tmp_path / "datasets" / name / "collect"
    return {r: sorted(json.loads(line)["identity"] for line in (collect_dir / f"{r}.jsonl").open()) for r in ("pypi", "csn")}


def test_collect(tmp_path):
    nas, ids = fake_nas(tmp_path)
    logger = get_logger("test-collect")
    description = get_dataset_description_from_arg_json(str(describe(tmp_path, nas, "a")), logger)
    collect_sources(description, logger)
    got = collected(tmp_path, "a")
    # held out (0), too small (1), not compiling (2), in the eval release by identity (5) or content (6), and only one of
    # the duplicates 3 and 4
    assert len(got["pypi"]) == 4 and len(got["csn"]) == 2
    assert not {ids[n] for n in (0, 1, 2, 5, 6)} & set(got["pypi"]) and not {ids[3], ids[4]} <= set(got["pypi"])
    assert all((tmp_path / "a" / "pypi" / identity / "m.py").exists() for identity in got["pypi"])
    summary = json.loads((tmp_path / "datasets" / "a" / "collect" / "summary.json").read_text())
    assert summary["collected"] == {"pypi": 4, "csn": 2} and summary["excluded"]["identities"] == 2

    # the same seed collects the same sources, and a collected request is not collected again
    collect_sources(get_dataset_description_from_arg_json(str(describe(tmp_path, nas, "b")), logger), logger)
    assert collected(tmp_path, "b") == got
    collect_sources(description, logger)
    assert collected(tmp_path, "a") == got
