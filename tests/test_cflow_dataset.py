import sys
from pathlib import Path

import pytest

from dataset_generation.bytecode2cflow import make_examples, render_with_skeletons
from dataset_generation.normalize_source import normalize_source
from pylingual.editable_bytecode import PYCFile
from pylingual.equivalence_check import compare_pyc
from pylingual.utils.generate_bytecode import UV_VERSIONS, CompileError, PyenvError, _uv_python, compile_version
from pylingual.utils.version import PythonVersion

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ["Conditional", "Exception", "Generator", "Loop", "Match", "With", "Python315"]
VERSIONS = [f"3.{minor}" for minor in range(6, 16)]
HOST = PythonVersion(sys.version_info[:2])


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("fixture", FIXTURES)
def test_gold_skeleton_round_trip(fixture, version, tmp_path):
    """The gold skeletons of a fixture render to source that recompiles to equivalent bytecode."""
    version = PythonVersion(version)
    if version != HOST and version in UV_VERSIONS and _uv_python(version) is None:
        pytest.skip(f"Python {version} not available")
    try:
        src = normalize_source((ROOT / "test" / f"{fixture}.py").read_text(), version=version.as_tuple(), replace_docstrings=True)
    except SyntaxError as e:
        pytest.skip(f"host ast cannot parse {fixture} for {version}: {e}")
    py, pyc_path = tmp_path / f"{fixture}.py", tmp_path / f"{fixture}.pyc"
    py.write_text(src)
    try:
        compile_version(py, pyc_path, version)
    except (CompileError, PyenvError) as e:
        pytest.skip(f"{fixture} does not compile in {version}: {e}")

    pyc = PYCFile(str(pyc_path))
    examples = make_examples(pyc, src)
    assert {e["name"] for e in examples} == {bc.name for bc in pyc.iter_bytecodes() if not bc.is_comprehension and any(i.starts_line for i in bc)}
    restored = tmp_path / "restored.py"
    restored.write_text(render_with_skeletons(pyc, src.split("\n"), examples))
    compile_version(restored, restored.with_suffix(".pyc"), version)
    failed = [str(r) for r in compare_pyc(pyc_path, restored.with_suffix(".pyc")) if not r.success]
    assert not failed, restored.read_text()
