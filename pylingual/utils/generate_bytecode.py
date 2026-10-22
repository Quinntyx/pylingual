#!/usr/bin/env python3

import functools
import subprocess
import sys
import py_compile
import platform
import os
import re
import shutil


from pylingual.utils.version import PythonVersion


UV_VERSIONS = {PythonVersion((3, x)) for x in range(8, 16)}


class CompileError(Exception):
    success = False
    bc_a = None


class PyenvError(Exception):
    pass


def _compile_native(py_file: str, out_file: str):
    try:
        py_compile.compile(py_file, cfile=out_file, doraise=True, optimize=0)
    except py_compile.PyCompileError as e:
        raise CompileError(str(e))
    return


@functools.lru_cache(maxsize=None)
def _uv_python(version: PythonVersion) -> str | None:
    """Resolve the interpreter uvx would use, so repeated compiles can skip uvx's per-call resolution."""
    try:
        found = subprocess.run(["uv", "python", "find", version.as_str()], shell=False, capture_output=True, text=True)
        path = found.stdout.strip()
        if found.returncode != 0 or not path:
            return None
        # confirm the version before trusting it, otherwise a bad resolve turns every compile into a CompileError
        check = subprocess.run([path, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], shell=False, capture_output=True, text=True)
        return path if check.stdout.strip() == version.as_str() else None
    except Exception:
        return None


def _compile_uv(py_file: str, out_file: str, version: PythonVersion):
    compile_cmd = f"import py_compile, sys; assert sys.version_info[:2] == {version.as_tuple()!r}; py_compile.compile({py_file!r}, cfile={out_file!r}, doraise=True)"

    python = _uv_python(version)
    cmd = [python, "-c", compile_cmd] if python else ["uvx", "--no-config", "--python", version.as_str(), "python", "-c", compile_cmd]

    output = subprocess.run(cmd, shell=False, capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})

    # Ignore stderr messages from uv downloading versions on demand
    stderr = re.sub(r"\s*Download(ing|ed)\s.+\n", "", output.stderr)
    # Ignore stderr messages from uv logging that compilation occurred
    stderr = re.sub(r"\s*Bytecode compiled \d+ files? in \d+ms\n", "", stderr)
    # uv can warn about a prerelease interpreter even when compilation succeeds.
    # Such warnings must not turn every 3.15 verification into a false failure.
    if output.returncode != 0:
        # Keep only the source diagnostic, not the compiler wrapper traceback:
        # error correction interprets the first reported line as a source line.
        stderr = stderr.rsplit("py_compile.PyCompileError: ", 1)[-1]
        raise CompileError(stderr or output.stdout or f"Compiler exited with status {output.returncode}")


def _compile_pyenv(py_file: str, out_file: str, version: PythonVersion):
    which_pyenv = shutil.which("pyenv")
    if not which_pyenv:
        raise PyenvError(f"Could not find pyenv installation to compile in version {version.as_str()}. Try running with --init-pyenv to enable verification for end-of-life Python versions.")

    version_win = None
    if platform.system() == "Windows":  # workaround for pyenv-win being bugged when passing versions like 3.x not 3.x.y

        def try_get_version_patch(version_str) -> int | None:
            try:
                return int(version_str.split(".", 2)[2])
            except IndexError:
                return None
            except ValueError:
                return None

        pyenv_versions_cmd = [which_pyenv, *"versions --bare".split()]
        pyenv_versions_output = subprocess.run(pyenv_versions_cmd, shell=False, capture_output=True, text=True)
        if pyenv_versions_output.stderr:
            raise CompileError(pyenv_versions_output.stderr)
        # get lastest pyenv version for the correct major.minor version
        pyenv_version_strings = list(filter(lambda x: x.startswith(f"{version.as_str()}"), pyenv_versions_output.stdout.splitlines()))
        if not any(x == version.as_str() for x in pyenv_version_strings):
            pyenv_real_versions = list(filter(lambda x: x is not None, map(try_get_version_patch, pyenv_version_strings)))
            if len(pyenv_real_versions) == 0:
                raise PyenvError(f"Could not find pyenv version for {version.as_str()}")
            version_win = f"{version.as_str()}.{max(pyenv_real_versions)}"

    compile_cmd = f"import py_compile, sys; assert sys.version_info[:2] == {version.as_tuple()!r}; py_compile.compile({py_file!r}, cfile={out_file!r})"

    cmd = [which_pyenv, *"exec python -c".split(), compile_cmd]

    output = subprocess.run(cmd, shell=False, capture_output=True, text=True, env={**os.environ, "PYENV_VERSION": version_win if version_win else version.as_str(), "PYTHONWARNINGS": "ignore"})

    if output.stderr:
        raise CompileError(output.stderr)


def _compile_gateway(py_file: str, out_file: str, version: PythonVersion, port: int):
    """Compile through an explicitly configured loopback/SSH compile gateway."""
    import base64
    import json
    from pathlib import Path
    import socket
    import tokenize

    with tokenize.open(py_file) as source_file:
        source = source_file.read()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        stream = connection.makefile("rwb")
        stream.write((json.dumps({"version": version.as_str(), "source": source, "filename": py_file}) + "\n").encode())
        stream.flush()
        line = stream.readline(16000000)
        if not line or not line.endswith(b"\n"):
            raise RuntimeError("Incomplete compilation gateway response")
        response = json.loads(line)
    if not response.get("success"):
        raise CompileError(response.get("error", "Compilation gateway failed"))
    bytecode = base64.b64decode(response["pyc_b64"], validate=True)
    if len(bytecode) < 16:
        raise RuntimeError("Invalid compilation gateway bytecode")
    Path(out_file).write_bytes(bytecode)


def compile_version(py_file, out_file, version):
    py_file = str(py_file)
    out_file = str(out_file)
    version = PythonVersion(version)

    gateway_port = os.environ.get("PYLINGUAL_COMPILE_GATEWAY_PORT")
    if gateway_port:
        _compile_gateway(py_file, out_file, version, int(gateway_port))
    elif version == sys.version_info:
        _compile_native(py_file=py_file, out_file=out_file)
    elif version in UV_VERSIONS:
        _compile_uv(py_file=py_file, out_file=out_file, version=version)
    else:
        _compile_pyenv(py_file=py_file, out_file=out_file, version=version)
