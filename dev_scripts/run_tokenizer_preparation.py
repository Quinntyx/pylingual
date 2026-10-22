"""Coordinate remote-only sampling, native compile RPCs and tokenizer-only fitting.

Run under ssh-agent inside the tmux-jobs workflow. Only source metadata and saved
artifacts live locally; sampled raw files move source-host -> training-host.
Native compile servers see normalized source in RAM, never package execution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import threading

from dev_scripts.run_remote_ablation import Compiler, Gateway, Handler


def run(argv, **kwargs):
    argv = [str(value) for value in argv]
    if argv[0] in ("ssh", "scp"):
        argv[1:1] = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
    return subprocess.run(argv, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--training-host", default="syssec-gpu4")
    parser.add_argument("--source-host", default="syssec-adwaith1")
    parser.add_argument("--nas-root", help="Mounted NAS path on training host; bypass source-host SSH entirely")
    parser.add_argument("--old-training-host", default="pylingual-train")
    parser.add_argument("--old-root", default="/home/henryh/pylingual-shared-full-20261016")
    parser.add_argument("--old-reserved", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--compile-copies", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SSH_AUTH_SOCK"):
        raise RuntimeError("Run under ssh-agent")
    args.workspace.mkdir(parents=True, exist_ok=True)
    run(["ssh-add", Path.home()/".ssh/id_ed25519"], stdin=subprocess.DEVNULL)
    known_hosts = args.workspace/"source-host.known_hosts"
    if not args.nas_root:
        keys = run(["ssh-keygen", "-F", "[syssec-docker0.utdallas.edu]:2227"], capture_output=True, text=True).stdout
        pinned = "\n".join(line for line in keys.splitlines() if line and not line.startswith("#"))+"\n"
        if len(pinned.strip()) < 20:
            raise ValueError("Trusted source-host key missing")
        known_hosts.write_text(pinned)
    exclusion_path = args.workspace/"excluded-source-identities.json"
    if not exclusion_path.exists():
        metadata_script = '''import json,pathlib,sqlite3
root=pathlib.Path(ROOT)
raw,canonical=set(),set()
for line in (root/'data-50k/accepted.jsonl').open():
    p=json.loads(line)['provenance'];raw.add(p['source_sha512']);canonical.add(p['canonical_sha256'])
con=sqlite3.connect('file:'+str(root/'data-lora314-plus50k/examples.sqlite')+'?mode=ro',uri=True)
for digest,identity in con.execute('SELECT source_sha512,canonical_sha256 FROM files'):
    raw.add(digest);canonical.add(identity)
print(json.dumps({'source_sha512':sorted(raw),'canonical_sha256':sorted(canonical)}))
'''.replace("ROOT", repr(args.old_root))
        exported = run(["ssh", "-o", "BatchMode=yes", args.old_training_host, "python3 -"], input=metadata_script, capture_output=True, text=True)
        identities = json.loads(exported.stdout)
        exclusion_path.write_text(json.dumps(identities)+"\n")
    else:
        identities = json.loads(exclusion_path.read_text())
    reserved = (set(json.loads(args.old_reserved.read_text())) | set(identities["source_sha512"])
                | set(identities.get("nas_identities", ())))
    reserved_path = args.workspace/"reserved-benchmark-sources.json"
    reserved_path.write_text(json.dumps(sorted(reserved))+"\n")
    if args.nas_root:
        source_script = args.remote_root+"/repo/dev_scripts/sample_nas_sources.py"
        source_reserved = args.remote_root+"/reserved-benchmark-sources.json"
    else:
        source_home = run(["ssh", "-o", "BatchMode=yes", args.source_host, 'printf %s "$HOME"'], capture_output=True, text=True).stdout.strip()
        if not source_home.startswith("/home/") or "\n" in source_home:
            raise ValueError("Invalid source home")
        source_root = source_home+"/pylingual-tokenizer-100k"
        run(["ssh", args.source_host, "mkdir", "-p", source_root])
        run(["scp", args.repo/"dev_scripts/sample_remote_sources.py", reserved_path, args.source_host+":"+source_root+"/"])
        source_script = source_root+"/sample_remote_sources.py"
        source_reserved = source_root+"/reserved-benchmark-sources.json"
    run(["ssh", args.training_host, "mkdir", "-p", args.remote_root+"/repo/dev_scripts"])
    staged_metadata = [exclusion_path, reserved_path] if args.nas_root else [known_hosts, exclusion_path]
    run(["scp", *staged_metadata, args.training_host+":"+args.remote_root+"/"])
    scripts = ["prepare_tokenizer_corpus.py", "tokenizer_opcode_registry.py", "train_generic_tokenizers.py", "sample_nas_sources.py"]
    run(["scp", *[args.repo/"dev_scripts"/name for name in scripts], args.training_host+":"+args.remote_root+"/repo/dev_scripts/"])
    nested = ["ssh", "-p", "2227", "-l", "henryh@syssec.utdallas.edu", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
              "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile="+args.remote_root+"/source-host.known_hosts", "syssec-docker0.utdallas.edu", "hostname"]
    if not args.nas_root:
        run(["ssh", "-A", "-o", "BatchMode=yes", args.training_host, shlex.join(nested)])
    compilers, pools, metadata = [], {}, {}
    try:
        for minor in range(6, 15):
            version = f"3.{minor}"
            executable = shutil.which("python"+version)
            if minor in (6, 7):
                executable = str(Path.home()/".pyenv/versions"/("3.6.15" if minor == 6 else "3.7.17")/"bin/python")
            if not executable or not Path(executable).is_file():
                raise RuntimeError("Missing existing compiler: "+version)
            pools[version] = queue.Queue()
            for _ in range(args.compile_copies):
                compiler = Compiler(executable, str(args.repo/"dev_scripts/compile_source_server.py"))
                compilers.append(compiler)
                if not compiler.metadata["version"].startswith(version+"."):
                    raise RuntimeError("Compiler version mismatch")
                if not compiler.request({"source": "x = 1\n", "filename": "<tokenizer-preflight>"}).get("success"):
                    raise RuntimeError("Compiler preflight failed")
                pools[version].put(compiler)
            metadata[version] = {"executable": executable, **compiler.metadata, "servers": args.compile_copies}
        provenance_path = args.workspace/"compiler-provenance.json"
        provenance_path.write_text(json.dumps(metadata, indent=2)+"\n")
        run(["scp", provenance_path, args.training_host+":"+args.remote_root+"/compiler-provenance.json"])
        print(json.dumps({"compiler_preflight": True, "servers": len(compilers), "versions": list(metadata),
                          "excluded_raw_files": len(identities["source_sha512"]), "excluded_canonical_files": len(identities["canonical_sha256"]),
                          "model_training_started": False}), flush=True)
        data = args.remote_root+("/data-smoke90" if args.smoke else "/data-100k")
        output = args.remote_root+("/tokenizers-smoke90" if args.smoke else "/tokenizers")
        python = args.remote_root+"/repo/.venv/bin/python"
        environment = ["env", "PYTHONPATH=.", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "RAYON_NUM_THREADS=24", "TOKENIZERS_PARALLELISM=true"]
        with Gateway(("127.0.0.1", 0), Handler) as gateway:
            gateway.compiler_pools = pools
            threading.Thread(target=gateway.serve_forever, daemon=True).start()
            remote_port = 48719
            prep = environment+[python, "-u", "-m", "dev_scripts.prepare_tokenizer_corpus", "--output", data,
                 "--exclusions", args.remote_root+"/excluded-source-identities.json", "--source-script", source_script,
                 "--source-reserved", source_reserved, "--known-hosts", args.remote_root+"/source-host.known_hosts",
                 "--gateway-port", str(remote_port), "--workers", str(args.workers), "--files", "90" if args.smoke else "100000"]
            if args.nas_root:
                prep += ["--nas-root", args.nas_root, "--nas-catalog", args.remote_root+"/nas-catalog.json.gz"]
            command = "cd "+shlex.quote(args.remote_root+"/repo")+" && "+shlex.join(prep)
            try:
                run(["ssh", *([] if args.nas_root else ["-A"]), "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                     "-R", f"127.0.0.1:{remote_port}:127.0.0.1:{gateway.server_address[1]}", args.training_host, command])
            finally:
                gateway.shutdown()
        fit = environment+[python, "-u", "-m", "dev_scripts.train_generic_tokenizers", "--data", data, "--output", output,
             "--template", args.remote_root+"/repo/dev_scripts/statement/tokenizer", "--compiler-provenance", args.remote_root+"/compiler-provenance.json",
             "--vocab-size", "2048" if args.smoke else "30000"]
        run(["ssh", "-o", "BatchMode=yes", args.training_host, "cd "+shlex.quote(args.remote_root+"/repo")+" && "+shlex.join(fit)])
        print(json.dumps({"complete": True, "tokenizers": output, "model_training_started": False}), flush=True)
    finally:
        for compiler in compilers:
            compiler.close()


if __name__ == "__main__":
    main()
