"""Run one approved training lane on a single-GPU host that has only the Hub (no NAS).

Every input comes from the pinned repos and revisions in the plan JSON. Every stage mirrors its
checkpoints to its own Hub repo, so rerunning the same lane on any host continues where the last
host stopped. Stages:
  statement     universal CodeT5 statement backbone
  segmentation  universal RoBERTa MLM pretraining, then the B/I/E segmentation backbone
  lora-3.14 / lora-3.15
                statement LoRA on the frozen, finished statement backbone
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi, snapshot_download

REPO = Path(__file__).resolve().parents[1]
TRAINING_VERSIONS = [f"3.{minor}" for minor in range(6, 15)]


def fetch(root, source, patterns, repo_type="dataset"):
    """Download pinned files once into root/hub/<repo>; return the local snapshot directory."""
    local = root/"hub"/source["repo_id"].replace("/", "--")
    snapshot_download(source["repo_id"], repo_type=repo_type, revision=source["revision"],
                      allow_patterns=patterns, local_dir=local)
    return local


def encoded(root, plan, path):
    directory = fetch(root, plan["encoded"], [path+"/*", path+"/**"])/path
    record = {**plan["encoded"], "path": path}
    (directory/"hub-source.json").write_text(json.dumps(record, indent=2)+"\n")
    return directory


def tokenizer(root, plan, task):
    return fetch(root, plan["tokenizer"], [task+"/*"], repo_type="model")/task


def final_model(root, repo_id, destination):
    """Materialize a finished stage's final model from its Hub repo (repo root, not checkpoints)."""
    if (destination/"config.json").exists():
        return destination
    api = HfApi()
    if not api.file_exists(repo_id, "run/result.json"):
        raise RuntimeError(f"{repo_id} has no finished model yet")
    revision = api.model_info(repo_id).sha
    snapshot_download(repo_id, revision=revision, local_dir=destination,
                      ignore_patterns=["last-checkpoint/*", "run/*", "README.md", ".gitattributes"])
    (destination/"hub-source.json").write_text(json.dumps({"repo_id": repo_id, "revision": revision}, indent=2)+"\n")
    return destination


def resumable(output, repo_id):
    api = HfApi()
    return (output/"provenance.json").exists() or (api.repo_exists(repo_id) and api.file_exists(repo_id, "run/provenance.json"))


def run(root, name, argv, attempts=3):
    log = root/"logs"/(name+".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONPATH": str(REPO), "TOKENIZERS_PARALLELISM": "false"}
    for attempt in range(1, attempts+1):
        command = argv()
        with log.open("a") as stream:
            stream.write(json.dumps({"lane_attempt": attempt, "argv": command, "time": time.time()})+"\n")
            stream.flush()
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=REPO, env=environment).returncode
        if code == 0:
            return
        # Every retry resumes from the newest local or Hub checkpoint.
        time.sleep(60)
    raise RuntimeError(f"{name} failed {attempts} times; see {log}")


def universal(root, plan, task, data, tokenizer_dir, initialization=None, smoke=0):
    stage = plan["stages"][task]
    output = root/"universal"/task
    if (output/"result.json").exists() and json.loads((output/"result.json").read_text()).get("hub"):
        return output

    def argv():
        command = [sys.executable, "-u", "-m", "dev_scripts.train_universal_models", "--task", task,
                   "--data", str(data), "--tokenizer", str(tokenizer_dir), "--output", str(output),
                   "--batch-size", str(stage["batch_size"]), "--accumulation", str(stage["accumulation"]),
                   "--save-steps", str(stage["save_steps"]), "--workers", str(stage.get("workers", 2)),
                   "--hub-repo", stage["hub_repo"]]
        if initialization is not None:
            command += ["--initialization", str(initialization)]
        if smoke:
            command += ["--smoke-only", "--steps", str(smoke)]
        if resumable(output, stage["hub_repo"]):
            command.append("--resume")
        return command
    run(root, task, argv)
    return output


def family(root, plan):
    """Package the finished statement backbone as the frozen format-2 family adapters attach to."""
    from dev_scripts.publish_shared_backbone import publish
    destination = root/"families/statement"
    if (destination/"family.json").exists() and json.loads((destination/"family.json").read_text()).get("complete"):
        return destination
    checkpoint = final_model(root, plan["stages"]["statement"]["hub_repo"], root/"universal/statement/final")
    publish(checkpoint, tokenizer(root, plan, "statement"), "statement", destination, training_versions=TRAINING_VERSIONS)
    metadata = json.loads((destination/"family.json").read_text())
    metadata["hub"] = json.loads((checkpoint/"hub-source.json").read_text())
    (destination/"family.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return destination


def lora(root, plan, version, smoke=0):
    stage = plan["stages"]["lora-"+version]
    base = family(root, plan)
    pools = fetch(root, plan["adapter_corpus"], [f"{version}/*/accepted.jsonl", f"{version}/*/manifest.json"])/version
    train = encoded(root, plan, f"adapter/{version}/statement/train")
    validation = encoded(root, plan, f"adapter/{version}/statement/validation")
    attempts = iter(range(1, 1000))

    def argv():
        # The adapter trainer needs a fresh output each attempt; it restores the Hub checkpoint itself.
        output = root/"lora"/f"{version}-attempt{next(attempts)}"
        while output.exists():
            output = root/"lora"/f"{version}-attempt{next(attempts)}"
        return [sys.executable, "-u", "-m", "dev_scripts.train_shared_adapter", "--family", str(base),
                "--data", str(pools/"train/accepted.jsonl"), "--validation-data", str(pools/"validation/accepted.jsonl"),
                "--encoded-data", str(train), "--encoded-validation", str(validation), "--version", version,
                "--output", str(output), "--epochs", str(stage["epochs"]), "--rank", str(stage["rank"]),
                "--batch-size", str(stage["batch_size"]), "--accumulation", str(stage["accumulation"]),
                "--learning-rate", str(stage["learning_rate"]), "--precision", stage["precision"],
                "--checkpoint-steps", str(stage["save_steps"]), "--seed", str(stage["seed"]),
                "--hub-repo", stage["hub_repo"]]+(["--steps", str(smoke)] if smoke else [])
    run(root, "lora-"+version, argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.home()/"pylingual-runs")
    parser.add_argument("--lane", choices=("statement", "segmentation", "lora-3.14", "lora-3.15"))
    parser.add_argument("--print-lanes", metavar="HOST", help="Print the plan's lanes for HOST and exit")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Pipeline check on smoke data with a throwaway plan only")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.print_lanes:
        print(" ".join(plan["lanes"][args.print_lanes]))
        return
    if args.lane is None:
        parser.error("--lane is required")
    if args.smoke_steps and not plan.get("smoke_only"):
        raise ValueError("--smoke-steps requires a plan marked smoke_only (throwaway repos)")
    if not plan.get("approved") and not args.smoke_steps:
        raise ValueError("Plan is not marked approved; get the user's yes before multi-hour training")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.lane == "statement":
        initialization = fetch(args.root, plan["initialization"], None, repo_type="model")
        universal(args.root, plan, "statement", encoded(args.root, plan, "universal/statement"),
                  tokenizer(args.root, plan, "statement"), initialization, args.smoke_steps)
    elif args.lane == "segmentation":
        data, tokenizer_dir = encoded(args.root, plan, "universal/segmentation"), tokenizer(args.root, plan, "segmentation")
        mlm = universal(args.root, plan, "mlm", data, tokenizer_dir, smoke=args.smoke_steps)
        # Segmentation always starts from the final MLM weights, even on a host that never ran MLM.
        initialization = mlm/"checkpoint" if (mlm/"checkpoint/config.json").exists() else \
            final_model(args.root, plan["stages"]["mlm"]["hub_repo"], mlm/"checkpoint")
        universal(args.root, plan, "segmentation", data, tokenizer_dir, initialization, args.smoke_steps)
    else:
        lora(args.root, plan, args.lane.split("-")[1], args.smoke_steps)
    print(json.dumps({"lane": args.lane, "complete": True}), flush=True)


if __name__ == "__main__":
    main()
