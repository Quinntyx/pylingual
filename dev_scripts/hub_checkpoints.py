"""Mirror resumable Trainer checkpoints to a public Hugging Face model repo (see AGENTS.md).

The repo holds `last-checkpoint/` (weights, optimizer, scheduler, scaler, RNG and trainer state)
plus `run/` (provenance and progress). The final model is pushed to the repo root when training
ends. A host that starts with an empty output directory restores both before resuming, so a dead
host costs at most one checkpoint interval.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from huggingface_hub import HfApi, snapshot_download
from transformers import TrainerCallback

RUN_FILES = ("provenance.json", "progress.json", "resume-history.jsonl", "resume-verification.json", "result.json")


def git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                          cwd=Path(__file__).resolve().parent).stdout.strip()


def ensure_repo(repo_id, card):
    api = HfApi()
    api.create_repo(repo_id, private=False, exist_ok=True)
    if not api.file_exists(repo_id, "README.md"):
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id,
                        commit_message="Describe run")


def restore(repo_id, output):
    """Download the Hub's last checkpoint into output/training when this host has none."""
    training = output/"training"
    if training.exists() and any(training.glob("checkpoint-*")):
        return None
    api = HfApi()
    if not api.repo_exists(repo_id) or not api.file_exists(repo_id, "last-checkpoint/trainer_state.json"):
        return None
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        snapshot_download(repo_id, allow_patterns=["last-checkpoint/*", "run/*"], local_dir=scratch)
        step = json.loads((Path(scratch)/"last-checkpoint/trainer_state.json").read_text())["global_step"]
        training.mkdir(parents=True, exist_ok=True)
        shutil.move(Path(scratch)/"last-checkpoint", training/f"checkpoint-{step}")
        for name in RUN_FILES:
            if (Path(scratch)/"run"/name).exists() and not (output/name).exists():
                shutil.copy(Path(scratch)/"run"/name, output/name)
    print(json.dumps({"restored_from_hub": repo_id, "global_step": step}), flush=True)
    return training/f"checkpoint-{step}"


def stage(output, checkpoint):
    """Hard-link a checkpoint so save_total_limit rotation cannot delete it mid-upload."""
    staging = output/"hub-staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(checkpoint, staging/"last-checkpoint", copy_function=os.link)
    (staging/"run").mkdir()
    for name in RUN_FILES:
        if (output/name).exists():
            shutil.copy(output/name, staging/"run"/name)
    return staging


class HubCheckpointPush(TrainerCallback):
    """Upload saved checkpoints in the background, one at a time; a save made while an upload runs
    is queued, and only the newest queued checkpoint is pushed once the upload finishes."""

    def __init__(self, repo_id, output):
        self.repo_id, self.output, self.api = repo_id, Path(output), HfApi()
        self.future = self.queued = None

    def _record(self, **status):
        (self.output/"hub-status.json").write_text(json.dumps({"repo_id": self.repo_id, **status}, indent=2)+"\n")

    def _collect(self, wait):
        if self.future is None or not (wait or self.future.done()):
            return
        future, self.future = self.future, None
        try:
            commit = future.result()
            self._record(state="pushed", step=self.pending_step, commit=commit.oid)
        except Exception as error:
            # Training continues; the next save retries with a newer checkpoint.
            print(json.dumps({"hub_push_failed": repr(error), "step": self.pending_step}), flush=True)
            self._record(state="failed", step=self.pending_step, error=repr(error))

    def _push(self, checkpoint, step):
        if not checkpoint.is_dir():
            return
        staging = stage(self.output, checkpoint)
        self.pending_step = step
        self.future = self.api.upload_folder(repo_id=self.repo_id, folder_path=staging, run_as_future=True,
                                             delete_patterns=["last-checkpoint/*"],
                                             commit_message=f"Checkpoint at step {step}")

    def _advance(self, wait=False):
        self._collect(wait)
        if self.future is None and self.queued is not None:
            (checkpoint, step), self.queued = self.queued, None
            self._push(checkpoint, step)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.queued = (Path(args.output_dir)/f"checkpoint-{state.global_step}", state.global_step)
            self._advance()

    def on_step_end(self, args, state, control, **kwargs):
        # Record a finished upload as soon as it lands, not only at the next save.
        if state.is_world_process_zero and (self.queued is not None or self.future is not None and self.future.done()):
            self._advance()

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            while self.future is not None or self.queued is not None:
                self._advance(wait=True)


def push_final(repo_id, folder, output, message):
    """Push the finished model to the repo root and the final run records to run/."""
    api = HfApi()
    api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message=message)
    run = Path(tempfile.mkdtemp(dir=output))
    try:
        for name in RUN_FILES:
            if (output/name).exists():
                shutil.copy(output/name, run/name)
        api.upload_folder(repo_id=repo_id, folder_path=run, path_in_repo="run", commit_message=message+" (run records)")
    finally:
        shutil.rmtree(run)
    return api.model_info(repo_id).sha
