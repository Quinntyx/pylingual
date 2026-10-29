import json
import math
import os
import pathlib
import time
from datetime import timedelta

import click
import numpy as np
import torch
from ControlFlowConfiguration import ControlFlowConfiguration, parse_control_flow_config_json
from train_cflow import load_cflow_dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from pylingual.control_flow_reconstruction.block_grammar import BlockGrammar, Shape
from pylingual.control_flow_reconstruction.model_view import INPUT_TOKENS, CflowRequest
from pylingual.control_flow_reconstruction.skeleton import parse_ref
from pylingual.control_flow_reconstruction.transition import ALIGN, SPECIAL, STMT, TOKENS, Parse, TransitionConfig, TransitionModel, advance, allowed_row, encode_input, features, gold_actions
from pylingual.models import TransitionPredictor


def build_vocab(texts) -> list[str]:
    """Special and input structure tokens first, then every other token in `texts` but statement references."""
    fixed = [*SPECIAL, *INPUT_TOKENS]
    return fixed + sorted({t for text in texts for t in text.split() if parse_ref(t) is None} - set(fixed))


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Encoder inputs, and the decoder features, allowed actions and action of each step of the gold skeletons."""
    n, length = max(row["n_stmts"] for row in rows), -(-max(row["length"] for row in rows) // ALIGN) * ALIGN
    steps = []
    for row in rows:
        grammar = BlockGrammar([Shape.parse(s) for s in row["shapes"].split()], row["version"])
        parse, walk = Parse(grammar.start()), []
        for action in gold_actions(row["target"].split()):
            walk.append((features(grammar, parse), allowed_row(grammar, parse.state, n), action.stmt if action.token == STMT else n + action.token))
            parse = advance(grammar, parse, action)
        steps.append(walk)
    t = max(map(len, steps))
    batch = {
        "input_ids": np.zeros((len(rows), length), dtype=np.int64),
        "refs": np.full((len(rows), length), -1, dtype=np.int64),
        "attention_mask": np.zeros((len(rows), length), dtype=bool),
        "stmt_pos": np.zeros((len(rows), n), dtype=np.int64),
        "features": np.zeros((len(rows), t, 9), dtype=np.int64),
        "allowed": np.zeros((len(rows), t, n + len(TOKENS)), dtype=bool),
        "labels": np.full((len(rows), t), -100, dtype=np.int64),
    }
    for b, (row, walk) in enumerate(zip(rows, steps)):
        batch["input_ids"][b, : row["length"]] = row["input_ids"]
        batch["refs"][b, : row["length"]] = row["refs"]
        batch["attention_mask"][b, : row["length"]] = True
        batch["stmt_pos"][b, : len(row["stmt_pos"])] = row["stmt_pos"]
        for s, (feats, allowed, label) in enumerate(walk):
            batch["features"][b, s], batch["allowed"][b, s], batch["labels"][b, s] = feats, allowed, label
    return {k: torch.from_numpy(v) for k, v in batch.items()}


def exact_match(predictor: TransitionPredictor, test) -> dict:
    """Share of greedy skeletons equal to the gold skeleton, by statement count."""
    predictions = predictor([CflowRequest.from_row(row) for row in test])
    buckets = {}
    for prediction, row in zip(predictions, test):
        bucket = buckets.setdefault(max(8, 2 ** math.ceil(math.log2(row["n_stmts"]))), [0, 0])
        bucket[0] += 1
        bucket[1] += bool(prediction) and prediction[0] == row["target"].split()
    return {
        "exact_match": sum(c for _, c in buckets.values()) / max(len(test), 1),
        "predicted": sum(map(bool, predictions)) / max(len(test), 1),
        "by_n_stmts": {f"<={b}": c / n for b, (n, c) in sorted(buckets.items())},
    }


def train_transition_model(config: ControlFlowConfiguration, public: bool = False):
    dataset = load_cflow_dataset(config)
    vocab = build_vocab(dataset["train"]["input"])
    index = {t: i for i, t in enumerate(vocab)}

    def tokenize(row):
        ids, refs, stmt_pos = encode_input(row["input"].split(), index)
        return {"input_ids": ids, "refs": refs, "stmt_pos": stmt_pos, "length": len(ids)}

    tokenized = dataset.map(tokenize, remove_columns=["input", "file"])

    params = config.model_parameters
    model = TransitionModel(TransitionConfig(
        vocab=vocab, d_model=params.d_model, layers=params.layers, heads=params.heads, d_ff=params.d_ff,
        max_stmts=config.max_stmts, max_input_tokens=config.max_input_tokens,
    ))  # fmt: skip

    model_dir = config.model_dir
    train_args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=config.training_parameters.batch_size,
        per_device_eval_batch_size=config.training_parameters.batch_size,
        learning_rate=config.training_parameters.learning_rate,
        num_train_epochs=config.training_parameters.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # emulated bf16 has no memory efficient attention kernel
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(including_emulation=False),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        prediction_loss_only=True,
        save_total_limit=2,
        train_sampling_strategy="group_by_length",
        remove_unused_columns=False,  # the collator walks the grammar over the gold skeleton
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        report_to="tensorboard",
        logging_steps=100,
        ddp_find_unused_parameters=False,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.model_repo_name,
        hub_private_repo=not public,
    )
    trainer = Trainer(model=model, args=train_args, data_collator=collate, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"])

    start = time.time()
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(model_dir)) if model_dir.is_dir() else None)
    duration = str(timedelta(seconds=time.time() - start))

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        trainer.save_model(str(model_dir))
        report = exact_match(TransitionPredictor(trainer.model, trainer.model.device), dataset["test"])
        (model_dir / "eval.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if config.push_to_hub:
            trainer.push_to_hub(commit_message=duration)


@click.command(help="Training script for the transition control flow model given a control flow json.")
@click.argument("json_path", type=str)
@click.option("--public", is_flag=True, default=False, help="Make uploaded HuggingFace repos public (default: private)")
def main(json_path: str, public: bool):
    control_flow_config = parse_control_flow_config_json(pathlib.Path(json_path))
    train_transition_model(control_flow_config, public)


if __name__ == "__main__":
    main()
