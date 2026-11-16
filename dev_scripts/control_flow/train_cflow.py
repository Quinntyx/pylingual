import itertools
import json
import math
import os
import pathlib
import time
from datetime import timedelta

import click
import torch
from ControlFlowConfiguration import ControlFlowConfiguration, parse_control_flow_config_json
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from pylingual.control_flow_reconstruction.model_view import INPUT_TOKENS, CflowRequest
from pylingual.control_flow_reconstruction.skeleton import CLOSE, KEYWORD_TEXT, LOOP, OPEN, PLACE, ref, to_model_order
from pylingual.models import SkeletonPredictor


def build_tokenizer(texts, max_stmts: int) -> PreTrainedTokenizerFast:
    """Word-level vocabulary: special, skeleton and input structure tokens first, then every other token in `texts`."""
    fixed = ["<pad>", "</s>", "<unk>", OPEN, CLOSE, LOOP, PLACE, *KEYWORD_TEXT, *INPUT_TOKENS, *map(ref, range(max_stmts))]
    vocab = {}
    for token in fixed + sorted({t for text in texts for t in text.split()}):
        vocab.setdefault(token, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", vocab["</s>"])])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", model_input_names=["input_ids", "attention_mask"])


def load_cflow_dataset(config: ControlFlowConfiguration):
    """The train/valid/test JSONL splits of every dataset dir, without the code objects too large for the model."""
    splits = {"train": "train", "validation": "valid", "test": "test"}
    dataset = load_dataset("json", data_files={split: [str(d / f"{name}.jsonl") for d in config.dataset_dirs] for split, name in splits.items()})
    return dataset.filter(lambda row: row["n_stmts"] <= config.max_stmts and len(row["input"].split()) < config.max_input_tokens)


def exact_match(model, tokenizer, test) -> dict:
    """Share of greedy skeletons equal to the gold skeleton, by statement count."""
    predictions = SkeletonPredictor(model, tokenizer, model.device)([CflowRequest.from_row(row) for row in test])
    buckets = {}
    for prediction, row in zip(predictions, test):
        bucket = buckets.setdefault(max(8, 2 ** math.ceil(math.log2(row["n_stmts"]))), [0, 0])
        bucket[0] += 1
        bucket[1] += bool(prediction) and prediction[0] == row["target"].split()
    return {
        "exact_match": sum(c for _, c in buckets.values()) / max(len(test), 1),
        "by_n_stmts": {f"<={b}": c / n for b, (n, c) in sorted(buckets.items())},
    }


def train_control_flow_model(config: ControlFlowConfiguration, public: bool = False):
    dataset = load_cflow_dataset(config)
    tokenizer = build_tokenizer(itertools.chain(dataset["train"]["input"], dataset["train"]["target"]), config.max_stmts)

    def tokenize(batch):
        encoded = tokenizer(batch["input"])
        encoded["labels"] = tokenizer([" ".join(to_model_order(target.split())) for target in batch["target"]])["input_ids"]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    params = config.model_parameters
    model_config = T5Config(
        vocab_size=len(tokenizer), d_model=params.d_model, d_kv=params.d_model // params.heads, d_ff=params.d_ff, num_layers=params.layers, num_decoder_layers=params.layers,
        num_heads=params.heads, relative_attention_num_buckets=64, relative_attention_max_distance=512, dropout_rate=0.1, feed_forward_proj="gated-gelu", tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.pad_token_id,
    )  # fmt: skip
    # read back by pylingual.models.SkeletonPredictor
    model_config.max_stmts = config.max_stmts
    model_config.max_input_tokens = config.max_input_tokens
    model_config.skeleton_keywords = sorted({t for target in dataset["train"]["target"] for t in target.split() if t in KEYWORD_TEXT})
    model = T5ForConditionalGeneration(model_config)

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
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        train_sampling_strategy="group_by_length",
        report_to="tensorboard",
        logging_steps=100,
        ddp_find_unused_parameters=False,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.model_repo_name,
        hub_private_repo=not public,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(model_dir)) if model_dir.is_dir() else None)
    duration = str(timedelta(seconds=time.time() - start))

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        trainer.save_model(str(model_dir))
        report = exact_match(trainer.model.eval(), tokenizer, dataset["test"])
        (model_dir / "eval.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if config.push_to_hub:
            trainer.push_to_hub(commit_message=duration)


@click.command(help="Training script for the control flow model given a control flow json.")
@click.argument("json_path", type=str)
@click.option("--public", is_flag=True, default=False, help="Make uploaded HuggingFace repos public (default: private)")
def main(json_path: str, public: bool):
    control_flow_config = parse_control_flow_config_json(pathlib.Path(json_path))
    train_control_flow_model(control_flow_config, public)


if __name__ == "__main__":
    main()
