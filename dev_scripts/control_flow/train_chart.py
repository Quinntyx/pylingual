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
from train_cflow import build_tokenizer, load_cflow_dataset
from transformers import T5Config, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from pylingual.control_flow_reconstruction.block_grammar import Shape
from pylingual.control_flow_reconstruction.chart import KEYWORDS, decode_order, gold_items
from pylingual.control_flow_reconstruction.chart_model import ChartParser
from pylingual.control_flow_reconstruction.model_view import STMT_SEP, CflowRequest
from pylingual.control_flow_reconstruction.skeleton import BLOCK_KEYWORDS
from pylingual.models import ChartPredictor

LABELS = ["order", "header_end", "keyword_end", "insert", "loop"]


def chart_labels(row: dict) -> dict:
    """ChartParser.forward labels of a dataset row, in the source order of its target."""
    gold = gold_items(row["target"].split())
    shapes = [Shape.parse(s) for s in row["shapes"].split()]
    n = len(gold.order)
    header_end, loop = [-100] * n, [-100] * n
    for p, e in gold.header_end.items():
        header_end[p] = e
        if shapes[gold.order[p]].opener == "cond":
            loop[p] = int(gold.loop[p])
    insert = [[0] * len(KEYWORDS) for _ in range(n + 1)]
    keyword_end = [[-100] * len(BLOCK_KEYWORDS) for _ in range(n + 1)]
    for g, kw, e in gold.keywords:
        insert[g][KEYWORDS.index(kw)] = 1
        if e is not None:
            keyword_end[g][BLOCK_KEYWORDS.index(kw)] = e
    return {"order": gold.order, "header_end": header_end, "loop": loop, "insert": insert, "keyword_end": keyword_end}


class Collator:
    """Pads inputs with the tokenizer and labels to the batch's statement count."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        batch = self.tokenizer.pad([{k: f[k] for k in ("input_ids", "attention_mask")} for f in features], return_tensors="pt")
        n = max(len(f["order"]) for f in features)
        for key, fill in (("order", -1), ("header_end", -100), ("loop", -100)):
            batch[key] = torch.tensor([f[key] + [fill] * (n - len(f[key])) for f in features])
        for key, fill in (("insert", 0), ("keyword_end", -100)):
            batch[key] = torch.tensor([f[key] + [[fill] * len(f[key][0])] * (n + 1 - len(f[key])) for f in features])
        return batch


def evaluate(predictor: ChartPredictor, test) -> dict:
    """Share of greedy skeletons equal to the gold skeleton, by statement count, and of predicted orders equal to the source order."""
    requests = [CflowRequest.from_row(row) for row in test]
    predictions = predictor(requests)
    buckets, orders = {}, 0
    for request, prediction, row in zip(requests, predictions, test):
        bucket = buckets.setdefault(max(8, 2 ** math.ceil(math.log2(row["n_stmts"]))), [0, 0])
        bucket[0] += 1
        bucket[1] += bool(prediction) and prediction[0] == row["target"].split()
        encoded = predictor.tokenizer(row["input"], return_tensors="pt").to(predictor.device)
        with torch.no_grad():
            vectors, counts = predictor.model.statements(encoded.input_ids, encoded.attention_mask)
            successors = predictor.model.successors(vectors, counts)[0].float().cpu().numpy()
        orders += decode_order(successors, request.shapes) == gold_items(row["target"].split()).order
    return {
        "exact_match": sum(c for _, c in buckets.values()) / max(len(test), 1),
        "order_match": orders / max(len(test), 1),
        "by_n_stmts": {f"<={b}": c / n for b, (n, c) in sorted(buckets.items())},
    }


def train_chart_model(config: ControlFlowConfiguration, public: bool = False):
    dataset = load_cflow_dataset(config)
    tokenizer = build_tokenizer(itertools.chain(dataset["train"]["input"], dataset["train"]["target"]), config.max_stmts)

    def tokenize(row):
        return tokenizer(row["input"]) | chart_labels(row)

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    params = config.model_parameters
    model_config = T5Config(
        vocab_size=len(tokenizer), d_model=params.d_model, d_kv=params.d_model // params.heads, d_ff=params.d_ff, num_layers=params.layers, num_heads=params.heads,
        relative_attention_num_buckets=64, relative_attention_max_distance=512, dropout_rate=0.1, feed_forward_proj="gated-gelu",
        is_encoder_decoder=False, use_cache=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )  # fmt: skip
    # read back by pylingual.models.ChartPredictor and ChartParser
    model_config.max_stmts = config.max_stmts
    model_config.max_input_tokens = config.max_input_tokens
    model_config.skeleton_keywords = [k for k in KEYWORDS if any(k in target.split() for target in dataset["train"]["target"])]
    model_config.statement_token_id = tokenizer.convert_tokens_to_ids(STMT_SEP)
    model = ChartParser(model_config)

    model_dir = config.cache_dir / "models" / f"{config.model_repo_name}-chart"
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
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        train_sampling_strategy="group_by_length",
        label_names=LABELS,
        prediction_loss_only=True,
        report_to="tensorboard",
        logging_steps=100,
        ddp_find_unused_parameters=False,
        push_to_hub=config.push_to_hub,
        hub_model_id=f"{config.model_repo_name}-chart",
        hub_private_repo=not public,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=Collator(tokenizer),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(model_dir)) if model_dir.is_dir() else None)
    duration = str(timedelta(seconds=time.time() - start))

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        trainer.save_model(str(model_dir))
        report = evaluate(ChartPredictor(trainer.model, tokenizer, trainer.model.device), dataset["test"])
        (model_dir / "eval.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if config.push_to_hub:
            trainer.push_to_hub(commit_message=duration)


@click.command(help="Training script for the chart parser control flow model (ARCH chart) given a control flow json.")
@click.argument("json_path", type=str)
@click.option("--public", is_flag=True, default=False, help="Make uploaded HuggingFace repos public (default: private)")
def main(json_path: str, public: bool):
    train_chart_model(parse_control_flow_config_json(pathlib.Path(json_path)), public)


if __name__ == "__main__":
    main()
