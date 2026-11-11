"""
Training script for the chart parser control flow model (ARCH chart).

Batches hold up to `batch_size` rows and `max_batch_tokens` padded tokens, from rows of similar length, so long code
objects train without running out of memory and short ones without padding. After each epoch the model decodes the
valid split, and the epoch with the most exact greedy skeletons on it is kept.

The json is a control flow config (ControlFlowConfiguration) with an optional "chart" section, see CHART_DEFAULTS.
"""

import dataclasses
import itertools
import json
import math
import pathlib
import random
import time
from datetime import timedelta

import click
import numpy as np
import torch
from ControlFlowConfiguration import ControlFlowConfiguration, ModelParameters, TrainingParameters
from train_cflow import build_tokenizer, load_cflow_dataset
from transformers import T5Config

from pylingual.control_flow_reconstruction.block_grammar import Shape
from pylingual.control_flow_reconstruction.chart import KEYWORDS, decode_order, gold_items
from pylingual.control_flow_reconstruction.chart_model import EDGE_KINDS, ChartParser
from pylingual.control_flow_reconstruction.model_view import MID, STMT_SEP, CflowRequest
from pylingual.control_flow_reconstruction.skeleton import BLOCK_KEYWORDS, ref
from pylingual.models import ChartPredictor

LABELS = ["order", "header_end", "keyword_end", "insert", "loop"]

CHART_DEFAULTS = {
    "seed": 42,
    "max_batch_tokens": 16384,  # padded tokens in a batch
    "warmup": 0.05,  # share of the steps
    "weight_decay": 0.01,
    "dropout": 0.1,
    "order_buckets": 32,  # relative position buckets of the successor head, 0 for none
    "order_candidates": 1,  # orders of the successor head the chart decodes, besides the statement order
    "edge_features": True,  # score the input's edges between statements in the order and block end heads
    "eval_beams": 1,
}


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


def token_batches(lengths: list[int], max_rows: int, max_tokens: int, rng: random.Random | None) -> list[list[int]]:
    """
    Batches of row indices with at most `max_rows` rows and `max_tokens` padded tokens. Rows are sorted by length in
    chunks of shuffled rows (all of them without `rng`), and the batches are shuffled.
    """
    order = list(range(len(lengths)))
    chunk = len(order)
    if rng:
        rng.shuffle(order)
        chunk = 64 * max_rows
    batches = []
    for start in range(0, len(order), chunk):
        batch, longest = [], 0
        for i in sorted(order[start : start + chunk], key=lambda i: lengths[i]):
            if batch and (len(batch) == max_rows or max(longest, lengths[i]) * (len(batch) + 1) > max_tokens):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, lengths[i])
        batches.append(batch)
    if rng:
        rng.shuffle(batches)
    return batches


def evaluate(predictor: ChartPredictor, rows, num_beams: int = 1) -> dict:
    """
    Share of greedy skeletons equal to the gold skeleton, by statement count; of greedy skeletons in the gold order, of
    gold skeletons among the beams, and of predicted orders (greedy successors) equal to the source order.
    """
    requests = [CflowRequest.from_row(row) for row in rows]
    predictions = predictor(requests, num_beams=num_beams)
    buckets, orders, greedy_orders, any_k = {}, 0, 0, 0
    for request, prediction, row in zip(requests, predictions, rows):
        gold = row["target"].split()
        gold_order = gold_items(gold).order
        bucket = buckets.setdefault(max(8, 2 ** math.ceil(math.log2(row["n_stmts"]))), [0, 0])
        bucket[0] += 1
        bucket[1] += bool(prediction) and prediction[0] == gold
        any_k += bool(prediction) and gold in prediction
        greedy_orders += bool(prediction) and gold_items(prediction[0]).order == gold_order
        encoded = predictor.tokenizer(row["input"], return_tensors="pt").to(predictor.device)
        with torch.no_grad():
            successors = predictor.model.successors(*predictor.model.statements(encoded.input_ids, encoded.attention_mask))[0].float().cpu().numpy()
        orders += decode_order(successors, request.shapes) == gold_order
    n = max(len(rows), 1)
    return {
        "exact_match": sum(c for _, c in buckets.values()) / n,
        "greedy_order_match": greedy_orders / n,
        "order_match": orders / n,
        f"any_{num_beams}": any_k / n,
        "by_n_stmts": {f"<={b}": c / k for b, (k, c) in sorted(buckets.items())},
    }


def parse_config(path: pathlib.Path) -> tuple[ControlFlowConfiguration, dict]:
    raw = json.loads(path.read_text())
    chart = CHART_DEFAULTS | raw.pop("chart", {})
    raw["model_parameters"] = ModelParameters(**raw.get("model_parameters", {}))
    raw["training_parameters"] = TrainingParameters(**raw["training_parameters"])
    return ControlFlowConfiguration(**raw), chart


def train_chart_model(config: ControlFlowConfiguration, chart: dict):
    seed = chart["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    dataset = load_cflow_dataset(config)
    tokenizer = build_tokenizer(itertools.chain(dataset["train"]["input"], dataset["train"]["target"]), config.max_stmts)

    def tokenize(rows: list[dict]) -> list[dict]:
        encoded = tokenizer([row["input"] for row in rows])
        return [{"input_ids": ids, "attention_mask": mask} | chart_labels(row) for ids, mask, row in zip(encoded["input_ids"], encoded["attention_mask"], rows)]

    rows = {split: dataset[split].to_list() for split in ("train", "validation")}
    tokenized = {split: tokenize(split_rows) for split, split_rows in rows.items()}

    params = config.model_parameters
    model_config = T5Config(
        vocab_size=len(tokenizer), d_model=params.d_model, d_kv=params.d_model // params.heads, d_ff=params.d_ff, num_layers=params.layers, num_heads=params.heads,
        relative_attention_num_buckets=64, relative_attention_max_distance=512, dropout_rate=chart["dropout"], feed_forward_proj="gated-gelu",
        is_encoder_decoder=False, use_cache=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )  # fmt: skip
    # read back by pylingual.models.ChartPredictor and ChartParser
    model_config.max_stmts = config.max_stmts
    model_config.max_input_tokens = config.max_input_tokens
    model_config.skeleton_keywords = [k for k in KEYWORDS if any(k in target.split() for target in dataset["train"]["target"])]
    model_config.statement_token_id = tokenizer.convert_tokens_to_ids(STMT_SEP)
    model_config.order_buckets = chart["order_buckets"]
    model_config.order_candidates = chart["order_candidates"]
    model_config.edge_features = chart["edge_features"]
    model_config.edge_token_ids = [[tokenizer.convert_tokens_to_ids(kind + d) for d in "+-="] for kind in EDGE_KINDS]
    model_config.mid_token_id = tokenizer.convert_tokens_to_ids(MID)
    model_config.ref_token_id = tokenizer.convert_tokens_to_ids(ref(0))  # the references are in order
    model = ChartParser(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params}", flush=True)

    model_dir = config.cache_dir / "models" / f"{config.model_repo_name}-chart"
    model_dir.mkdir(parents=True, exist_ok=True)
    collate = Collator(tokenizer)
    train, valid = tokenized["train"], tokenized["validation"]
    train_lengths = [len(r["input_ids"]) for r in train]
    valid_lengths = [len(r["input_ids"]) for r in valid]
    tp = config.training_parameters
    rng = random.Random(seed)
    steps_per_epoch = len(token_batches(train_lengths, tp.batch_size, chart["max_batch_tokens"], random.Random(seed)))
    total = steps_per_epoch * tp.epochs
    warmup = max(1, int(chart["warmup"] * total))
    decay = [p for name, p in model.named_parameters() if p.ndim >= 2 and "relative_attention_bias" not in name]
    no_decay = [p for name, p in model.named_parameters() if not (p.ndim >= 2 and "relative_attention_bias" not in name)]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": chart["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=tp.learning_rate, fused=device.type == "cuda")
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    print(f"{len(train)} train rows, {len(valid)} valid rows, {steps_per_epoch} steps per epoch, {total} steps, bf16 {bf16}", flush=True)

    def to_device(batch):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    history, best, step, start = [], None, 0, time.time()
    for epoch in range(tp.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch_rows in token_batches(train_lengths, tp.batch_size, chart["max_batch_tokens"], rng):
            batch = to_device(collate([train[i] for i in batch_rows]))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                loss = model(**batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running += loss.item()
            seen += 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {running / seen:.4f} lr {schedule.get_last_lr()[0]:.2e} {timedelta(seconds=int(time.time() - start))}", flush=True)
                running, seen = 0.0, 0

        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            for batch_rows in token_batches(valid_lengths, tp.batch_size, chart["max_batch_tokens"], None):
                losses.append((model(**to_device(collate([valid[i] for i in batch_rows])))["loss"].item(), len(batch_rows)))
        valid_loss = sum(l * k for l, k in losses) / sum(k for _, k in losses)
        report = evaluate(ChartPredictor(model, tokenizer, device), rows["validation"], chart["eval_beams"])
        model.train()
        entry = {"epoch": epoch, "step": step, "valid_loss": valid_loss, **report, "elapsed": str(timedelta(seconds=int(time.time() - start)))}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if best is None or report["exact_match"] > best["exact_match"]:
            best = entry
            model.save_pretrained(str(model_dir))
            tokenizer.save_pretrained(str(model_dir))

    summary = {
        "trainable_parameters": n_params,
        "seed": seed,
        "chart": chart,
        "training_parameters": dataclasses.asdict(tp),
        "model_parameters": dataclasses.asdict(params),
        "wall_time": str(timedelta(seconds=int(time.time() - start))),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
        "best": best,
        "history": history,
    }
    (model_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)


@click.command(help="Training script for the chart parser control flow model (ARCH chart) given a control flow json.")
@click.argument("json_path", type=str)
def main(json_path: str):
    config, chart = parse_config(pathlib.Path(json_path))
    if config.push_to_hub:
        raise click.UsageError("the chart trainer does not upload models")
    train_chart_model(config, chart)


if __name__ == "__main__":
    main()
