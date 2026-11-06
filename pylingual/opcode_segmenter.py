"""Instruction-aligned inference for frozen atomic-opcode tokenizers."""
from __future__ import annotations
import torch
from pylingual.opcodes import SEPARATOR, instruction_token_indices


class OpcodeSegmenter:
    def __init__(self, bank, tokenizer, version=None, *, python_version=None):
        self.bank, self.model, self.tokenizer = bank, bank.model, tokenizer
        self.version = version
        self.python_version = python_version if python_version is not None else version
        self.device = next(self.model.parameters()).device

    def for_version(self, version, *, python_version=None):
        return type(self)(self.bank, self.tokenizer, version, python_version=python_version)

    def instruction_token_lengths(self, instructions):
        return [len(self.tokenizer(self.bank.bind_bytecode(text, self.python_version))["input_ids"])
                for text in instructions]

    def __call__(self, requests, batch_size=8, **kwargs):
        if isinstance(requests, str):
            return self([requests], batch_size=batch_size, **kwargs)[0]
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive segmentation batch size required")
        results = []
        with self.bank.use(self.version), torch.no_grad():
            self.model.eval()
            for start in range(0, len(requests), batch_size):
                texts = [requests[i] for i in range(start, min(start+batch_size, len(requests)))]
                features, alignments = [], []
                for text in texts:
                    bound = self.bank.bind_bytecode(text, self.python_version)
                    encoded, indices = instruction_token_indices(self.tokenizer, bound)
                    limit = self.model.config.max_position_embeddings - self.tokenizer.pad_token_id - 1
                    if len(encoded["input_ids"]) > limit:
                        raise ValueError("Segmentation window exceeds backbone positional capacity")
                    features.append(dict(encoded)); alignments.append(indices)
                batch = self.tokenizer.pad(features, padding=True, return_tensors="pt").to(self.device)
                scores = self.model(**batch).logits.softmax(-1)
                for text, indices, row in zip(texts, alignments, scores):
                    instructions = text.split(SEPARATOR)
                    first = {}
                    for position, instruction in enumerate(indices):
                        if instruction is not None:
                            first.setdefault(instruction, position)
                    if set(first) != set(range(len(instructions))):
                        raise ValueError("Segmentation tokens do not cover every instruction")
                    predictions, cursor = [], 0
                    for instruction, raw in enumerate(instructions):
                        confidence, label = row[first[instruction]].max(-1)
                        entity = self.model.config.id2label[int(label)]
                        if entity not in {"B", "I", "E"}:
                            raise ValueError("Segmentation backbone must use B/I/E labels")
                        predictions.append({"entity": entity, "score": float(confidence), "word": raw,
                                            "start": cursor, "end": cursor+len(raw), "index": instruction+1})
                        cursor += len(raw)+len(SEPARATOR)
                    results.append(predictions)
        return results
