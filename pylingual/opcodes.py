"""Fixed opcode placeholders: bytecode fields only, never generated Python source."""
from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path

SEPARATOR = " <SEP> "
# 16 slots proved too few: CPython 3.11 alone added 29 pyc-visible instructions and 3.12
# another 24, so a backbone that ages across releases needs headroom (see SHARED_TOKENIZER_TRAINING.md).
RESERVED_SLOT_COUNT = 64


def reserved_token(index: int) -> str:
    return f"<future_opcode_{index:02d}>"


RESERVED_TOKENS = tuple(reserved_token(i) for i in range(RESERVED_SLOT_COUNT))
# Normal model views prefix jump/exception targets before the actual opcode.
# Statement rows can replace numeric target offsets with TARGET_N labels.
OPCODE_FIELD = re.compile(
    r"^(\s*(?:(?:E->|\^~>|v~>)\s+)*(?:(?:\d+|TARGET_\d+)\s+)?)"
    r"([A-Z][A-Z0-9_]*)(?=\s|,|-FUTURE(?:\s|$)|$)"
)


def opcode_names(bytecode: str) -> set[str]:
    return {match.group(2) for instruction in bytecode.split(SEPARATOR)
            if (match := OPCODE_FIELD.match(instruction))}


def known_token(name: str) -> str:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise ValueError(f"Invalid opcode identity: {name!r}")
    return f"<opcode:{name}>"


def bind_bytecode(bytecode: str, known: set[str], bindings: dict[str, int] | None = None) -> str:
    """Replace only each instruction's opcode field; keep operands verbatim.

    Bindings map future mnemonic -> slot index 0..RESERVED_SLOT_COUNT-1, not arbitrary vocabulary IDs.
    The saved registry supplies the immutable ID of each placeholder.
    """
    bindings = bindings or {}
    if len(bindings) > RESERVED_SLOT_COUNT or len(set(bindings.values())) != len(bindings):
        raise ValueError("Future opcode bindings exceed capacity or reuse a slot")
    if any(type(index) is not int or not 0 <= index < RESERVED_SLOT_COUNT for index in bindings.values()):
        raise ValueError(f"Future opcode slot outside 0..{RESERVED_SLOT_COUNT - 1}")
    if known.intersection(bindings):
        raise ValueError("Cannot remap an existing known opcode")
    result = []
    for instruction in bytecode.split(SEPARATOR):
        match = OPCODE_FIELD.match(instruction)
        if match is None:
            raise ValueError(f"Unrecognized instruction field: {instruction[:100]!r}")
        name = match.group(2)
        if name in known:
            replacement = known_token(name)
        elif name in bindings:
            replacement = reserved_token(bindings[name])
        else:
            raise ValueError(f"Unbound opcode: {name}")
        result.append(instruction[:match.start(2)] + replacement + instruction[match.end(2):])
    return SEPARATOR.join(result)


def instruction_token_indices(tokenizer, bound_bytecode: str, **tokenizer_kwargs):
    """Offset-based alignment also covers non-special atomic opcode AddedTokens.

    Returns encoding and one instruction index per token; structural tokens get
    None. Do not assume AddedTokens preserve the old tokenizer.word_ids() layout.
    """
    spans = []
    cursor = 0
    for instruction in bound_bytecode.split(SEPARATOR):
        spans.append((cursor, cursor + len(instruction)))
        cursor += len(instruction) + len(SEPARATOR)
    encoded = tokenizer(bound_bytecode, return_offsets_mapping=True, **tokenizer_kwargs)
    indices = []
    position = 0
    for start, end in encoded.pop("offset_mapping"):
        if start == end:
            indices.append(None)
            continue
        while position < len(spans) and start >= spans[position][1]:
            position += 1
        if position >= len(spans):
            indices.append(None)
            continue
        left, right = spans[position]
        if end > left and start < right:
            if end > right and bound_bytecode[right:end].strip():
                raise ValueError("Tokenizer token crosses an instruction boundary")
            indices.append(position)
        else:
            indices.append(None)
    return encoded, indices


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash stable tokenizer semantics, excluding mutable per-call padding state."""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    value = {"backend": backend,
             "special_ids": sorted(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)),
             "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class OpcodeRegistry:
    """Immutable, tokenizer-validated opcode IDs; bindings live in each adapter."""
    def __init__(self, data: dict, tokenizer=None):
        self.data = json.loads(json.dumps(data))
        if self.data.get("format") != 1 or self.data.get("preprocessing") != "opcode-field-placeholders-v1":
            raise ValueError("Unsupported opcode registry")
        self.known = self.data.get("known_opcodes", {})
        slots = self.data.get("reserved_slots", [])
        if not isinstance(self.known, dict) or not self.known or len(slots) != RESERVED_SLOT_COUNT:
            raise ValueError(f"Registry requires known opcodes and exactly {RESERVED_SLOT_COUNT} reserved slots")
        self.slots = {}
        for item in slots:
            index = item.get("slot")
            if type(index) is not int or not 0 <= index < RESERVED_SLOT_COUNT or index in self.slots:
                raise ValueError("Invalid or duplicate reserved slot")
            if item.get("token") != reserved_token(index):
                raise ValueError("Reserved opcode token mismatch")
            self.slots[index] = item
        entries = list(self.known.values()) + list(self.slots.values())
        for name, item in self.known.items():
            if item.get("token") != known_token(name):
                raise ValueError("Known opcode token mismatch")
        ids = [item.get("id") for item in entries]
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid or duplicate opcode vocabulary IDs")
        if set(self.data.get("source_generation_forbidden_ids", [])) != set(ids):
            raise ValueError("Generation exclusions must cover all opcode and reserved IDs")
        self.forbidden_ids = sorted(ids)
        self.fingerprint = hashlib.sha256(json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if tokenizer is not None:
            self.validate_tokenizer(tokenizer)

    @classmethod
    def load(cls, path, tokenizer=None):
        return cls(json.loads(Path(path).read_text()), tokenizer)

    def validate_tokenizer(self, tokenizer):
        for item in list(self.known.values()) + list(self.slots.values()):
            token, identifier = item["token"], item["id"]
            if tokenizer.convert_tokens_to_ids(token) != identifier or tokenizer(token, add_special_tokens=False)["input_ids"] != [identifier]:
                raise ValueError("Opcode registry/tokenizer ID or atomicity mismatch")
            if identifier in tokenizer.all_special_ids:
                raise ValueError("Opcode IDs must not be structural special tokens")

    def validate_bindings(self, bindings):
        if not isinstance(bindings, dict):
            raise ValueError("Opcode bindings must be a mnemonic-to-slot mapping")
        for name in bindings:
            known_token(name)
        if any(type(slot) is not int or not 0 <= slot < RESERVED_SLOT_COUNT for slot in bindings.values()):
            raise ValueError(f"Future opcode slot outside 0..{RESERVED_SLOT_COUNT - 1}")
        if len(bindings) > RESERVED_SLOT_COUNT or len(set(bindings.values())) != len(bindings):
            raise ValueError("Future opcode bindings exceed capacity or reuse a slot")
        if set(self.known).intersection(bindings):
            raise ValueError("Cannot remap an existing known opcode")
        return dict(bindings)

    def bindings_for(self, names):
        missing = sorted(set(names) - set(self.known))
        return self.validate_bindings({name: slot for slot, name in enumerate(missing)})

    def row_ids(self, bindings):
        return sorted(self.slots[slot]["id"] for slot in self.validate_bindings(bindings).values())

    def bind(self, bytecode, bindings=None):
        return bind_bytecode(bytecode, set(self.known), self.validate_bindings(bindings or {}))
