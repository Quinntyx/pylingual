"""Compatibility imports; the production opcode lexer lives in pylingual.opcodes."""
from pylingual.opcodes import (SEPARATOR, RESERVED_SLOT_COUNT, RESERVED_TOKENS, OPCODE_FIELD,
    opcode_names, known_token, bind_bytecode, instruction_token_indices,
    OpcodeRegistry, tokenizer_fingerprint)
