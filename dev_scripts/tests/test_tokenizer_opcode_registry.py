import unittest
from dev_scripts.tokenizer_opcode_registry import RESERVED_SLOT_COUNT, RESERVED_TOKENS, bind_bytecode, known_token, opcode_names


class OpcodeRegistryTests(unittest.TestCase):
    def test_capacity(self):
        self.assertEqual(len(RESERVED_TOKENS), RESERVED_SLOT_COUNT)
        self.assertEqual(len(set(RESERVED_TOKENS)), RESERVED_SLOT_COUNT)

    def test_only_opcode_fields_are_bound(self):
        known = {"LOAD_CONST", "RETURN_VALUE"}
        self.assertEqual(bind_bytecode("LOAD_CONST , LOAD_FAST <SEP> RETURN_VALUE", known),
                         "<opcode:LOAD_CONST> , LOAD_FAST <SEP> <opcode:RETURN_VALUE>")

    def test_operand_opcode_name_is_not_registered(self):
        self.assertEqual(opcode_names("LOAD_CONST , FUTURE_OPCODE <SEP> RETURN_VALUE"), {"LOAD_CONST", "RETURN_VALUE"})

    def test_numeric_jump_target_prefix_is_preserved(self):
        self.assertEqual(bind_bytecode('^~> 80 RETURN_CONST , None', {'RETURN_CONST'}),
                         '^~> 80 <opcode:RETURN_CONST> , None')

    def test_named_and_exception_target_prefixes(self):
        for prefix in ('E-> ', 'v~> ^~> 80 ', 'E-> v~> ^~> TARGET_2 ', '80 '):
            text = prefix+'LOAD_CONST , RETURN_VALUE E-END'
            self.assertEqual(opcode_names(text), {'LOAD_CONST'})
            self.assertEqual(bind_bytecode(text, {'LOAD_CONST'}),
                             prefix+'<opcode:LOAD_CONST> , RETURN_VALUE E-END')

    def test_future_annotations_flag_is_not_rewritten(self):
        text = 'SETUP_ANNOTATIONS-FUTURE'
        self.assertEqual(opcode_names(text), {'SETUP_ANNOTATIONS'})
        self.assertEqual(bind_bytecode(text, {'SETUP_ANNOTATIONS'}),
                         '<opcode:SETUP_ANNOTATIONS>-FUTURE')

    def test_future_binding(self):
        self.assertEqual(bind_bytecode("NEW_OPCODE , 3", set(), {"NEW_OPCODE": 15}), "<future_opcode_15> , 3")

    def test_unbound_is_not_silently_subworded(self):
        with self.assertRaisesRegex(ValueError, "Unbound"):
            bind_bytecode("NEW_OPCODE", set())

    def test_overflow(self):
        with self.assertRaises(ValueError):
            bind_bytecode("NEW_0", set(), {f"NEW_{i}": i for i in range(RESERVED_SLOT_COUNT + 1)})

    def test_out_of_range(self):
        for slot in (-1, RESERVED_SLOT_COUNT):
            with self.assertRaises(ValueError):
                bind_bytecode("NEW", set(), {"NEW": slot})

    def test_aliases_must_not_share_slots(self):
        with self.assertRaises(ValueError):
            bind_bytecode("NEW", set(), {"NEW": 0, "OTHER": 0})

    def test_known_identity_is_immutable(self):
        with self.assertRaises(ValueError):
            bind_bytecode("LOAD_CONST", {"LOAD_CONST"}, {"LOAD_CONST": 0})

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            known_token("not an opcode")

    def test_invalid_instruction(self):
        with self.assertRaises(ValueError):
            bind_bytecode("", set())


if __name__ == "__main__":
    unittest.main()
