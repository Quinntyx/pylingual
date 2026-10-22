import sys
import unittest
import xdis.opcodes
from pylingual.editable_bytecode.EditableBytecode import EditableBytecode
from pylingual.masking.model_disasm import create_global_masker


class NativeCodeObjectCompatibilityTests(unittest.TestCase):
    def editable(self,source):
        opcode = getattr(xdis.opcodes, 'opcode_%d%d' % sys.version_info[:2])
        return EditableBytecode(compile(source,'<regression>','exec'),opcode,sys.version_info[:2])

    def test_inserted_local_updates_native_nlocals(self):
        bytecode = self.editable('value = 1\n')
        bytecode.co_varnames.append('_inserted_local')
        rebuilt = bytecode.to_code(no_lnotab=True)
        self.assertEqual(rebuilt.co_nlocals,len(rebuilt.co_varnames))
        self.assertIn('_inserted_local',rebuilt.co_varnames)

    @unittest.skipUnless(sys.version_info[:2] == (3,14),'Native 3.14 annotation inlining')
    def test_annotated_classes_can_be_masked(self):
        for source in ['class C:\n    x: int\n','class C:\n    x: int = 1\n    y: str = "a"\n']:
            with self.subTest(source=source):
                bytecode = self.editable(source)
                for nested in bytecode.iter_bytecodes():
                    rebuilt = nested.to_code(no_lnotab=True)
                    self.assertEqual(rebuilt.co_nlocals,len(rebuilt.co_varnames))
                self.assertIsNotNone(create_global_masker(bytecode))
