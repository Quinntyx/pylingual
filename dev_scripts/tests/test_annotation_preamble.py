"""xdis exposes common constants as either indices or real Python objects."""
import unittest
from types import SimpleNamespace
from pylingual.editable_bytecode.EditableBytecode import _annotation_preamble_signature

class AnnotationPreambleTests(unittest.TestCase):
 def test_old_and_new_xdis_common_constant_representations_match(self):
  for value in (1,NotImplementedError):
   with self.subTest(value=value):
    instruction=SimpleNamespace(opname='LOAD_COMMON_CONSTANT',arg=1,argval=value)
    self.assertEqual(_annotation_preamble_signature([instruction]),(('LOAD_COMMON_CONSTANT',1),))
 def test_other_opcodes_keep_decoded_values(self):
  instruction=SimpleNamespace(opname='LOAD_FAST_BORROW',arg=0,argval='format')
  self.assertEqual(_annotation_preamble_signature([instruction]),(('LOAD_FAST_BORROW','format'),))
 def test_wrong_common_constant_is_not_accepted(self):
  instruction=SimpleNamespace(opname='LOAD_COMMON_CONSTANT',arg=2,argval=NotImplementedError)
  self.assertNotEqual(_annotation_preamble_signature([instruction]),(('LOAD_COMMON_CONSTANT',1),))
if __name__=='__main__':unittest.main()
