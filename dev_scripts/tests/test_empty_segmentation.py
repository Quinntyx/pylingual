"""Zero-instruction code objects retain their positions without entering a model."""
import unittest
from types import SimpleNamespace
from pylingual.decompiler import Decompiler
from pylingual.opcodes import SEPARATOR, bind_bytecode

class StrictSegmenter:
    def __init__(self):
        self.requests=[]
        self.length_requests=[]
    def instruction_token_lengths(self,instructions):
        self.length_requests.extend(instructions)
        for instruction in instructions:
            bind_bytecode(instruction,{'LOAD_CONST','RETURN_VALUE'})
        return [3 for instruction in instructions]
    def __call__(self,requests,**kwargs):
        texts=list(requests);self.requests.extend(texts)
        return [[{'word':instruction,'entity':'I','score':1.0} for instruction in text.split(SEPARATOR)] for text in texts]

class EmptySegmentationTests(unittest.TestCase):
    def make_decompiler(self,requests):
        value=object.__new__(Decompiler)
        value.name='fixture';value.trust_lnotab=False;value.timeout=None
        value.segmentation_requests=requests;value.segmenter=StrictSegmenter()
        value.ordered_instructions=[[SimpleNamespace(starts_line=99) for _ in request.split(SEPARATOR)] if request else [] for request in requests]
        return value
    def test_empty_objects_before_between_after_nonempty_keep_alignment(self):
        value=self.make_decompiler(['','LOAD_CONST , None'+SEPARATOR+'RETURN_VALUE','','RETURN_VALUE',''])
        value.run_segmentation()
        self.assertEqual([[p['entity'] for p in row] for row in value.segmentation_results],[[],['B','I'],[],['B'],[]])
        self.assertEqual([[i.starts_line for i in row] for row in value.ordered_instructions],[[],[1,None],[],[2],[]])
        self.assertNotIn('',value.segmenter.length_requests)
        self.assertNotIn('',value.segmenter.requests)
    def test_all_empty_objects_skip_model(self):
        value=self.make_decompiler(['',''])
        value.run_segmentation()
        self.assertEqual(value.segmentation_results,[[],[]])
        self.assertEqual(value.segmenter.requests,[])
        self.assertEqual(value.segmenter.length_requests,[])
    def test_no_objects_skip_model(self):
        value=self.make_decompiler([])
        value.run_segmentation()
        self.assertEqual(value.segmentation_results,[])
        self.assertEqual(value.segmenter.requests,[])
    def test_malformed_nonempty_instruction_still_rejected(self):
        value=self.make_decompiler([SEPARATOR+'RETURN_VALUE'])
        with self.assertRaisesRegex(ValueError,'Unrecognized instruction field'):
            value.run_segmentation()

if __name__=='__main__':unittest.main()
