import ast
import unittest
from pylingual.masking.ast_masker import custom_unparse
from pylingual.masking.global_masker import Masker

class ASTMaskerRuntimeCompatibilityTests(unittest.TestCase):
    def test_fstring_round_trips(self):
        for source in ['result = f"{value}"', 'result = f"{value!r:>12}"', 'result = f"{value:.{precision}f}"', 'result = f"{(lambda: value)()}"', 'result = f"{ {\'a\': value} }"']:
            with self.subTest(source=source):
                original = ast.parse(source)
                rendered = custom_unparse(original, Masker())
                self.assertEqual(ast.dump(ast.parse(rendered), include_attributes=False), ast.dump(original, include_attributes=False))
