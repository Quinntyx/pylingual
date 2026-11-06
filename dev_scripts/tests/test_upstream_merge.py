"""Shared-family contracts retained through the upstream dev merge."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pylingual.models import CacheTranslator, _single_slot_cache

class UpstreamMergeTests(unittest.TestCase):
    def test_nonredis_constructor_keeps_legacy_positional_device(self):
        model=SimpleNamespace(to=Mock())
        translator=CacheTranslator(model,None,"cpu")
        model.to.assert_called_once_with("cpu")
        self.assertFalse(translator.redis_enabled)

    def test_single_slot_cache_clear_and_explicit_cpu_fallback(self):
        calls=[]
        import torch
        @_single_slot_cache
        def loader(key,device=None):
            calls.append((key,device))
            if device is None:raise torch.OutOfMemoryError("simulated allocation")
            return object()
        with patch("pylingual.models._free_device_memory"):
            value=loader("family")
            self.assertIs(value,loader("family"))
            self.assertEqual(calls,[("family",None),("family",torch.device("cpu"))])
            loader.cache_clear()
            self.assertIsNot(value,loader("family"))

    def test_backoff_halves_actual_batch_without_replaying_prior_success(self):
        translator=object.__new__(CacheTranslator);calls=[]
        def generate(batch):
            calls.append(list(batch))
            if len(batch)>2:raise RuntimeError("simulated batch OOM")
            return list(batch)
        translator._translate_and_decode=generate
        with patch("pylingual.models.torch.cuda.is_available",return_value=False):
            values=list(range(35));self.assertEqual(translator._translate_with_backoff(values),values)
        self.assertEqual(calls[-3:],[list(range(32,35)),[32],[33],[34]][-3:])
        self.assertEqual(sum(32 in batch for batch in calls),2)

if __name__=="__main__":unittest.main()
