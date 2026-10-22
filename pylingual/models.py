from __future__ import annotations

import yaml
import logging

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING
from functools import lru_cache

from pylingual.masking.model_disasm import (
    fix_jump_targets,
    normalize_masks,
    restore_masks,
)
from pylingual.utils.lists import flatten
from pylingual.utils.tracked_list import TrackedDataset, TRANSLATION_STEP
from pylingual.utils.version import PythonVersion
from pylingual.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
    import transformers
    import huggingface_hub
else:
    lazy_import("torch")
    lazy_import("transformers")
    lazy_import("huggingface_hub")

logger = logging.getLogger(__name__)


# translator with caching
class CacheTranslator:
    """
    Adds cache support for statement translation

    :param translator : The loaded translation model
    :param maxsize : The maximum amount of cached items
    """

    def __init__(self, model : transformers.T5ForConditionalGeneration, tokenizer: transformers.RobertaTokenizer, device = torch.device,maxsize=50000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.input_prefix = ""
        self.generation_kwargs = {}
        self.decode_excluded_ids = {0, 1, 2, 3, 4}
        self.model.to(self.device)

    def __getitem__(self, item):
        self.cache.move_to_end(item)
        return self.cache[item]

    def _prepare_bytecode(self, text):
        return text

    def _translate_and_decode(
        self,
        translation_requests: TrackedDataset | list[str],
        batch_size: int = 32,
        **kwargs,
    ) -> list[str]:

        decoded_results = []

        # process in batches
        for i in range(0, len(translation_requests), batch_size):
            # TrackedDataset is indexable but does not support slices, and its
            # iterator is stateful. Indexed access also makes retries replayable.
            batch = [self.input_prefix + self._prepare_bytecode(translation_requests[j]) for j in range(i, min(i + batch_size, len(translation_requests)))]
            encoded = self.tokenizer(batch, return_tensors= "pt", padding=True).to(self.device)

            with torch.no_grad():
                output = self.model.generate(**encoded, max_length=512, **self.generation_kwargs)

            for tokens in output:
                # explicitly filter out the special tokens we want to skip: <pad>, <s>, </s>, <unk>, <mask>
                filtered_tokens = [tok for tok in tokens.tolist() if tok not in self.decode_excluded_ids]
                # decode the remaining tokens
                decoded_results.append(self.tokenizer.decode(filtered_tokens, skip_special_tokens=False))

        return decoded_results

    def _translate_with_backoff(self, translation_requests: TrackedDataset) -> list[str]:
        try:
            return self._translate_and_decode(translation_requests, batch_size=32)
        except TimeoutError:
            raise
        except Exception as e:
            logger.info(f"Lowering translation batch size ({e})")
        # Try with batch_size = 1 if normal translation fails:
        try:
            return self._translate_and_decode(translation_requests, batch_size=1)
        except TimeoutError:
            raise
        except Exception as e:
            logger.info(f"Lowering translation batch size ({e})")
        translation_results = []
        # try one-by-one if batch_size=1 fails
        for index in range(len(translation_requests)):
            request = translation_requests[index]
            try:
                translation_results.append(self._translate_and_decode([request], batch_size=1)[0])
            except TimeoutError:
                raise
            except Exception:
                # last resort fallback
                translation_results.append("'''Decompiler error: line too long for translation. Please decompile this statement manually.'''")
        return translation_results

    def __call__(self, args: list, check_timeout: callable = None, **_):
        normalized_args = [normalize_masks(fix_jump_targets(x)) for x in args]

        # New are those not in the local cache
        new = TrackedDataset(
            TRANSLATION_STEP,
            list({norm for norm, _ in normalized_args if norm not in self.cache}),
            check_timeout=check_timeout,
        )

        # Now, "new" has been updated to those not in local
        for arg, result in zip(new.x, self._translate_with_backoff(new)):
            self.cache[arg] = result

        results = [restore_masks(self[norm], order) for norm, order in normalized_args]
        while len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return results


class VersionedTranslator(CacheTranslator):
    """A per-version output cache over one shared, unmerged backbone."""
    def __init__(self, bank, tokenizer, device, version, prefix, use_adapter=True):
        super().__init__(bank.model, tokenizer, device)
        self.bank = bank
        self.version = version if use_adapter else None
        self.input_prefix = prefix.format(version=version)
        self.python_version = version
        if bank.registry is not None:
            self.generation_kwargs = {"bad_words_ids": [[identifier] for identifier in bank.registry.forbidden_ids]}
            # Trained BPE tokenizers can flag ordinary Python tokens as special.
            # Discard only actual control tokens and input-only opcode tokens.
            control_roles = ("bos", "eos", "pad", "unk", "mask", "cls", "sep")
            control_ids = (getattr(tokenizer, f"{role}_token_id", None) for role in control_roles)
            self.decode_excluded_ids = {identifier for identifier in control_ids if identifier is not None} | set(bank.registry.forbidden_ids)

    def _prepare_bytecode(self, text):
        return self.bank.bind_bytecode(text, self.python_version)

    def __call__(self, *args, **kwargs):
        # The output cache and all generation, including retries, share the lock.
        with self.bank.use(self.version):
            return super().__call__(*args, **kwargs)


@lru_cache(maxsize=1)
def load_models(
    config_file: Path = Path("pylingual/decompiler_config.yaml"),
    version: PythonVersion = PythonVersion(3.9),
    token=False,
) -> tuple[transformers.Pipeline, CacheTranslator]:
    logger.info(f"Loading models for {version}...")
    with config_file.open() as f:
        config = yaml.safe_load(f)
        shared = config.get("SHARED_BACKBONES", {})
        legacy = config.get(f"v{version}", {})

    from pylingual.shared_models import load_family, load_version
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    root = Path(config_file).resolve().parent
    use_adapters = shared.get("USE_ADAPTERS", True)
    use_statement_adapters = shared.get("USE_STATEMENT_ADAPTERS", use_adapters)
    use_segmentation_adapters = shared.get("USE_SEGMENTATION_ADAPTERS", use_adapters)
    if "SEGMENTATION_MODEL" in shared:
        if use_segmentation_adapters:
            segmenter = load_version(root / shared["SEGMENTATION_MODEL"], "segmentation", str(version), device)
        else:
            from pylingual.lora import VersionedPipeline
            bank, _, _, pipeline = load_family(root / shared["SEGMENTATION_MODEL"], "segmentation", device)
            segmenter = pipeline.for_version(None, python_version=str(version)) if hasattr(pipeline, "for_version") else VersionedPipeline(pipeline, bank, None)
    else:
        seg_config = legacy["SEGMENTATION_MODEL"]
        segmenter = _load_legacy_segmenter(seg_config, device, token)
    if "STATEMENT_MODEL" in shared:
        if use_statement_adapters:
            bank, tokenizer, metadata = load_version(root / shared["STATEMENT_MODEL"], "statement", str(version), device)
        else:
            bank, tokenizer, metadata, _ = load_family(root / shared["STATEMENT_MODEL"], "statement", device)
        return segmenter, VersionedTranslator(bank, tokenizer, device, str(version), metadata.get("source_prefix", ""), use_adapter=use_statement_adapters)
    stmt_config = legacy["STATEMENT_MODEL"]
    translation_model = transformers.T5ForConditionalGeneration.from_pretrained(stmt_config["REPO"], revision=stmt_config["REVISION"], token=token)
    translation_tokenizer = transformers.RobertaTokenizer.from_pretrained(stmt_config["TOKENIZER"], token=token)
    return segmenter, CacheTranslator(translation_model, translation_tokenizer, device)


def _load_legacy_segmenter(seg_config, device, token):

    #################################
    # Segmentation model components #
    #################################
    segmentation_model = transformers.AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=seg_config["REPO"],
        revision=seg_config["REVISION"],
        token=token,
    )
    tokenizer_directory = Path(seg_config["TOKENIZER"])
    segmentation_tokenizer_file = str(tokenizer_directory / "tokenizer.json") if tokenizer_directory.is_dir() else huggingface_hub.hf_hub_download(repo_id=seg_config["TOKENIZER"], filename="tokenizer.json", token=token)
    segmentation_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=segmentation_tokenizer_file,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return transformers.pipeline(
        "token-classification",
        model=segmentation_model,
        tokenizer=segmentation_tokenizer,
        aggregation_strategy="none",
        device=device,
    )
