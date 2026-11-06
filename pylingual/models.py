from __future__ import annotations

import functools
import gc
import hashlib
import json
import yaml
import logging
import threading

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

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
    import redis
    import torch
    import transformers
    import huggingface_hub
else:
    lazy_import("redis")
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

    def __init__(self, model: transformers.T5ForConditionalGeneration, tokenizer: transformers.RobertaTokenizer, device="cpu", maxsize=50000, *, python_version: str | None = None, redis_cache_server_ip: str = None, redis_port: int = 1679):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.input_prefix = ""
        self.generation_kwargs = {}
        self.decode_excluded_ids = {0, 1, 2, 3, 4}
        self.model.to(self.device)

        # Redis Caching
        self.redis_enabled = redis_cache_server_ip is not None

        if self.redis_enabled:
            self.redis_db = redis.StrictRedis(host=redis_cache_server_ip, port=redis_port, db=0, encoding="utf-8", decode_responses=True)
            self.redis_namespace_prefix = f"{python_version}:"

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

    def _translate_with_backoff(self, translation_requests: TrackedDataset | list[str], batch_size: int = 32) -> list[str]:
        translation_results = []
        # process in batches
        for i in range(0, len(translation_requests), batch_size):
            batch = [translation_requests[j] for j in range(i, min(i + batch_size, len(translation_requests)))]
            try:
                translation_results.extend(self._translate_and_decode(batch))
                continue
            except TimeoutError:
                raise
            except Exception as e:
                error = repr(e)
            # handled outside the except block so tensors can be freed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) == 1:
                # fallback
                logger.info(f"Could not translate statement ({error})")
                translation_results.append("'''Decompiler error: line too long for translation. Please decompile this statement manually.'''")
            else:
                # retry with a smaller batch size
                next_size = max(1, len(batch) // 2)
                logger.info(f"Lowering translation batch size to {next_size} ({error})")
                translation_results.extend(self._translate_with_backoff(batch, batch_size=next_size))
        return translation_results

    def __call__(self, args: list, check_timeout: callable = None, **_):
        normalized_args = [normalize_masks(fix_jump_targets(x)) for x in args]

        if self.redis_enabled:
            # Update local cache from redis
            to_fetch_from_redis = list({norm for norm, _ in normalized_args if norm not in self.cache})
            redis_results = {key: self.redis_db.get(f"{self.redis_namespace_prefix}{key}") for key in to_fetch_from_redis}
            for translation_key, translation_result in redis_results.items():
                if translation_result:
                    self.cache[translation_key] = translation_result

        # New are those not in the local cache
        new = TrackedDataset(
            TRANSLATION_STEP,
            # sorted by length so batches need less padding; order doesn't matter since results are keyed by request
            sorted({norm for norm, _ in normalized_args if norm not in self.cache}, key=len),
            check_timeout=check_timeout,
        )

        # Now, "new" has been updated to those not in local
        for arg, result in zip(new.x, self._translate_with_backoff(new)):
            self.cache[arg] = result
            if self.redis_enabled:
                self.redis_db.set(f"{self.redis_namespace_prefix}{arg}", self.cache[arg])

        results = [restore_masks(self[norm], order) for norm, order in normalized_args]
        while len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return results

# offline support if models are downloaded
def _cached_first(fn, *args, **kwargs):
    """Try the local HF cache first; fall back to downloading if it's missing."""
    try:
        result = fn(*args, local_files_only=True, **kwargs)
        # tokenizers don't raise when their files are missing, they come back with only the special tokens
        if isinstance(result, transformers.PreTrainedTokenizerBase) and len(result) <= len(result.all_special_tokens):
            raise OSError("tokenizer has no vocabulary")
        return result
    except (OSError, ValueError):
        logger.info(f"{getattr(fn, '__qualname__', fn)} not found in local cache for {args[:1]}, downloading...")
        return fn(*args, local_files_only=False, **kwargs)

class VersionedTranslator(CacheTranslator):
    """A per-version output cache over one shared, unmerged backbone."""
    def __init__(self, bank, tokenizer, device, version, prefix, use_adapter=True, *, redis_cache_server_ip=None, redis_port=1679, cache_namespace=None):
        super().__init__(bank.model, tokenizer, device=device, python_version=version, redis_cache_server_ip=redis_cache_server_ip, redis_port=redis_port)
        if self.redis_enabled:
            self.redis_namespace_prefix = f"shared:{cache_namespace or id(bank)}:{version}:{use_adapter}:"
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


def _free_device_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _single_slot_cache(fn):
    """Keep only the latest models loaded, freeing the old ones first; falls back to CPU on GPU OOM."""
    lock = threading.Lock()
    loaded_key = loaded_models = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal loaded_key, loaded_models
        key = (args, tuple(sorted(kwargs.items())))
        with lock:
            if loaded_key == key:
                return loaded_models
            loaded_key = loaded_models = None
            _free_device_memory()
            try:
                loaded_models = fn(*args, **kwargs)
            except torch.OutOfMemoryError:
                logger.warning("Not enough GPU memory to load models, falling back to CPU")
                _free_device_memory()
                loaded_models = fn(*args, **{**kwargs, "device": torch.device("cpu")})
            loaded_key = key
            return loaded_models

    def cache_clear():
        nonlocal loaded_key, loaded_models
        with lock:
            loaded_key = loaded_models = None
            _free_device_memory()
    wrapper.cache_clear = cache_clear
    return wrapper

@_single_slot_cache
def load_models(
    config_file: Path = Path("pylingual/decompiler_config.yaml"),
    version: PythonVersion = PythonVersion(3.9),
    redis_cache_server_ip: str = None,
    redis_port: int = 1679,
    token=False,
    device: torch.device | None = None,
) -> tuple[transformers.Pipeline, CacheTranslator]:
    logger.info(f"Loading models for {version}...")
    with config_file.open() as f:
        config = yaml.safe_load(f)
        shared = config.get("SHARED_BACKBONES", {})
        legacy = config.get(f"v{version}", {})

    from pylingual.shared_models import load_family, load_version
    # Keep upstream's explicit-device and CUDA/CPU policy (MPS is deliberately avoided).
    device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        family_identity = str((root / shared["STATEMENT_MODEL"]).resolve()) + json.dumps(metadata, sort_keys=True)
        namespace = hashlib.sha256(family_identity.encode()).hexdigest()
        return segmenter, VersionedTranslator(bank, tokenizer, device, str(version), metadata.get("source_prefix", ""), use_adapter=use_statement_adapters, redis_cache_server_ip=redis_cache_server_ip, redis_port=redis_port, cache_namespace=namespace)
    stmt_config = legacy["STATEMENT_MODEL"]
    translation_model = _cached_first(transformers.T5ForConditionalGeneration.from_pretrained, stmt_config["REPO"], revision=stmt_config["REVISION"], token=token)
    translation_tokenizer = _cached_first(transformers.RobertaTokenizer.from_pretrained, stmt_config["TOKENIZER"], token=token)
    return segmenter, CacheTranslator(translation_model, translation_tokenizer, device=device, python_version=str(version), redis_cache_server_ip=redis_cache_server_ip, redis_port=redis_port)


def _load_legacy_segmenter(seg_config, device, token):

    #################################
    # Segmentation model components #
    #################################
    segmentation_model = _cached_first(
        transformers.AutoModelForTokenClassification.from_pretrained,
        pretrained_model_name_or_path=seg_config["REPO"],
        revision=seg_config["REVISION"],
        token=token,
    )
    tokenizer_directory = Path(seg_config["TOKENIZER"])
    segmentation_tokenizer_file = str(tokenizer_directory / "tokenizer.json") if tokenizer_directory.is_dir() else _cached_first(huggingface_hub.hf_hub_download, repo_id=seg_config["TOKENIZER"], filename="tokenizer.json", token=token)
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
