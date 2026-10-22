"""Public API, loaded lazily so tokenizer tooling needs no decompiler/GPU stack."""
__all__ = ["decompile"]


def __getattr__(name):
    if name == "decompile":
        from .decompiler import decompile
        globals()[name] = decompile
        return decompile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
