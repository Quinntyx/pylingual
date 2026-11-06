"""Adapter-local sparse input corrections; shared/tied embedding tables stay frozen."""
from __future__ import annotations
import weakref
import torch
from torch import nn


class TokenRows(nn.Module):
    def __init__(self, embedding, ids):
        super().__init__()
        self.register_buffer("token_ids", torch.tensor(ids, device=embedding.weight.device, dtype=torch.long))
        self.delta = nn.Parameter(embedding.weight.new_zeros(len(ids), embedding.embedding_dim))

    def forward(self, input_ids):
        positions = torch.searchsorted(self.token_ids, input_ids.contiguous()).clamp(max=len(self.token_ids)-1)
        matches = self.token_ids[positions] == input_ids
        return self.delta[positions] * matches.unsqueeze(-1)


class EncoderTokenDeltas(nn.Module):
    """Hooks encoder inputs only; never replace a tied embedding module or its weight."""
    def __init__(self, model, task):
        super().__init__()
        self.rows = nn.ModuleDict()
        self.active = None
        self._embedding = weakref.ref(model.get_input_embeddings())
        self.task = task
        if task == "statement":
            if not hasattr(model, "get_encoder"):
                raise ValueError("Statement embedding corrections require an encoder-decoder model")
            self._handle = model.get_encoder().register_forward_pre_hook(self._encoder_hook, with_kwargs=True)
        elif task == "segmentation":
            self._handle = model.get_input_embeddings().register_forward_hook(self._embedding_hook)
        else:
            raise ValueError("Unknown input embedding task")

    def _selected(self):
        if self.active is None:
            return None
        rows = self.rows[self.active]
        return rows if rows.token_ids.numel() else None

    def add(self, key, ids):
        rows = TokenRows(self._embedding(), ids)
        rows.train(self.training)
        self.rows[key] = rows

    def _encoder_hook(self, module, args, kwargs):
        rows = self._selected()
        if rows is None:
            return args, kwargs
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise ValueError("New-token embedding updates require encoder input_ids, not precomputed inputs_embeds")
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("Cannot combine opcode corrections with precomputed inputs_embeds")
        output = self._embedding()(ids)
        kwargs = dict(kwargs)
        kwargs["inputs_embeds"] = output + rows(ids).to(output.dtype)
        if args:
            args = (None,) + args[1:]
        else:
            kwargs["input_ids"] = None
        return args, kwargs

    def _embedding_hook(self, module, args, output):
        rows = self._selected()
        return output if rows is None else output + rows(args[0]).to(output.dtype)
