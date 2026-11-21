# SPDX-License-Identifier: Apache-2.0
"""Batch KV caches whose rows end at different columns (Inkling 2).

Batched speculative verify accepts a different number of drafts per row. A
plain ``BatchKVCache`` shares one end column, so a ragged rollback would have
to shift every short row (a full KV copy per layer visit and verify cycle).
Here each row keeps its own end: trims only move the ends, the next tokens of
each row are written at its own end, and Inkling's small-query attention
(``fused.small_attention``) reads the row ends. Anything else that looks at
the cache (prefill-sized attention, filter, extend, extract, state) first
aligns the rows, which shifts the short rows right once, as ``finalize`` does
for right padding.

Batch caches are upgraded in place (``upgrade``) the first time a rollback
needs different counts per row, so cache construction stays untouched.
"""

from __future__ import annotations

import mlx.core as mx


class RaggedRowsMixin:
    """Per-row end columns on top of a ``BatchKVCache``."""

    # fused_batch: this cache supports vector rollback through the model.
    _omlx_mtp_batch_rollback_cache = True

    @property
    def row_ends(self):
        return self.__dict__.get("_row_ends")

    def _ends(self) -> list[int]:
        ends = self.__dict__.get("_row_ends")
        if ends is not None:
            return list(ends)
        return [self._idx] * int(self.offset.shape[0])

    def _set_ends(self, ends: list[int]) -> None:
        self._idx = max(ends) if ends else 0
        self._row_ends = None if len(set(ends)) <= 1 else list(ends)

    def trim_rows(self, counts) -> int:
        """Drop the newest ``counts[b]`` tokens of each row ``b``."""
        counts = [max(0, int(n)) for n in counts]
        if self.__dict__.get("_row_ends") is None and len(set(counts)) == 1:
            return super().trim(counts[0])
        ends = self._ends()
        counts = [min(n, e) for n, e in zip(counts, ends)]
        self.offset = self.offset - mx.array(counts, dtype=self.offset.dtype)
        self._set_ends([e - n for e, n in zip(ends, counts)])
        return max(counts) if counts else 0

    def trim(self, n):
        if self.__dict__.get("_row_ends") is not None:
            return self.trim_rows([n] * len(self._row_ends))
        return super().trim(n)

    def update_and_fetch(self, keys, values):
        ends = self.__dict__.get("_row_ends")
        if ends is None:
            return super().update_and_fetch(keys, values)
        n = keys.shape[2]
        need = max(ends) + n
        if self.keys.shape[2] < need:
            step = self.step
            grow = ((need - self.keys.shape[2] + step - 1) // step) * step
            b, h, _, dk = self.keys.shape
            dv = self.values.shape[3]
            self.keys = mx.concatenate([self.keys, mx.zeros((b, h, grow, dk), self.keys.dtype)], axis=2)
            self.values = mx.concatenate(
                [self.values, mx.zeros((b, h, grow, dv), self.values.dtype)], axis=2
            )
        for row, e in enumerate(ends):
            self.keys[row : row + 1, :, e : e + n, :] = keys[row : row + 1]
            self.values[row : row + 1, :, e : e + n, :] = values[row : row + 1]
        self.offset = self.offset + n
        self._set_ends([e + n for e in ends])
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def align(self) -> None:
        """Shift short rows right so every row ends at ``_idx`` again."""
        ends = self.__dict__.get("_row_ends")
        if ends is None:
            return
        end = self._idx
        mx.eval(self.left_padding)
        pads = [int(p) for p in self.left_padding.tolist()]
        for row, (e, lp) in enumerate(zip(ends, pads)):
            shift = end - e
            if shift <= 0:
                continue
            self.keys[row : row + 1, :, lp + shift : end, :] = self.keys[row : row + 1, :, lp:e, :]
            self.values[row : row + 1, :, lp + shift : end, :] = self.values[row : row + 1, :, lp:e, :]
        self.left_padding = mx.array([lp + end - e for e, lp in zip(ends, pads)], dtype=self.left_padding.dtype)
        self._row_ends = None
        mx.eval(self.keys, self.values, self.left_padding)

    # Everything outside decode/verify sees aligned rows.
    def filter(self, batch_indices):
        self.align()
        return super().filter(batch_indices)

    def extend(self, other):
        self.align()
        if hasattr(other, "align"):
            other.align()
        return super().extend(other)

    def extract(self, idx):
        self.align()
        return super().extract(idx)

    def prepare(self, *args, **kwargs):
        self.align()
        return super().prepare(*args, **kwargs)

    def make_mask(self, *args, **kwargs):
        self.align()
        return super().make_mask(*args, **kwargs)

    def size(self):
        self.align()
        return super().size()

    def keys_and_values(self):
        self.align()
        return super().keys_and_values()

    @property
    def state(self):
        self.align()
        return super().state

    @state.setter
    def state(self, v):
        self.__dict__.pop("_row_ends", None)
        type(self).__mro__[2].state.fset(self, v)


_classes: dict = {}


def upgrade(kv):
    """Give a batch KV cache per-row ends (in place); other caches pass through."""
    if isinstance(kv, RaggedRowsMixin):
        return kv
    base = type(kv)
    if not hasattr(kv, "left_padding") or not hasattr(kv, "_idx"):
        return kv
    cls = _classes.get(base)
    if cls is None:
        cls = _classes[base] = type(f"Inkling2Ragged{base.__name__}", (RaggedRowsMixin, base), {})
    kv.__class__ = cls
    kv._row_ends = None
    return kv
