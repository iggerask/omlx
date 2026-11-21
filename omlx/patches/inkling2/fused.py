# SPDX-License-Identifier: Apache-2.0
"""Fused Metal kernels for Inkling 2 decode and speculative verify.

A decode step runs 48 layer visits and is bound by kernel launches, not by
weight reads: the stock graph issues ~4,300 GPU kernels per token, most of
them tiny elementwise, cast and reduction ops on [B, L, hidden] vectors.
These kernels replace groups of them for short inputs (decode, verify and
small batches, ``L <= MAX_TOKENS``). Prefill (longer inputs) uses the
long variants: ``static_conv_long`` / ``dynamic_conv_long`` (one thread per
channel and token), ``head_norms`` at any length (part ``headslong``), the
SwiGLU and expert-sum kernels around the sorted expert matmuls (``moelong``)
and ``mix_messages_long`` (``msglong``).

- ``static_conv``: a depthwise causal short conv in one kernel (padded window,
  taps, weight-dtype rounding, residual, scale) that also writes the padded
  window the cache keeps.
- ``head_norms``: the per-head L2 clips of q/k/r, the unit RMS of v and the
  long-position query scaling in one kernel; q/k/v come out head-major.

Set ``OMLX_INKLING2_FUSED=0`` to keep the stock path, or list parts in
``OMLX_INKLING2_FUSED_OFF`` (conv, dconv, convlong, heads, headslong, moe,
moelong, state, msg, msglong, attn, rows).
"""

from __future__ import annotations

import math
import os

import mlx.core as mx

MAX_TOKENS = 16
_ONE = mx.ones((1,), dtype=mx.float32)
mx.eval(_ONE)


# Parts that stay off unless OMLX_INKLING2_FUSED_ON lists them. Batched DSpark
# verify works but loses to plain batched decode on this model: every extra
# verified token of a batch reads about as many new experts as a plain token.
_DEFAULT_OFF = {"batch"}


def enabled(part: str = "") -> bool:
    """``OMLX_INKLING2_FUSED=0`` turns every fusion off; ``..._OFF=moe,state``
    turns off single parts and ``..._ON=batch`` turns on default-off parts."""
    if os.environ.get("OMLX_INKLING2_FUSED", "1") == "0":
        return False
    if not part:
        return True
    if part in os.environ.get("OMLX_INKLING2_FUSED_OFF", "").split(","):
        return False
    if part in _DEFAULT_OFF:
        return part in os.environ.get("OMLX_INKLING2_FUSED_ON", "").split(",")
    return True


# ---------------------------------------------------------------------------
# Short convolution
# ---------------------------------------------------------------------------

_CONV_SOURCE = """
    const uint c = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    const int C = x_shape[2];
    const int L = x_shape[1];
    if ((int)c >= C) return;
    constexpr int KM = K - 1;
    float win[KM + MAXL];
    for (int j = 0; j < KM; ++j) {
        win[j] = state[b * state_strides[0] + j * state_strides[1] + c * state_strides[2]];
    }
    for (int t = 0; t < L; ++t) {
        win[KM + t] = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
    }
    const int R = KM + L;
    for (int r = 0; r < R; ++r) {
        xp[(b * R + r) * C + c] = win[r];
    }
    float wf[K];
    for (int j = 0; j < K; ++j) {
        wf[j] = float(w[c * w_strides[0] + j * w_strides[1]]);
    }
    for (int t = 0; t < L; ++t) {
        float acc = wf[0] * win[t];
        for (int j = 1; j < K; ++j) {
            acc = acc + wf[j] * win[t + j];
        }
        const float o = float(WT(acc));
        out[(b * L + t) * C + c] = OT((o + win[KM + t]) * SCALE);
    }
"""
_conv_kernels: dict = {}


def static_conv(x: mx.array, w: mx.array, state: mx.array, scale: float):
    """Fused ``model._static_conv``: returns ``(out, xp)`` with ``xp`` the FP32
    padded window ``[state, x]``."""
    b, length, c = x.shape
    k = w.shape[-1]
    key = (k, float(scale))
    kern = _conv_kernels.get(key)
    if kern is None:
        src = _CONV_SOURCE.replace("SCALE", f"{float(scale)!r}f")
        kern = _conv_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_static_conv_k{k}_{len(_conv_kernels)}",
            input_names=["x", "state", "w"],
            output_names=["out", "xp"],
            source=src,
            ensure_row_contiguous=False,
        )
    if state.dtype != mx.float32:
        state = state.astype(mx.float32)
    out, xp = kern(
        inputs=[x, state, w],
        template=[("K", k), ("MAXL", MAX_TOKENS), ("WT", w.dtype), ("OT", x.dtype)],
        grid=(c, b, 1),
        threadgroup=(min(256, c), 1, 1),
        output_shapes=[(b, length, c), (b, k - 1 + length, c)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out, xp


# ---------------------------------------------------------------------------
# Head norms: q/k/r L2 clips, v unit RMS, query/relative log scaling
# ---------------------------------------------------------------------------

# One simdgroup per (b, t, head) row of q, k, v or r. Rows are laid out
# [q heads | k heads | v heads | r heads] for every (b, t).
_HEADS_SOURCE = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const int L = q_shape[1];
    constexpr int ROWS = HQ + 2 * HKV + HQ;
    const int bt = row / ROWS;
    const int j = row % ROWS;
    const int b = bt / L;
    const int t = bt % L;
    const float tf = HAS_TAU ? tau[bt] : 1.0f;
    if (j < HQ) {
        // q: clip(x * q_scale), then * tau; written [B, HQ, L, D].
        const int h = j;
        float v[D / 32];
        float ss = 0.0f;
        for (int i = 0; i < D / 32; ++i) {
            const int d = lane + 32 * i;
            const float xv = float(q[b * q_strides[0] + t * q_strides[1] + (h * D + d) * q_strides[2]])
                * float(q_scale[h * q_scale_strides[0] + d * q_scale_strides[1]]);
            v[i] = xv;
            ss += xv * xv;
        }
        ss = simd_sum(ss);
        const float f = min(QK_CLIP / (sqrt(ss) + 1e-6f), 1.0f);
        for (int i = 0; i < D / 32; ++i) {
            const int d = lane + 32 * i;
            T o = T(v[i] * f);
            if (HAS_TAU) o = T(float(o) * tf);
            qo[((b * HQ + h) * L + t) * D + d] = o;
        }
    } else if (j < HQ + HKV) {
        // k: clip(x), written [B, HKV, L, D].
        const int h = j - HQ;
        float v[D / 32];
        float ss = 0.0f;
        for (int i = 0; i < D / 32; ++i) {
            const int d = lane + 32 * i;
            const float xv = float(k[b * k_strides[0] + t * k_strides[1] + (h * D + d) * k_strides[2]]);
            v[i] = xv;
            ss += xv * xv;
        }
        ss = simd_sum(ss);
        const float f = min(QK_CLIP / (sqrt(ss) + 1e-6f), 1.0f);
        for (int i = 0; i < D / 32; ++i) {
            const int d = lane + 32 * i;
            ko[((b * HKV + h) * L + t) * D + d] = T(v[i] * f);
        }
    } else if (j < HQ + 2 * HKV) {
        // v: unit RMS in FP32, written [B, HKV, L, D].
        const int h = j - HQ - HKV;
        float v[D / 32];
        float ss = 0.0f;
        for (int i = 0; i < D / 32; ++i) {
            const int d = lane + 32 * i;
            const float xv = float(vv[b * vv_strides[0] + t * vv_strides[1] + (h * D + d) * vv_strides[2]]);
            v[i] = xv;
            ss += xv * xv;
        }
        ss = simd_sum(ss);
        const float f = metal::precise::rsqrt(ss / float(D) + 1e-6f);
        for (int i = 0; i < D / 32; ++i) {
            const int d = lane + 32 * i;
            vo[((b * HKV + h) * L + t) * D + d] = T(v[i] * f);
        }
    } else {
        // r: clip(x * r_scale), then * tau; written [B, L, HQ, DR].
        const int h = j - HQ - 2 * HKV;
        float xv = 0.0f;
        if (lane < DR) {
            xv = float(r[b * r_strides[0] + t * r_strides[1] + (h * DR + lane) * r_strides[2]])
                * float(r_scale[h * r_scale_strides[0] + lane * r_scale_strides[1]]);
        }
        const float ss = simd_sum(xv * xv);
        const float f = min(R_CLIP / (sqrt(ss) + 1e-6f), 1.0f);
        if (lane < DR) {
            T o = T(xv * f);
            if (HAS_TAU) o = T(float(o) * tf);
            ro[((b * L + t) * HQ + h) * DR + lane] = o;
        }
    }
"""
_head_kernels: dict = {}


def head_norms(q, k, v, r, q_scale, r_scale, tau, *, n_heads, n_kv, head_dim, d_rel, qk_clip, r_clip):
    """Fused clips for ``Inkling2Attention.project``.

    ``q``/``r``: [B, L, H*D] / [B, L, H*d_rel] (strided views are fine),
    ``k``/``v``: [B, L, HKV*D]. ``tau``: [B, L, 1, 1] FP32 or None.
    Returns q [B, H, L, D], k/v [B, HKV, L, D] and r [B, L, H, d_rel].
    """
    b, length, _ = q.shape
    has_tau = tau is not None
    key = (n_heads, n_kv, head_dim, d_rel, float(qk_clip), float(r_clip), has_tau, q.dtype)
    kern = _head_kernels.get(key)
    if kern is None:
        src = _HEADS_SOURCE.replace("QK_CLIP", f"{float(qk_clip)!r}f").replace(
            "R_CLIP", f"{float(r_clip)!r}f"
        )
        kern = _head_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_head_norms_{len(_head_kernels)}",
            input_names=["q", "k", "vv", "r", "q_scale", "r_scale", "tau"],
            output_names=["qo", "ko", "vo", "ro"],
            source=src,
            ensure_row_contiguous=False,
        )
    if tau is None:
        tau_in = _ONE
    else:
        tau_in = mx.contiguous(mx.broadcast_to(tau.reshape(-1, length), (b, length)).astype(mx.float32))
    rows = b * length * (2 * n_heads + 2 * n_kv)
    dt = q.dtype
    qo, ko, vo, ro = kern(
        inputs=[q, k, v, r, q_scale, r_scale, tau_in],
        template=[
            ("T", dt),
            ("HQ", n_heads),
            ("HKV", n_kv),
            ("D", head_dim),
            ("DR", d_rel),
            ("HAS_TAU", has_tau),
        ],
        grid=(32 * rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (b, n_heads, length, head_dim),
            (b, n_kv, length, head_dim),
            (b, n_kv, length, head_dim),
            (b, length, n_heads, d_rel),
        ],
        output_dtypes=[dt, dt, dt, dt],
    )
    return qo, ko, vo, ro


# ---------------------------------------------------------------------------
# MoE: NVFP4 tensor scale + softcapped SwiGLU, and the weighted expert sum
# ---------------------------------------------------------------------------

# y: [N, K, 1, 2I] interleaved gate/up rows; act: [N, K, 1, I].
_SWIGLU_SOURCE = """
    const uint i = thread_position_in_grid.x;
    const uint nk = thread_position_in_grid.y;
    if ((int)i >= I) return;
    const size_t base = (size_t)nk * (2 * I);
    float g = float(y[base + 2 * i]);
    float u = float(y[base + 2 * i + 1]);
    if (HAS_SCALE) {
        const T s = T(gscale[inds[nk]] * INV_DENOM);
        g = float(T(g * float(s)));
        u = float(T(u * float(s)));
    }
    const float o = metal::precise::tanh(g / GCAP) * GCAP * (1.0f / (1.0f + metal::precise::exp(-g)))
        * (metal::precise::tanh(u / UCAP) * UCAP);
    act[(size_t)nk * I + i] = T(o);
"""

# z: [N, K, 1, H]; out[n, h] = sum_k z * w[n, k] * (s[inds[n, k]] / denom).
_COMBINE_SOURCE = """
    const uint h = thread_position_in_grid.x;
    const uint n = thread_position_in_grid.y;
    if ((int)h >= H) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float w = weights[n * K + k];
        if (HAS_SCALE) w = w * (gscale[inds[n * K + k]] * INV_DENOM);
        const size_t row = PERM ? (size_t)perm[n * K + k] : (size_t)n * K + k;
        acc += float(z[row * H + h]) * w;
    }
    out[(size_t)n * H + h] = OT(acc);
"""
_moe_kernels: dict = {}


def _moe_kernel(name, source, inputs, outputs, consts):
    key = (name, consts)
    kern = _moe_kernels.get(key)
    if kern is None:
        src = source
        for k, v in consts:
            src = src.replace(k, f"{float(v)!r}f")
        kern = _moe_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_{name}_{len(_moe_kernels)}",
            input_names=inputs,
            output_names=outputs,
            source=src,
        )
    return kern


def swiglu(y, inds, gscale, inv_denom, gcap, ucap):
    """``y * (gscale[inds] * inv_denom)`` (BF16-rounded like the reference)
    then the softcapped SwiGLU over interleaved gate/up columns."""
    n, k, _, two_i = y.shape
    i = two_i // 2
    has = gscale is not None
    kern = _moe_kernel(
        "swiglu", _SWIGLU_SOURCE, ["y", "inds", "gscale"], ["act"],
        (("INV_DENOM", inv_denom), ("GCAP", gcap), ("UCAP", ucap)),
    )
    return kern(
        inputs=[y, inds, gscale if has else _ONE],
        template=[("T", y.dtype), ("I", i), ("HAS_SCALE", has)],
        grid=(i, n * k, 1),
        threadgroup=(min(256, i), 1, 1),
        output_shapes=[(n, k, 1, i)],
        output_dtypes=[y.dtype],
    )[0]


def combine(z, inds, weights, gscale, inv_denom, out_dtype, perm=None):
    """Weighted sum over the top-k expert outputs with the NVFP4 tensor scale.

    With ``perm``, ``z`` holds the outputs in expert-sorted order and the
    output of (token n, slot k) is row ``perm[n * K + k]``."""
    n, k = inds.shape[0], inds.shape[1]
    h = z.shape[-1]
    has = gscale is not None
    kern = _moe_kernel(
        "combine", _COMBINE_SOURCE, ["z", "inds", "weights", "gscale", "perm"], ["out"],
        (("INV_DENOM", inv_denom),),
    )
    return kern(
        inputs=[z, inds, weights, gscale if has else _ONE, perm if perm is not None else _ONE],
        template=[("OT", out_dtype), ("H", h), ("K", k), ("HAS_SCALE", has), ("PERM", perm is not None)],
        grid=(h, n, 1),
        threadgroup=(min(256, h), 1, 1),
        output_shapes=[(n, h)],
        output_dtypes=[out_dtype],
    )[0]


# ---------------------------------------------------------------------------
# Dynamic short conv (full-attention k/v): taps static_w[lag] + dyn[head, lag]
# ---------------------------------------------------------------------------

_DCONV_SOURCE = """
    const uint c = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    const int C = x_shape[2];
    const int L = x_shape[1];
    if ((int)c >= C) return;
    constexpr int KM = K - 1;
    const int head = c / (C / HEADS);
    float win[KM + MAXL];
    for (int j = 0; j < KM; ++j) {
        win[j] = state[b * state_strides[0] + j * state_strides[1] + c * state_strides[2]];
    }
    for (int t = 0; t < L; ++t) {
        win[KM + t] = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
    }
    const int R = KM + L;
    for (int r = 0; r < R; ++r) {
        xp[(b * R + r) * C + c] = win[r];
    }
    float sw[K];
    for (int lag = 0; lag < K; ++lag) {
        sw[lag] = float(w[lag * w_strides[0] + c * w_strides[1]]);
    }
    for (int t = 0; t < L; ++t) {
        float acc = 0.0f;
        for (int lag = 0; lag < K; ++lag) {
            const float d = float(dyn[b * dyn_strides[0] + t * dyn_strides[1] + (head * K + lag) * dyn_strides[2]]);
            const float term = win[KM + t - lag] * (sw[lag] + d);
            acc = lag == 0 ? term : acc + term;
        }
        out[(b * L + t) * C + c] = OT((acc + win[KM + t]) * SCALE);
    }
"""
_dconv_kernels: dict = {}


def dynamic_conv(x, dyn, static_w, state, heads: int, scale: float):
    """Fused ``model._dynamic_conv``: returns ``(out, xp)``."""
    b, length, c = x.shape
    k = static_w.shape[0]
    key = (k, heads, float(scale))
    kern = _dconv_kernels.get(key)
    if kern is None:
        kern = _dconv_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_dynamic_conv_{len(_dconv_kernels)}",
            input_names=["x", "dyn", "w", "state"],
            output_names=["out", "xp"],
            source=_DCONV_SOURCE.replace("SCALE", f"{float(scale)!r}f"),
            ensure_row_contiguous=False,
        )
    if state.dtype != mx.float32:
        state = state.astype(mx.float32)
    return kern(
        inputs=[x, dyn, static_w, state],
        template=[("K", k), ("HEADS", heads), ("MAXL", MAX_TOKENS), ("OT", x.dtype)],
        grid=(c, b, 1),
        threadgroup=(min(256, c), 1, 1),
        output_shapes=[(b, length, c), (b, k - 1 + length, c)],
        output_dtypes=[x.dtype, mx.float32],
    )


# ---------------------------------------------------------------------------
# Loop message gates: (1 - tanh(forget)) * x + tanh(gain) * message
# ---------------------------------------------------------------------------

# Stage 1: the down projection split over hidden chunks so the 2 x H x r
# weights stream coalesced through many threadgroups. Each threadgroup (one
# thread per rank output) recomputes the unit RMS of its row, rounds the
# normalized inputs to BF16 like the reference, and writes a partial sum.
_MSG_DOWN_SOURCE = """
    const uint g = threadgroup_position_in_grid.x;
    const uint nr = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const int n = nr / ROWS;
    threadgroup float red[2 * (RANK / 32)];
    const device float* xr = x + (size_t)nr * H;
    const device float* mr = msg + (size_t)nr * H;
    float sx = 0.0f, sm = 0.0f;
    for (int i = tid; i < H; i += RANK) {
        const float a = xr[i];
        const float m = mr[i];
        sx += a * a;
        sm += m * m;
    }
    sx = simd_sum(sx);
    sm = simd_sum(sm);
    if (tid % 32 == 0) { red[tid / 32] = sx; red[RANK / 32 + tid / 32] = sm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tx = 0.0f, tm = 0.0f;
    for (int i = 0; i < RANK / 32; ++i) { tx += red[i]; tm += red[RANK / 32 + i]; }
    const float fx = metal::precise::rsqrt(tx / float(H) + 1e-6f);
    const float fm = metal::precise::rsqrt(tm / float(H) + 1e-6f);
    const device WT* d0 = down + (size_t)n * 2 * H * RANK;
    const device WT* d1 = d0 + (size_t)H * RANK;
    float acc0 = 0.0f, acc1 = 0.0f;
    const int i0 = g * CH;
    for (int i = i0; i < i0 + CH; ++i) {
        const float zx = float(bfloat16_t(xr[i] * fx));
        const float zm = float(bfloat16_t(mr[i] * fm));
        acc0 += zx * float(d0[(size_t)i * RANK + tid]);
        acc1 += zm * float(d1[(size_t)i * RANK + tid]);
    }
    part[((size_t)nr * G + g) * RANK + tid] = acc0;
    part[(((size_t)NR + nr) * G + g) * RANK + tid] = acc1;
"""

# Stage 2, one thread per (n, row, h): reduce the partials to the BF16-rounded
# low-rank state, then gain/forget columns h and H + h and the mix.
_MSG_UP_SOURCE = """
    const uint h = thread_position_in_grid.x;
    const uint nr = thread_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const int n = nr / ROWS;
    threadgroup float lr[RANK];
    if (tid < RANK) {
        float a0 = 0.0f, a1 = 0.0f;
        for (int g = 0; g < G; ++g) {
            a0 += part[((size_t)nr * G + g) * RANK + tid];
            a1 += part[(((size_t)NR + nr) * G + g) * RANK + tid];
        }
        lr[tid] = float(bfloat16_t(a0 + a1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)h >= H) return;
    const device WT* gf = gain_forget + (size_t)n * RANK * 2 * H;
    float ga = 0.0f, fo = 0.0f;
    for (int k = 0; k < RANK; ++k) {
        const float l = lr[k];
        ga += l * float(gf[(size_t)k * 2 * H + h]);
        fo += l * float(gf[(size_t)k * 2 * H + H + h]);
    }
    const float gain = metal::precise::tanh(ga);
    const float forget = metal::precise::tanh(fo);
    const size_t i = (size_t)nr * H + h;
    out[i] = (1.0f - forget) * x[i] + gain * msg[i];
"""
_msg_kernels: dict = {}
_MSG_CHUNK = 64


def mix_messages(x, message, down, gain_forget):
    """Fused ``model._mix_messages`` for [N, B, L, H] FP32 inputs."""
    n, b, length, hidden = x.shape
    rows = b * length
    rank = down.shape[-1]
    ch = _MSG_CHUNK if hidden % _MSG_CHUNK == 0 else 1
    groups = hidden // ch
    key = (hidden, rank, rows, n, down.dtype)
    kerns = _msg_kernels.get(key)
    if kerns is None:
        kerns = _msg_kernels[key] = (
            mx.fast.metal_kernel(
                name=f"inkling2_msg_down_{len(_msg_kernels)}",
                input_names=["x", "msg", "down"],
                output_names=["part"],
                source=_MSG_DOWN_SOURCE,
            ),
            mx.fast.metal_kernel(
                name=f"inkling2_msg_up_{len(_msg_kernels)}",
                input_names=["part", "gain_forget", "x", "msg"],
                output_names=["out"],
                source=_MSG_UP_SOURCE,
            ),
        )
    x = x.astype(mx.float32)
    message = message.astype(mx.float32)
    nr = n * rows
    part = kerns[0](
        inputs=[x, message, down],
        template=[("H", hidden), ("RANK", rank), ("ROWS", rows), ("CH", ch), ("G", groups), ("NR", nr),
                  ("WT", down.dtype)],
        grid=(rank * groups, nr, 1),
        threadgroup=(rank, 1, 1),
        output_shapes=[(2, nr, groups, rank)],
        output_dtypes=[mx.float32],
    )[0]
    return kerns[1](
        inputs=[part, gain_forget, x, message],
        template=[("H", hidden), ("RANK", rank), ("ROWS", rows), ("G", groups), ("NR", nr),
                  ("WT", gain_forget.dtype)],
        grid=(hidden, nr, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )[0]


# ---------------------------------------------------------------------------
# Small-query attention (decode, verify, small batches)
# ---------------------------------------------------------------------------

# Split-key attention. Stage 1: one threadgroup per (b, kv head, block of query
# rows, key split). Query rows are the GROUP q heads of the kv head times the L
# positions, so each K/V row is read once for all of them. Each simdgroup
# streams every NSG-th key of the split with an online softmax per query row
# (lanes hold D/32 dims); the simdgroups merge into one partial (m, l, acc) per
# split. Stage 2 merges the splits and adds the zero-value sink once.
# Row b's query t sits at a = end[b] - L + t; keys j in [max(lp[b], a - W + 1), a]
# are visible, with the relative bias rel[b, t, h, a - j] for a - j < R.
_ATTN_SOURCE = """
    constexpr int DPL = D / 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int hk = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int qblk = threadgroup_position_in_grid.x / NSPLIT;
    const int split = threadgroup_position_in_grid.x % NSPLIT;
    const int L = params[0];
    const int span0 = params[1];
    const int chunk = params[2];
    const int lp = params[3 + b];
    const int end = params[3 + B + b];
    const int nrows = L * GROUP;
    const int r0 = qblk * QR;
    const int nq = min(QR, nrows - r0);
    const float qscale = params_f[0] * LOG2E;

    float qv[QR][DPL];
    float m[QR], l[QR], acc[QR][DPL];
    int qa[QR], qh[QR], qt[QR];
    for (int r = 0; r < QR; ++r) {
        m[r] = -INFINITY; l[r] = 0.0f;
        for (int i = 0; i < DPL; ++i) { acc[r][i] = 0.0f; qv[r][i] = 0.0f; }
        qa[r] = -1; qh[r] = 0; qt[r] = 0;
        if (r < nq) {
            const int row = r0 + r;
            const int t = row / GROUP;
            const int h = hk * GROUP + row % GROUP;
            qt[r] = t; qh[r] = h; qa[r] = end - L + t;
            const device T* qp = q + b * q_strides[0] + h * q_strides[1] + t * q_strides[2];
            for (int i = 0; i < DPL; ++i) {
                qv[r][i] = float(qp[(lane + 32 * i) * q_strides[3]]) * qscale;
            }
        }
    }
    int kmax = qa[0];
    int kmin = qa[0];
    for (int r = 1; r < nq; ++r) { kmax = max(kmax, qa[r]); kmin = min(kmin, qa[r]); }
    int lo = lp;
    if (WINDOW > 0) lo = max(lo, kmin - WINDOW + 1);
    const int s0 = span0 + split * chunk;
    const int j0 = max(lo, s0);
    const int j1 = min(kmax, s0 + chunk - 1);
    const device T* kb = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb = v + b * v_strides[0] + hk * v_strides[1];
    for (int j = j0 + (int)sg; j <= j1; j += NSG) {
        float kv[DPL], vv[DPL];
        for (int i = 0; i < DPL; ++i) {
            kv[i] = float(kb[j * k_strides[2] + (lane + 32 * i) * k_strides[3]]);
            vv[i] = float(vb[j * v_strides[2] + (lane + 32 * i) * v_strides[3]]);
        }
        for (int r = 0; r < QR; ++r) {
            if (r >= nq) break;
            const int d = qa[r] - j;
            if (d < 0 || (WINDOW > 0 && d >= WINDOW)) continue;
            float s = 0.0f;
            for (int i = 0; i < DPL; ++i) s += qv[r][i] * kv[i];
            s = simd_sum(s);
            if (d < R) {
                s += float(rel[((b * L + qt[r]) * HQ + qh[r]) * R + d]) * LOG2E;
            }
            const float mn = max(m[r], s);
            const float f = fast::exp2(m[r] - mn);
            const float p = fast::exp2(s - mn);
            l[r] = l[r] * f + p;
            for (int i = 0; i < DPL; ++i) acc[r][i] = acc[r][i] * f + p * vv[i];
            m[r] = mn;
        }
    }
    threadgroup float sm[NSG];
    threadgroup float sl[NSG];
    threadgroup float sacc[NSG * D];
    for (int r = 0; r < nq; ++r) {
        if (lane == 0) { sm[sg] = m[r]; sl[sg] = l[r]; }
        for (int i = 0; i < DPL; ++i) sacc[sg * D + lane + 32 * i] = acc[r][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float mt = -INFINITY;
            for (int g = 0; g < NSG; ++g) mt = max(mt, sm[g]);
            float lt = 0.0f;
            float o[DPL];
            for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
            if (mt > -INFINITY) {
                for (int g = 0; g < NSG; ++g) {
                    const float f = fast::exp2(sm[g] - mt);
                    lt += sl[g] * f;
                    for (int i = 0; i < DPL; ++i) o[i] += sacc[g * D + lane + 32 * i] * f;
                }
            }
            // partial layout: [B, HQ, L, NSPLIT, D + 2]
            device float* pp = part + ((((size_t)b * HQ + qh[r]) * L + qt[r]) * NSPLIT + split) * (D + 2);
            for (int i = 0; i < DPL; ++i) pp[lane + 32 * i] = o[i];
            if (lane == 0) { pp[D] = mt; pp[D + 1] = lt; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Stage 2: one simdgroup per (b, h, t): merge NSPLIT partials and the sink.
_ATTN_MERGE_SOURCE = """
    constexpr int DPL = D / 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x;   // ((b * HQ + h) * L + t)
    const int L = part_shape[2];
    const int t = row % L;
    const int h = (row / L) % HQ;
    const int b = row / (L * HQ);
    const device float* pp = part + (size_t)row * NSPLIT * (D + 2);
    const float sk = float(sink[b * sink_strides[0] + t * sink_strides[1] + h * sink_strides[2]]) * LOG2E;
    float mt = sk;
    for (int g = 0; g < NSPLIT; ++g) mt = max(mt, pp[g * (D + 2) + D]);
    float lt = fast::exp2(sk - mt);
    float o[DPL];
    for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
    for (int g = 0; g < NSPLIT; ++g) {
        const float mg = pp[g * (D + 2) + D];
        if (mg == -INFINITY) continue;
        const float f = fast::exp2(mg - mt);
        lt += pp[g * (D + 2) + D + 1] * f;
        for (int i = 0; i < DPL; ++i) o[i] += pp[g * (D + 2) + lane + 32 * i] * f;
    }
    const float inv = 1.0f / lt;
    device T* op = out + (size_t)row * D;
    for (int i = 0; i < DPL; ++i) op[lane + 32 * i] = T(o[i] * inv);
"""
_attn_kernels: dict = {}
_ATTN_QR = 8
_ATTN_NSG = 4
_ATTN_KEYS_PER_SPLIT = 128
_ATTN_MAX_SPLITS = 64


def small_attention(q, k, v, rel, sink, *, scale, window, left_padding=None, row_ends=None):
    """``kernels.banded_sink_attention`` for short query blocks.

    q: [B, Hq, L, D]; k, v: [B, Hkv, S, D]; rel: [B, L, Hq, R]; sink: [B, L, Hq].
    ``row_ends[b]`` is the number of key columns row b has written (default S),
    so rows may end at different columns. Returns [B, Hq, L, D].
    """
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    group = hq // hkv
    r = rel.shape[-1]
    ends = list(row_ends) if row_ends is not None else [s] * b
    # Keys any query can see: from the shortest row's window start for sliding layers.
    base = max(0, min(ends) - length - int(window) + 1) if window else 0
    span = s - base
    nsplit = max(1, min(_ATTN_MAX_SPLITS, -(-span // _ATTN_KEYS_PER_SPLIT)))
    chunk = -(-span // nsplit)
    key = (hq, hkv, d, r, int(window), q.dtype, b, nsplit)
    kerns = _attn_kernels.get(key)
    if kerns is None:
        kerns = _attn_kernels[key] = (
            mx.fast.metal_kernel(
                name=f"inkling2_small_attention_{len(_attn_kernels)}",
                input_names=["q", "k", "v", "rel", "params", "params_f"],
                output_names=["part"],
                source=_ATTN_SOURCE,
                ensure_row_contiguous=False,
            ),
            mx.fast.metal_kernel(
                name=f"inkling2_small_attention_merge_{len(_attn_kernels)}",
                input_names=["part", "sink"],
                output_names=["out"],
                source=_ATTN_MERGE_SOURCE,
                ensure_row_contiguous=False,
            ),
        )
    head = [length, base, chunk]
    if left_padding is None:
        params = mx.array(head + [0] * b + ends, dtype=mx.int32)
    else:
        params = mx.concatenate(
            [mx.array(head, dtype=mx.int32), left_padding.astype(mx.int32), mx.array(ends, dtype=mx.int32)]
        )
    rows = length * group
    blocks = -(-rows // _ATTN_QR)
    tmpl = [
        ("T", q.dtype), ("D", d), ("HQ", hq), ("GROUP", group), ("R", r), ("WINDOW", int(window)),
        ("QR", _ATTN_QR), ("NSG", _ATTN_NSG), ("B", b), ("NSPLIT", nsplit),
    ]
    part = kerns[0](
        inputs=[q, k, v, rel, params, mx.array([scale], dtype=mx.float32)],
        template=tmpl,
        grid=(blocks * nsplit * 32 * _ATTN_NSG, hkv, b),
        threadgroup=(32 * _ATTN_NSG, 1, 1),
        output_shapes=[(b, hq, length, nsplit, d + 2)],
        output_dtypes=[mx.float32],
    )[0]
    return kerns[1](
        inputs=[part, sink],
        template=[("T", q.dtype), ("D", d), ("HQ", hq), ("NSPLIT", nsplit)],
        grid=(32 * b * hq * length, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]


# Long inputs (prefill): keep the FP32 matmuls of the reference and fuse the
# elementwise work around them. Prep: unit RMS of x and message rows, BF16
# rounding, laid out [N, 2, rows, H] like the reference stack. Post: gates ->
# (1 - tanh(forget)) * x + tanh(gain) * message.
_MSG_PREP_SOURCE = """
    const uint row = threadgroup_position_in_grid.x;   // over N * 2 * ROWS
    const uint tid = thread_position_in_threadgroup.x;
    const int n = row / (2 * ROWS);
    const int part = (row / ROWS) % 2;
    const int r = row % ROWS;
    const device float* src = (part == 0 ? x : msg) + ((size_t)n * ROWS + r) * H;
    threadgroup float red[NT / 32];
    float ss = 0.0f;
    for (int i = tid; i < H; i += NT) { const float a = src[i]; ss += a * a; }
    ss = simd_sum(ss);
    if (tid % 32 == 0) red[tid / 32] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < NT / 32; ++i) tot += red[i];
    const float f = metal::precise::rsqrt(tot / float(H) + 1e-6f);
    device float* dst = z + (size_t)row * H;
    for (int i = tid; i < H; i += NT) dst[i] = float(bfloat16_t(src[i] * f));
"""
_MSG_POST_SOURCE = """
    const uint h = thread_position_in_grid.x;
    const uint nr = thread_position_in_grid.y;
    if ((int)h >= H) return;
    const size_t g = (size_t)nr * 2 * H;
    const float gain = metal::precise::tanh(gates[g + h]);
    const float forget = metal::precise::tanh(gates[g + H + h]);
    const size_t i = (size_t)nr * H + h;
    out[i] = (1.0f - forget) * x[i] + gain * msg[i];
"""


def mix_messages_long(x, message, down, gain_forget):
    """``model._mix_messages`` for prefill-sized inputs: same FP32 matmuls,
    fused normalization and gating."""
    n, b, length, hidden = x.shape
    rows = b * length
    key = ("long", hidden)
    kerns = _msg_kernels.get(key)
    if kerns is None:
        kerns = _msg_kernels[key] = (
            mx.fast.metal_kernel(name="inkling2_msg_prep", input_names=["x", "msg"],
                                 output_names=["z"], source=_MSG_PREP_SOURCE),
            mx.fast.metal_kernel(name="inkling2_msg_post", input_names=["gates", "x", "msg"],
                                 output_names=["out"], source=_MSG_POST_SOURCE),
        )
    x = x.astype(mx.float32)
    message = message.astype(mx.float32)
    nt = 256
    z = kerns[0](
        inputs=[x, message],
        template=[("H", hidden), ("ROWS", rows), ("NT", nt)],
        grid=(nt * n * 2 * rows, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(2 * n, rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    low = (z @ down.reshape(2 * n, hidden, -1)).reshape(n, 2, rows, -1).sum(axis=1)
    low = low.astype(mx.bfloat16).astype(mx.float32)
    gates = low @ gain_forget
    return kerns[1](
        inputs=[gates, x, message],
        template=[("H", hidden)],
        grid=(hidden, n * rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )[0]


# ---------------------------------------------------------------------------
# Short convs for long inputs (prefill): one thread per (channel, token)
# ---------------------------------------------------------------------------

_CONV_LONG_SOURCE = """
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const int C = x_shape[2];
    const int L = x_shape[1];
    if ((int)c >= C || (int)t >= L) return;
    constexpr int KM = K - 1;
    const int R = KM + L;
    const float xt = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
    xp[((size_t)b * R + KM + t) * C + c] = xt;
    if ((int)t < KM) {
        xp[((size_t)b * R + t) * C + c] = state[b * state_strides[0] + t * state_strides[1] + c * state_strides[2]];
    }
    // window element j of token t: padded index t + j (j = 0 is the oldest tap).
    float win[K];
    for (int j = 0; j < K; ++j) {
        const int pi = (int)t + j;
        win[j] = pi < KM
            ? state[b * state_strides[0] + pi * state_strides[1] + c * state_strides[2]]
            : float(x[b * x_strides[0] + (pi - KM) * x_strides[1] + c * x_strides[2]]);
    }
    float acc;
    if (DYN) {
        const int head = c / (C / HEADS);
        acc = 0.0f;
        for (int lag = 0; lag < K; ++lag) {
            const float tap = float(w[lag * w_strides[0] + c * w_strides[1]])
                + float(dyn[b * dyn_strides[0] + t * dyn_strides[1] + (head * K + lag) * dyn_strides[2]]);
            const float term = win[KM - lag] * tap;
            acc = lag == 0 ? term : acc + term;
        }
        out[((size_t)b * L + t) * C + c] = OT((acc + xt) * SCALE);
    } else {
        acc = float(w[c * w_strides[0]]) * win[0];
        for (int j = 1; j < K; ++j) acc = acc + float(w[c * w_strides[0] + j * w_strides[1]]) * win[j];
        const float o = float(WT(acc));
        out[((size_t)b * L + t) * C + c] = OT((o + xt) * SCALE);
    }
"""
_conv_long_kernels: dict = {}


def _conv_long(x, w, state, scale, dyn=None, heads=1):
    b, length, c = x.shape
    k = w.shape[-1] if dyn is None else w.shape[0]
    is_dyn = dyn is not None
    key = (k, float(scale), is_dyn, heads)
    kern = _conv_long_kernels.get(key)
    if kern is None:
        kern = _conv_long_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_conv_long_{len(_conv_long_kernels)}",
            input_names=["x", "state", "w", "dyn"],
            output_names=["out", "xp"],
            source=_CONV_LONG_SOURCE.replace("SCALE", f"{float(scale)!r}f"),
            ensure_row_contiguous=False,
        )
    if state.dtype != mx.float32:
        state = state.astype(mx.float32)
    return kern(
        inputs=[x, state, w, dyn if is_dyn else x],
        template=[("K", k), ("DYN", is_dyn), ("HEADS", heads), ("WT", w.dtype), ("OT", x.dtype)],
        grid=(c, length, b),
        threadgroup=(min(256, c), 1, 1),
        output_shapes=[(b, length, c), (b, k - 1 + length, c)],
        output_dtypes=[x.dtype, mx.float32],
    )


def static_conv_long(x, w, state, scale):
    return _conv_long(x, w, state, scale)


def dynamic_conv_long(x, dyn, static_w, state, heads, scale):
    return _conv_long(x, static_w, state, scale, dyn=dyn, heads=heads)


# ---------------------------------------------------------------------------
# Depth mixing with its neighbours (decode/verify rows): residual add, unit
# RMS of the partial sum, softmax over depth sources, and the RMSNorm + BF16
# cast of each mixed input, in one kernel per token row.
# ---------------------------------------------------------------------------

_RM_TG = 256
_res_mix_kernels: dict = {}


def _res_mix_kernel(blocks: int, queries: int, add: bool, norm: bool, hidden: int, eps: float):
    key = (blocks, queries, add, norm, hidden, eps)
    kern = _res_mix_kernels.get(key)
    if kern is not None:
        return kern
    per = (hidden + _RM_TG - 1) // _RM_TG
    srcs = [f"b{i}" for i in range(blocks)]
    ns = blocks + 1
    loads = "\n".join(f"            sv[{i}] = {n}[base + h];" for i, n in enumerate(srcs))
    source = f"""
    constexpr uint NS = {ns};
    constexpr uint PER = {per};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    threadgroup float red[NS * Q > Q ? NS * Q : Q][TG / 32];
    // 1. the partial sum and its unit RMS
    float pv[PER];
    float ss = 0.0f;
    for (uint j = 0; j < PER; ++j) {{
        uint h = tid + j * TG;
        float r = 0.0f;
        if (h < H) {{
            r = float(x[base + h]);
            if (ADD) r = (BOUNDARY ? 0.0f : r) + float(a[base + h]);
            keep_r[base + h] = r;
        }}
        pv[j] = r;
        ss += r * r;
    }}
    {{
        float v = simd_sum(ss);
        if (lane == 0) red[0][sg] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[0][j];
        float inv = metal::precise::rsqrt(t / float(H) + 1e-6f);
        for (uint j = 0; j < PER; ++j) {{
            pv[j] = pv[j] * inv;
            uint h = tid + j * TG;
            if (!ADD && h < H) keep_r[base + h] = pv[j];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    // 2. logits of every source against every query
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    for (uint j = 0; j < PER; ++j) {{
        uint h = tid + j * TG;
        if (h >= H) break;
        float sv[NS];
{loads}
        sv[NS - 1] = pv[j];
        for (uint qi = 0; qi < Q; ++qi) {{
            float qv = query[qi * H + h];
            for (uint n = 0; n < NS; ++n) acc[n * Q + qi] += sv[n] * qv;
        }}
    }}
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
    // 3. the mixtures, then (NORM) RMSNorm with the query's weight and BF16
    float mv[Q][PER];
    float sq[Q];
    for (uint qi = 0; qi < Q; ++qi) sq[qi] = 0.0f;
    for (uint j = 0; j < PER; ++j) {{
        uint h = tid + j * TG;
        float sv[NS];
        if (h < H) {{
{loads}
        }} else {{
            for (uint n = 0; n + 1 < NS; ++n) sv[n] = 0.0f;
        }}
        sv[NS - 1] = pv[j];
        for (uint qi = 0; qi < Q; ++qi) {{
            float o = 0.0f;
            for (uint n = 0; n < NS; ++n) o = n == 0 ? w[qi] * sv[0] : o + w[n * Q + qi] * sv[n];
            mv[qi][j] = o;
            sq[qi] += o * o;
        }}
    }}
    if (NORM) {{
        for (uint qi = 0; qi < Q; ++qi) {{
            float v = simd_sum(sq[qi]);
            if (lane == 0) red[qi][sg] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint qi = 0; qi < Q; ++qi) {{
            float t = 0.0f;
            for (uint j = 0; j < TG / 32; ++j) t += red[qi][j];
            float inv = metal::precise::rsqrt(t / float(H) + EPS);
            for (uint j = 0; j < PER; ++j) {{
                uint h = tid + j * TG;
                if (h < H) out[((size_t)row * Q + qi) * H + h] =
                    OT(float(nw[qi * H + h]) * (mv[qi][j] * inv));
            }}
        }}
    }} else {{
        for (uint qi = 0; qi < Q; ++qi)
            for (uint j = 0; j < PER; ++j) {{
                uint h = tid + j * TG;
                if (h < H) out[((size_t)row * Q + qi) * H + h] = OT(mv[qi][j]);
            }}
    }}
"""
    source = source.replace("EPS", f"{float(eps)!r}f")
    kern = _res_mix_kernels[key] = mx.fast.metal_kernel(
        name=f"inkling2_res_mix_{len(_res_mix_kernels)}",
        input_names=[*srcs, "x", "a", "query", "nw", "scale"],
        output_names=["keep_r", "out"],
        source=source,
    )
    return kern


def res_mix(blocks, x, query, scale, *, a=None, boundary=False, norm_w=None, eps=1e-6):
    """Depth mixing of decode/verify rows with its neighbours fused.

    ``x`` [B, L, H]: the partial sum (pre-attention ``h``), or with ``a`` the
    residual input ``h + a`` (``a`` alone at a block boundary). Returns
    ``(keep, mixed)``: ``keep`` is the FP32 unit-RMS partial (without ``a``)
    or the FP32 residual (with ``a``); ``mixed`` is [B, L, Q, H], BF16 after
    RMSNorm with ``norm_w`` [Q, H] rows, or FP32 without ``norm_w``.
    """
    b, length, hidden = x.shape
    nq = query.shape[0]
    add = a is not None
    norm = norm_w is not None
    kern = _res_mix_kernel(len(blocks), nq, add, norm, hidden, float(eps))
    outs = kern(
        inputs=[*[s.astype(mx.float32) for s in blocks], x, a if add else x,
                query.astype(mx.float32), norm_w if norm else _ONE, mx.array([scale], dtype=mx.float32)],
        template=[("H", hidden), ("Q", nq), ("TG", _RM_TG), ("ADD", add), ("BOUNDARY", bool(boundary)),
                  ("NORM", norm), ("OT", mx.bfloat16 if norm else mx.float32)],
        grid=(b * length * _RM_TG, 1, 1),
        threadgroup=(_RM_TG, 1, 1),
        output_shapes=[(b, length, hidden), (b, length, nq, hidden)],
        output_dtypes=[mx.float32, mx.bfloat16 if norm else mx.float32],
    )
    return outs[0], outs[1]


_MULTI_NORM_SOURCE = """
    constexpr uint PER = (H + TG - 1) / TG;
    uint row = threadgroup_position_in_grid.x;   // n * rows + r
    uint n = row / ROWS;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    threadgroup float red[TG / 32];
    float v[PER];
    float ss = 0.0f;
    for (uint j = 0; j < PER; ++j) {
        uint h = tid + j * TG;
        v[j] = h < H ? float(x[base + h]) : 0.0f;
        ss += v[j] * v[j];
    }
    float s = simd_sum(ss);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint j = 0; j < TG / 32; ++j) t += red[j];
    float inv = metal::precise::rsqrt(t / float(H) + EPS);
    for (uint j = 0; j < PER; ++j) {
        uint h = tid + j * TG;
        if (h < H) out[base + h] = bfloat16_t(float(w[n * H + h]) * (v[j] * inv));
    }
"""
_multi_norm_kernels: dict = {}


def multi_norm(x, w, eps):
    """RMSNorm of ``x[n]`` [N, B, L, H] with weight ``w[n]`` [N, H], cast to BF16."""
    n, b, length, hidden = x.shape
    key = float(eps)
    kern = _multi_norm_kernels.get(key)
    if kern is None:
        kern = _multi_norm_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_multi_norm_{len(_multi_norm_kernels)}",
            input_names=["x", "w"],
            output_names=["out"],
            source=_MULTI_NORM_SOURCE.replace("EPS", f"{float(eps)!r}f"),
        )
    return kern(
        inputs=[x, w],
        template=[("H", hidden), ("TG", _RM_TG), ("ROWS", b * length)],
        grid=(n * b * length * _RM_TG, 1, 1),
        threadgroup=(_RM_TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------------------
# NVFP4 expert matvecs for decode/verify rows (group 16, E2M1 codes, E4M3
# group scales; the per-expert tensor scale is applied after the dot product).
# MLX's fast quantized matvec needs K % 512 == 0; Inkling's K = 2304 / 1152
# take its slower path. Here a simdgroup streams 512 values per step (32 lanes
# x one 16-value group) over a gate/up row pair (2 x 2304 = 9 steps) or over
# one down row of all top-k experts (8 x 1152 = 18 steps): no idle lanes.
# ---------------------------------------------------------------------------

_NVFP4_HEADER = """
// E2M1 code -> float, MLX's bit placement (fp4.h): |c| << 9 as half, x 2^14.
inline float e2m1(uint c) {
    const float v = float(as_type<half>(ushort((c & 7u) << 9))) * 16384.0f;
    return (c & 8u) ? -v : v;
}
// E4M3 byte -> float: exponent/mantissa into a half, x 2^8 (exact, incl. subnormals).
inline float e4m3(uint b) {
    const float v = float(as_type<half>(ushort((b & 0x7Fu) << 7))) * 256.0f;
    return (b & 0x80u) ? -v : v;
}
inline float bf(ushort u) { return as_type<float>(uint(u) << 16); }
// x . 16 E2M1 codes, like MLX's qdot (sum, then the caller scales).
inline float dot16(uint2 w, thread const float* xv) {
    float acc = 0.0f;
    for (int j = 0; j < 8; ++j) acc += xv[j] * e2m1(w.x >> (4 * j));
    for (int j = 0; j < 8; ++j) acc += xv[8 + j] * e2m1(w.y >> (4 * j));
    return acc;
}
inline void load16(const device ushort* p, thread float* xv) {
    const ushort4 a = ((const device ushort4*)p)[0], b = ((const device ushort4*)p)[1];
    const ushort4 c = ((const device ushort4*)p)[2], d = ((const device ushort4*)p)[3];
    xv[0] = bf(a.x); xv[1] = bf(a.y); xv[2] = bf(a.z); xv[3] = bf(a.w);
    xv[4] = bf(b.x); xv[5] = bf(b.y); xv[6] = bf(b.z); xv[7] = bf(b.w);
    xv[8] = bf(c.x); xv[9] = bf(c.y); xv[10] = bf(c.z); xv[11] = bf(c.w);
    xv[12] = bf(d.x); xv[13] = bf(d.y); xv[14] = bf(d.z); xv[15] = bf(d.w);
}
"""

_NV_W13_SOURCE = """
    // grid: (32 * NSG, tiles, N * K); each simdgroup owns RP activation columns.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nk = threadgroup_position_in_grid.z;
    const uint n = nk / K;
    const uint e = inds[nk];
    const device T* xr = x + (size_t)n * H;
    const float sc = float(T(gscale[e] * INV_DENOM));
    for (uint rp = 0; rp < RP; ++rp) {
        const uint i = (threadgroup_position_in_grid.y * NSG + sg) * RP + rp;
        if (i >= I) break;
        // rows 2i (gate) and 2i + 1 (up) are adjacent: 2H contiguous values.
        const device uint32_t* wp = w + ((size_t)e * 2 * I + 2 * i) * (H / 8);
        const device uint8_t* sp = s + ((size_t)e * 2 * I + 2 * i) * (H / 16);
        float part0 = 0.0f, part1 = 0.0f;
        for (uint step = 0; step < (2 * H) / 512; ++step) {
            const uint p = step * 512 + lane * 16;
            const bool up = p >= H;
            const uint off = up ? p - H : p;
            float xv[16];
            load16((const device ushort*)(xr + off), xv);
            const float d = e4m3(sp[p / 16]) * dot16(((const device uint2*)wp)[p / 16], xv);
            if (up) part1 += d; else part0 += d;
        }
        float g = simd_sum(part0);
        float u = simd_sum(part1);
        if (lane == 0) {
            g = float(T(float(T(g)) * sc));
            u = float(T(float(T(u)) * sc));
            const float o = metal::precise::tanh(g / GCAP) * GCAP * (1.0f / (1.0f + metal::precise::exp(-g)))
                * (metal::precise::tanh(u / UCAP) * UCAP);
            act[(size_t)nk * I + i] = T(o);
        }
    }
"""

_NV_W2_SOURCE = """
    // grid: (32 * NSG, tiles, N); each simdgroup owns RO output rows and sums
    // the K experts' down rows as one dot product of K * I values.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n = threadgroup_position_in_grid.z;
    for (uint ro = 0; ro < RO; ++ro) {
        const uint r = (threadgroup_position_in_grid.y * NSG + sg) * RO + ro;
        if (r >= H) break;
        float part = 0.0f;
        for (uint step = 0; step < (K * I) / 512; ++step) {
            const uint p = step * 512 + lane * 16;
            const uint k = p / I;
            const uint off = p - k * I;
            const uint e = inds[n * K + k];
            const float wk = weights[n * K + k] * (HAS_SCALE ? gscale[e] * INV_DENOM : 1.0f);
            const device uint32_t* wp = w + ((size_t)e * H + r) * (I / 8);
            const device uint8_t* sp = s + ((size_t)e * H + r) * (I / 16);
            const device T* ar = act + ((size_t)n * K + k) * I;
            float xv[16];
            load16((const device ushort*)(ar + off), xv);
            part += wk * (e4m3(sp[off / 16]) * dot16(((const device uint2*)wp)[off / 16], xv));
        }
        part = simd_sum(part);
        if (lane == 0) out[(size_t)n * H + r] = OT(part);
    }
"""
_nv_kernels: dict = {}


def _nv_kernel(name, source, inputs, outputs, consts):
    key = (name, consts)
    kern = _nv_kernels.get(key)
    if kern is None:
        src = source
        for k, v in consts:
            src = src.replace(k, f"{float(v)!r}f")
        kern = _nv_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_{name}_{len(_nv_kernels)}",
            input_names=inputs,
            output_names=outputs,
            source=src,
            header=_NVFP4_HEADER,
        )
    return kern


NV_RP = int(os.environ.get("OMLX_INKLING2_NV_RP", "2"))
NV_RO = int(os.environ.get("OMLX_INKLING2_NV_RO", "2"))
NV_NSG = int(os.environ.get("OMLX_INKLING2_NV_NSG", "2"))


def nvfp4_swiglu(x, proj, inds, inv_denom, gcap, ucap):
    """Gate/up projection of the top-k experts + scale + softcapped SwiGLU.

    x: [N, H] BF16; proj: ExpertProj (nvfp4, interleaved gate/up rows);
    inds: [N, K]. Returns act [N, K, 1, I] like ``swiglu(w13(x))``.
    """
    n, k = inds.shape
    hidden = x.shape[-1]
    i = proj.weight.shape[1] // 2
    kern = _nv_kernel(
        "nv_w13", _NV_W13_SOURCE, ["x", "w", "s", "inds", "gscale"], ["act"],
        (("INV_DENOM", inv_denom), ("GCAP", gcap), ("UCAP", ucap)),
    )
    tiles = (i + NV_NSG * NV_RP - 1) // (NV_NSG * NV_RP)
    return kern(
        inputs=[x.reshape(n, hidden), proj.weight, proj.scales, inds.astype(mx.uint32), proj.global_scale],
        template=[("T", x.dtype), ("H", hidden), ("I", i), ("K", k), ("RP", NV_RP), ("NSG", NV_NSG)],
        grid=(32 * NV_NSG, tiles, n * k),
        threadgroup=(32 * NV_NSG, 1, 1),
        output_shapes=[(n, k, 1, i)],
        output_dtypes=[x.dtype],
    )[0]


def nvfp4_down_sum(act, proj, inds, weights, inv_denom, out_dtype):
    """Down projection of the top-k experts and their weighted sum: [N, H]."""
    n, k = inds.shape
    i = act.shape[-1]
    hidden = proj.weight.shape[1]
    kern = _nv_kernel(
        "nv_w2", _NV_W2_SOURCE, ["act", "w", "s", "inds", "weights", "gscale"], ["out"],
        (("INV_DENOM", inv_denom),),
    )
    tiles = (hidden + NV_NSG * NV_RO - 1) // (NV_NSG * NV_RO)
    return kern(
        inputs=[act, proj.weight, proj.scales, inds.astype(mx.uint32), weights.astype(mx.float32), proj.global_scale],
        template=[("T", act.dtype), ("OT", out_dtype), ("H", hidden), ("I", i), ("K", k), ("RO", NV_RO),
                  ("NSG", NV_NSG), ("HAS_SCALE", True)],
        grid=(32 * NV_NSG, tiles, n),
        threadgroup=(32 * NV_NSG, 1, 1),
        output_shapes=[(n, hidden)],
        output_dtypes=[out_dtype],
    )[0]


# ---------------------------------------------------------------------------
# Long-context single-row decode attention (full-attention layers, B = 1,
# L = 1). The full layers read the whole KV per token (3.3 GB at 100k), and
# MLX's vector SDPA runs that at ~170 GB/s. Here each simdgroup walks 32-key
# tiles: lanes own keys for the scores (q of the GROUP heads sharing the kv
# head sits in threadgroup memory), then own 4 dims each for the V sum
# (coalesced rows), with one online-softmax update per tile. Partials use the
# small_attention layout, so its merge kernel adds the sink and normalizes.
# ---------------------------------------------------------------------------

_LONG_ATTN_SOURCE = """
    // grid: (32 * NSG, NSPLIT / NSG, HKV). Each simdgroup owns one key split and
    // writes its own partial for the ROWS = GROUP * L query rows of its kv head.
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int ROWS = GROUP * L;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint split = threadgroup_position_in_grid.y * NSG + sg;
    const uint hk = threadgroup_position_in_grid.z;
    const int S = k_shape[2];
    const int chunk = dims[0];
    const int lp = dims[1];   // left padding of the row: keys before it are not real
    const float qscale = scale[0] * LOG2E;
    threadgroup float4 qs[ROWS][D / 4];
    for (uint i = tid; i < ROWS * D / 4; i += 32 * NSG) {
        const uint row = i / (D / 4), c = i % (D / 4);
        const uint g = row / L, t = row % L;
        const device T* qp = q + (hk * GROUP + g) * q_strides[1] + t * q_strides[2] + (c * 4) * q_strides[3];
        qs[row][c] = float4(float(qp[0]), float(qp[q_strides[3]]), float(qp[2 * q_strides[3]]), float(qp[3 * q_strides[3]])) * qscale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split >= NSPLIT) return;
    const int s0 = int(split) * chunk;
    const int s1 = min(S, s0 + chunk);
    const device T* kb = k + hk * k_strides[1];
    const device T* vb = v + hk * v_strides[1];
    float m[ROWS], l[ROWS], acc[ROWS][4];
    for (int r = 0; r < ROWS; ++r) { m[r] = -INFINITY; l[r] = 0.0f; for (int i = 0; i < 4; ++i) acc[r][i] = 0.0f; }
    for (int tile = s0; tile < s1; tile += 32) {
        const int j = tile + int(lane);
        float sc[ROWS];
        if (j < s1 && j >= lp) {
            float kd[ROWS];
            for (int r = 0; r < ROWS; ++r) kd[r] = 0.0f;
            const device T* kp = kb + (size_t)j * k_strides[2];
            for (int c = 0; c < D / 4; ++c) {
                const float4 kv = float4(float(kp[4 * c]), float(kp[4 * c + 1]), float(kp[4 * c + 2]), float(kp[4 * c + 3]));
                for (int r = 0; r < ROWS; ++r) kd[r] += dot(qs[r][c], kv);
            }
            for (int r = 0; r < ROWS; ++r) {
                const int g = r / L, t = r % L;
                const int d = S - L + t - j;   // causal distance of query t
                float s = d < 0 ? -INFINITY : kd[r];
                if (d >= 0 && d < R) s += float(rel[(t * HQ + hk * GROUP + g) * R + d]) * LOG2E;
                sc[r] = s;
            }
        } else {
            for (int r = 0; r < ROWS; ++r) sc[r] = -INFINITY;
        }
        float p[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            const float mn = max(m[r], simd_max(sc[r]));
            if (mn == -INFINITY) { p[r] = 0.0f; continue; }
            const float f = m[r] == -INFINITY ? 0.0f : fast::exp2(m[r] - mn);
            p[r] = sc[r] == -INFINITY ? 0.0f : fast::exp2(sc[r] - mn);
            l[r] = l[r] * f + simd_sum(p[r]);
            for (int i = 0; i < 4; ++i) acc[r][i] *= f;
            m[r] = mn;
        }
        const int nk = min(32, s1 - tile);
        for (int kk = 0; kk < nk; ++kk) {
            const device T* vp = vb + (size_t)(tile + kk) * v_strides[2] + lane * 4;
            const float4 vv = float4(float(vp[0]), float(vp[1]), float(vp[2]), float(vp[3]));
            for (int r = 0; r < ROWS; ++r) {
                const float pk = simd_shuffle(p[r], ushort(kk));
                for (int i = 0; i < 4; ++i) acc[r][i] += pk * vv[i];
            }
        }
    }
    for (int r = 0; r < ROWS; ++r) {
        const int g = r / L, t = r % L;
        // partial layout: [1, HQ, L, NSPLIT, D + 2]
        device float* pp = part + (((size_t)(hk * GROUP + g) * L + t) * NSPLIT + split) * (D + 2);
        for (int i = 0; i < 4; ++i) pp[lane * 4 + i] = acc[r][i];
        if (lane == 0) { pp[D] = m[r]; pp[D + 1] = l[r]; }
    }
"""
_long_attn_kernels: dict = {}
LONG_ATTN_MIN_KEYS = int(os.environ.get("OMLX_INKLING2_LONG_ATTN_MIN", "4096"))
LONG_ATTN_KEYS_PER_SPLIT = int(os.environ.get("OMLX_INKLING2_LONG_ATTN_SPLIT", "256"))
LONG_ATTN_NSG = int(os.environ.get("OMLX_INKLING2_LONG_ATTN_NSG", "8"))


LONG_ATTN_MAX_QUERIES = 4


def long_decode_attention(q, k, v, rel, sink, *, scale, left_padding=None):
    """Full-attention decode/verify for one row: q [1, Hq, L, D], L <= 8.

    ``left_padding`` ([1] array, optional) skips the row's padded key columns
    on the GPU, so no host sync is needed."""
    _, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    group = hq // hkv
    r = rel.shape[-1]
    nsplit = max(1, -(-s // LONG_ATTN_KEYS_PER_SPLIT))
    nsplit = -(-nsplit // LONG_ATTN_NSG) * LONG_ATTN_NSG
    chunk = -(-s // nsplit)
    key = (hq, hkv, d, r, q.dtype, LONG_ATTN_NSG, length)
    kern = _long_attn_kernels.get(key)
    if kern is None:
        kern = _long_attn_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_long_decode_attention_{len(_long_attn_kernels)}",
            input_names=["q", "k", "v", "rel", "dims", "scale"],
            output_names=["part"],
            source=_LONG_ATTN_SOURCE,
            ensure_row_contiguous=False,
        )
    merge = _attn_kernels.get(("merge", d, hq, nsplit, q.dtype))
    if merge is None:
        merge = _attn_kernels[("merge", d, hq, nsplit, q.dtype)] = mx.fast.metal_kernel(
            name=f"inkling2_long_attention_merge_{len(_attn_kernels)}",
            input_names=["part", "sink"],
            output_names=["out"],
            source=_ATTN_MERGE_SOURCE,
            ensure_row_contiguous=False,
        )
    lp = (left_padding.reshape(1).astype(mx.int32) if left_padding is not None
          else mx.zeros((1,), dtype=mx.int32))
    part = kern(
        inputs=[q, k, v, mx.contiguous(rel.reshape(length, hq, r)),
                mx.concatenate([mx.array([chunk], dtype=mx.int32), lp]),
                mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype), ("D", d), ("GROUP", group), ("R", r), ("NSG", LONG_ATTN_NSG),
                  ("NSPLIT", nsplit), ("L", length), ("HQ", hq)],
        grid=(32 * LONG_ATTN_NSG, nsplit // LONG_ATTN_NSG, hkv),
        threadgroup=(32 * LONG_ATTN_NSG, 1, 1),
        output_shapes=[(1, hq, length, nsplit, d + 2)],
        output_dtypes=[mx.float32],
    )[0]
    return merge(
        inputs=[part, sink],
        template=[("T", q.dtype), ("D", d), ("HQ", hq), ("NSPLIT", nsplit)],
        grid=(32 * hq * length, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]
