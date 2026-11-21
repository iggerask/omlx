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
    const device float* d0 = down + (size_t)n * 2 * H * RANK;
    const device float* d1 = d0 + (size_t)H * RANK;
    float acc0 = 0.0f, acc1 = 0.0f;
    const int i0 = g * CH;
    for (int i = i0; i < i0 + CH; ++i) {
        const float zx = float(bfloat16_t(xr[i] * fx));
        const float zm = float(bfloat16_t(mr[i] * fm));
        acc0 += zx * d0[(size_t)i * RANK + tid];
        acc1 += zm * d1[(size_t)i * RANK + tid];
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
    const device float* gf = gain_forget + (size_t)n * RANK * 2 * H;
    float ga = 0.0f, fo = 0.0f;
    for (int k = 0; k < RANK; ++k) {
        const float l = lr[k];
        ga += l * gf[(size_t)k * 2 * H + h];
        fo += l * gf[(size_t)k * 2 * H + H + h];
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
    key = (hidden, rank, rows, n)
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
        template=[("H", hidden), ("RANK", rank), ("ROWS", rows), ("CH", ch), ("G", groups), ("NR", nr)],
        grid=(rank * groups, nr, 1),
        threadgroup=(rank, 1, 1),
        output_shapes=[(2, nr, groups, rank)],
        output_dtypes=[mx.float32],
    )[0]
    return kerns[1](
        inputs=[part, gain_forget, x, message],
        template=[("H", hidden), ("RANK", rank), ("ROWS", rows), ("G", groups), ("NR", nr)],
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
