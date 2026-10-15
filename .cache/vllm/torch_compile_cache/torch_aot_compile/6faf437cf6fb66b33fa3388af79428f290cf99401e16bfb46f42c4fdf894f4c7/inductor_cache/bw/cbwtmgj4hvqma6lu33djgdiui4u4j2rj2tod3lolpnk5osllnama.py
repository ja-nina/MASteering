

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 2}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'declared_constexpr_names': ['num_q_heads', 'num_kv_heads', 'head_dim', 'rotary_dim', 'half_rotary', 'eps', 'norm_beta', 'INPUT_DTYPE', 'HEAD_BLOCK', 'ROT_HALF_BLOCK', 'HAS_PASS', 'HAS_MROPE', 'MROPE_SECTION_H', 'MROPE_SECTION_W'], 'kernel_name': '_fused_qk_rmsnorm_rope_gate_kernel_0', 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'q_gate_ptr': '*bf16', 'k_ptr': '*bf16', 'q_out_ptr': '*bf16', 'k_out_ptr': '*bf16', 'gate_out_ptr': '*bf16', 'q_weight_ptr': '*bf16', 'k_weight_ptr': '*bf16', 'cos_sin_cache_ptr': '*bf16', 'positions_ptr': '*i64', 'q_gate_stride_t': 'i32', 'k_stride_t': 'i32', 'q_out_stride_t': 'i32', 'k_out_stride_t': 'i32', 'gate_out_stride_t': 'i32', 'cache_stride_p': 'i32', 'positions_stride_m': 'i32', 'positions_stride_t': 'constexpr', 'num_q_heads': 'constexpr', 'num_kv_heads': 'constexpr', 'head_dim': 'constexpr', 'rotary_dim': 'constexpr', 'half_rotary': 'constexpr', 'eps': 'constexpr', 'norm_beta': 'constexpr', 'INPUT_DTYPE': 'constexpr', 'HEAD_BLOCK': 'constexpr', 'ROT_HALF_BLOCK': 'constexpr', 'HAS_PASS': 'constexpr', 'HAS_MROPE': 'constexpr', 'MROPE_SECTION_H': 'constexpr', 'MROPE_SECTION_W': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'positions_stride_t': 1, 'num_q_heads': 16, 'num_kv_heads': 4, 'head_dim': 256, 'rotary_dim': 64, 'half_rotary': 32, 'eps': 1e-06, 'norm_beta': 1.0, 'INPUT_DTYPE': triton.language.bfloat16, 'HEAD_BLOCK': 256, 'ROT_HALF_BLOCK': 32, 'HAS_PASS': True, 'HAS_MROPE': True, 'MROPE_SECTION_H': 11, 'MROPE_SECTION_W': 10}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    positions_stride_m,
    positions_stride_t,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    norm_beta: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    # --- RMSNorm: variance over the full head_dim ---
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32) + norm_beta
    # Round-trip through INPUT_DTYPE so the RoPE input matches the bf16-storage
    # behavior of the unfused (qk_rmsnorm -> memory -> apply_rope) reference path.
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm-only, no rotation ---
    # The rotary head [0, rotary_dim) will be overwritten by the RoPE store below.
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # --- Partial RoPE on the first rotary_dim elements ---
    # Triton lacks easy sub-vector slicing of x_norm, so we recompute the
    # normalized rotary halves on a smaller block (next_pow2(half_rotary)).
    # The extra ~rotary_dim element reload hits L1, so the cost is negligible.
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    w_rot1 = (
        tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32) + norm_beta
    )
    w_rot2 = (
        tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        + norm_beta
    )
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    # Always use int64 for positions to avoid overflow in address computation.
    pos_t = tl.load(positions_ptr + token * positions_stride_t).to(tl.int64)
    if HAS_MROPE:
        pos_h = tl.load(
            positions_ptr + positions_stride_m + token * positions_stride_t
        ).to(tl.int64)
        pos_w = tl.load(
            positions_ptr + 2 * positions_stride_m + token * positions_stride_t
        ).to(tl.int64)
        is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
        is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
        pos = tl.where(is_h, pos_h, tl.where(is_w, pos_w, pos_t))
    else:
        pos = pos_t
    cache_offset = pos * cache_stride_p
    cos = tl.load(
        cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    o1 = x_rot1 * cos - x_rot2 * sin
    o2 = x_rot2 * cos + x_rot1 * sin
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, o2, mask=rot_mask)

    # --- Gate copy (q heads only, verbatim) ---
    if not is_k:
        gate_in_base = in_base + head_dim
        gate_out_base = gate_out_ptr + token * gate_out_stride_t + local_head * head_dim
        g = tl.load(gate_in_base + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out_base + head_offs, g, mask=head_mask)
