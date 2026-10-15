r"""
Compile-time auto-tuning block: 

import torch
from math import inf, nan
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6faf437cf6fb66b33fa3388af79428f290cf99401e16bfb46f42c4fdf894f4c7/inductor_cache/du/cdu2ok4aaqazxes7mhsvlyisqcpbhy64n6dr6puhep3nphnnlg4y.py
# Topologically Sorted Source Nodes: [float_1, add, rms_norm_default], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   add => add
#   float_1 => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg1_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf0]
#   %arg0_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %add : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf0,%convert_element_type_default_1
triton_red_fused__to_copy_add_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 201334784}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.13422592, 'kernel_flop': 0, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 4096.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6faf437cf6fb66b33fa3388af79428f290cf99401e16bfb46f42c4fdf894f4c7/inductor_cache/fw/cfwywkmy6kzszglrvxcapofqf3hhnfdzdtg5ozshfuc2ebwyvalx.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg2_1, 32, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_1 = async_compile.triton('triton_poi_fused_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 134217728}, 'kernel_num_gb': 0.067108864, 'kernel_flop': 0, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    raw_stream0 = get_raw_stream(0)
raw_stream0 = get_raw_stream(0)
arg1_1 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg0_1 = generate_example_value((4096,), (1,), 'cuda:0', torch.bfloat16, 0, (4096,))
buf1 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_rms_norm_0.run(arg1_1, arg0_1, buf1, 8192, 4096, stream=raw_stream0)
del arg1_1, arg0_1, buf1

raw_stream0 = get_raw_stream(0)
buf4 = generate_example_value((8192, 32, 128), (4096, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_zeros_1.run(buf4, 33554432, stream=raw_stream0)
del buf4

"""
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6faf437cf6fb66b33fa3388af79428f290cf99401e16bfb46f42c4fdf894f4c7/inductor_cache/du/cdu2ok4aaqazxes7mhsvlyisqcpbhy64n6dr6puhep3nphnnlg4y.py
# Topologically Sorted Source Nodes: [float_1, add, rms_norm_default], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   add => add
#   float_1 => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg1_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf0]
#   %arg0_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %add : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf0,%convert_element_type_default_1
triton_red_fused__to_copy_add_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_add_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 201334784}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.13422592, 'kernel_flop': 0, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_rms_norm_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 4096.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6faf437cf6fb66b33fa3388af79428f290cf99401e16bfb46f42c4fdf894f4c7/inductor_cache/fw/cfwywkmy6kzszglrvxcapofqf3hhnfdzdtg5ozshfuc2ebwyvalx.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg2_1, 32, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_1 = async_compile.triton('triton_poi_fused_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 134217728}, 'kernel_num_gb': 0.067108864, 'kernel_flop': 0, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        s59 = arg2_1
        s18 = arg5_1
        assert_size_stride(arg1_1, (s18, 4096), (4096, 1), 'input')
        assert_size_stride(arg0_1, (4096, ), (1, ), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf1 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, add, rms_norm_default], Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_rms_norm_0.run(arg1_1, arg0_1, buf1, s18, 4096, stream=raw_stream0)
            del arg0_1
            del arg1_1
            assert_size_stride(arg3_1, (12288, 4096), (4096, 1), 'input')
            arg3_1 = copy_if_misaligned(arg3_1)
            buf2 = empty_strided_cuda((s18, 12288), (12288, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, (4096, 12288), (1, 4096), 0), out=buf2)
            del arg3_1
            assert_size_stride(arg4_1, (64, 4096), (4096, 1), 'input')
            arg4_1 = copy_if_misaligned(arg4_1)
            buf3 = empty_strided_cuda((s18, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf1, reinterpret_tensor(arg4_1, (4096, 64), (1, 4096), 0), out=buf3)
            del arg4_1
            buf4 = reinterpret_tensor(buf1, (s18, 32, 128), (4096, 128, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
            triton_poi_fused_zeros_1_xnumel = 4096*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_zeros_1.run(buf4, triton_poi_fused_zeros_1_xnumel, stream=raw_stream0)
        return (buf2, buf3, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = 8192
    arg3_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((64, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = 8192
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
