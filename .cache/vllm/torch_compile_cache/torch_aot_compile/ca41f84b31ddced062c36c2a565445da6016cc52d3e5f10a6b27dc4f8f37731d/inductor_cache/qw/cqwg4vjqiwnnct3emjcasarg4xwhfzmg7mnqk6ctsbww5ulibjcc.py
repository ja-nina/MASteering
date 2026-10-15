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


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/ca41f84b31ddced062c36c2a565445da6016cc52d3e5f10a6b27dc4f8f37731d/inductor_cache/pe/cpe22c4hr42zycb3qfr3lrfmo2m34etv4weqb6gpmznuf5zugo27.py
# Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, steer_buffer], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   add => add_6
#   float_1 => convert_element_type_2
#   fused_add_rms_norm_default => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   steer_buffer => convert_element_type_default_6
# Graph fragment:
#   %mm : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf2 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf2]
#   %arg3_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type_2 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %add_6 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_6), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   return %buf2,%convert_element_type_default_6,%convert_element_type_default_7
triton_red_fused__to_copy_add_fused_add_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 402661376}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.268443648, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp9 = tmp4.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp9, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = tl.full([1, 1], 4096.0, tl.float32)
        tmp16 = (tmp7 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 1.0, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp20 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/ca41f84b31ddced062c36c2a565445da6016cc52d3e5f10a6b27dc4f8f37731d/inductor_cache/on/confxmq2w4ehm3wdyhf5vr345edchenuqrz4ts4tjym53f75va4i.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul => mul_20
#   silu => add_19, convert_element_type_5, convert_element_type_6, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s18, 24576][24576, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 12288), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_5,), kwargs = {})
#   %exp : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_19 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_5, %add_19), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 12288, 9223372036854775807), kwargs = {})
#   %mul_20 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %slice_2), kwargs = {})
#   return %mul_20
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 805306368}, 'kernel_num_gb': 0.603979776, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 24576*x1), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (12288 + x0 + 24576*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/ca41f84b31ddced062c36c2a565445da6016cc52d3e5f10a6b27dc4f8f37731d/inductor_cache/3z/c3zvtwxtvrmngxsovslkz7oyhcc4iexdlnomm3pa4st3kde36rig.py
# Topologically Sorted Source Nodes: [float_2, add_1, fused_add_rms_norm_default_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   add_1 => add_32
#   float_2 => convert_element_type_9
#   fused_add_rms_norm_default_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %mm_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %buf4 : Tensor  = PlaceHolder[target=buf4]
#   %buf10 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf10]
#   %arg12_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %convert_element_type_9 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_32), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf10,%convert_element_type_default_2,%convert_element_type_default_3
triton_red_fused__to_copy_add_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 335552512}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.201334784, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp9 = tmp4.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp9, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = tl.full([1, 1], 4096.0, tl.float32)
        tmp16 = (tmp7 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 1.0, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp20 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# Original path: /BS/conformal-circuits/work/mamba/envs/pvf/lib/python3.12/site-packages/vllm/model_executor/layers/fused_qk_norm_rope.py:16
_fused_qk_rmsnorm_rope_gate_kernel_0 = async_compile.triton('_fused_qk_rmsnorm_rope_gate_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 2}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'declared_constexpr_names': ['num_q_heads', 'num_kv_heads', 'head_dim', 'rotary_dim', 'half_rotary', 'eps', 'norm_beta', 'INPUT_DTYPE', 'HEAD_BLOCK', 'ROT_HALF_BLOCK', 'HAS_PASS', 'HAS_MROPE', 'MROPE_SECTION_H', 'MROPE_SECTION_W'], 'kernel_name': '_fused_qk_rmsnorm_rope_gate_kernel_0', 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'q_gate_ptr': '*bf16', 'k_ptr': '*bf16', 'q_out_ptr': '*bf16', 'k_out_ptr': '*bf16', 'gate_out_ptr': '*bf16', 'q_weight_ptr': '*bf16', 'k_weight_ptr': '*bf16', 'cos_sin_cache_ptr': '*bf16', 'positions_ptr': '*i64', 'q_gate_stride_t': 'i32', 'k_stride_t': 'i32', 'q_out_stride_t': 'i32', 'k_out_stride_t': 'i32', 'gate_out_stride_t': 'i32', 'cache_stride_p': 'i32', 'positions_stride_m': 'i32', 'positions_stride_t': 'constexpr', 'num_q_heads': 'constexpr', 'num_kv_heads': 'constexpr', 'head_dim': 'constexpr', 'rotary_dim': 'constexpr', 'half_rotary': 'constexpr', 'eps': 'constexpr', 'norm_beta': 'constexpr', 'INPUT_DTYPE': 'constexpr', 'HEAD_BLOCK': 'constexpr', 'ROT_HALF_BLOCK': 'constexpr', 'HAS_PASS': 'constexpr', 'HAS_MROPE': 'constexpr', 'MROPE_SECTION_H': 'constexpr', 'MROPE_SECTION_W': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'positions_stride_t': 1, 'num_q_heads': 16, 'num_kv_heads': 4, 'head_dim': 256, 'rotary_dim': 64, 'half_rotary': 32, 'eps': 1e-06, 'norm_beta': 1.0, 'INPUT_DTYPE': triton.language.bfloat16, 'HEAD_BLOCK': 256, 'ROT_HALF_BLOCK': 32, 'HAS_PASS': True, 'HAS_MROPE': True, 'MROPE_SECTION_H': 11, 'MROPE_SECTION_W': 10}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
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
buf0 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg4_1 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg3_1 = generate_example_value((4096,), (1,), 'cuda:0', torch.bfloat16, 0, (4096,))
buf1 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
buf5 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_fused_add_rms_norm_0.run(buf0, arg4_1, arg3_1, buf1, buf5, 8192, 4096, stream=raw_stream0)
del buf0, arg4_1, arg3_1, buf5

raw_stream0 = get_raw_stream(0)
buf6 = generate_example_value((8192, 24576), (24576, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24576))
buf7 = generate_example_value((8192, 12288), (12288, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12288))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_silu_slice_1.run(buf6, buf7, 100663296, stream=raw_stream0)
del buf6, buf7

raw_stream0 = get_raw_stream(0)
buf8 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg12_1 = generate_example_value((4096,), (1,), 'cuda:0', torch.bfloat16, 0, (4096,))
buf9 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
buf11 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_fused_add_rms_norm_2.run(buf8, buf1, arg12_1, buf9, buf11, 8192, 4096, stream=raw_stream0)
del buf1, buf8, arg12_1, buf9, buf11

raw_stream0 = get_raw_stream(0)
tmp_arg_0 = generate_example_value((8192, 8192), (10240, 1), 'cuda:0', torch.bfloat16, 0, (8192, 8192))
tmp_arg_1 = generate_example_value((8192, 1024), (10240, 1), 'cuda:0', torch.bfloat16, 8192, (8192, 1024))
buf13 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
buf14 = generate_example_value((8192, 1024), (1024, 1), 'cuda:0', torch.bfloat16, 0, (8192, 1024))
buf15 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg15_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg16_1 = generate_example_value((256,), (1,), 'cuda:0', torch.bfloat16, 0, (256,))
arg17_1 = generate_example_value((1048576, 64), (64, 1), 'cuda:0', torch.bfloat16, 0, (1048576, 64))
arg18_1 = generate_example_value((3, 8192), (8193, 1), 'cuda:0', torch.int64, 0, (3, 8192))
with torch.cuda._DeviceGuard(0):
    _fused_qk_rmsnorm_rope_gate_kernel_0.run(tmp_arg_0, tmp_arg_1, buf13, buf14, buf15, arg15_1, arg16_1, arg17_1, arg18_1, 10240, 10240, 4096, 1024, 4096, 64, 8193, 1, 8192, 20, 1, stream=raw_stream0)
del tmp_arg_0, tmp_arg_1, buf13, buf14, buf15, arg15_1, arg16_1, arg17_1, arg18_1

"""
# AOT ID: ['31_inference']
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


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/ca41f84b31ddced062c36c2a565445da6016cc52d3e5f10a6b27dc4f8f37731d/inductor_cache/pe/cpe22c4hr42zycb3qfr3lrfmo2m34etv4weqb6gpmznuf5zugo27.py
# Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, steer_buffer], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   add => add_6
#   float_1 => convert_element_type_2
#   fused_add_rms_norm_default => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   steer_buffer => convert_element_type_default_6
# Graph fragment:
#   %mm : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf2 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf2]
#   %arg3_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type_2 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %add_6 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_6), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   return %buf2,%convert_element_type_default_6,%convert_element_type_default_7
triton_red_fused__to_copy_add_fused_add_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 402661376}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.268443648, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp9 = tmp4.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp9, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = tl.full([1, 1], 4096.0, tl.float32)
        tmp16 = (tmp7 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 1.0, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp20 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/ca41f84b31ddced062c36c2a565445da6016cc52d3e5f10a6b27dc4f8f37731d/inductor_cache/on/confxmq2w4ehm3wdyhf5vr345edchenuqrz4ts4tjym53f75va4i.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul => mul_20
#   silu => add_19, convert_element_type_5, convert_element_type_6, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s18, 24576][24576, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 12288), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_5,), kwargs = {})
#   %exp : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_19 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_5, %add_19), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 12288, 9223372036854775807), kwargs = {})
#   %mul_20 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %slice_2), kwargs = {})
#   return %mul_20
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 805306368}, 'kernel_num_gb': 0.603979776, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 12288)
    x1 = xindex // 12288
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 24576*x1), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (12288 + x0 + 24576*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/ca41f84b31ddced062c36c2a565445da6016cc52d3e5f10a6b27dc4f8f37731d/inductor_cache/3z/c3zvtwxtvrmngxsovslkz7oyhcc4iexdlnomm3pa4st3kde36rig.py
# Topologically Sorted Source Nodes: [float_2, add_1, fused_add_rms_norm_default_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   add_1 => add_32
#   float_2 => convert_element_type_9
#   fused_add_rms_norm_default_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %mm_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %buf4 : Tensor  = PlaceHolder[target=buf4]
#   %buf10 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf10]
#   %arg12_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %convert_element_type_9 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_32), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf10,%convert_element_type_default_2,%convert_element_type_default_3
triton_red_fused__to_copy_add_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 335552512}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.201334784, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp9 = tmp4.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp9, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp15 = tl.full([1, 1], 4096.0, tl.float32)
        tmp16 = (tmp7 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 1.0, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp20 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# Original path: /BS/conformal-circuits/work/mamba/envs/pvf/lib/python3.12/site-packages/vllm/model_executor/layers/fused_qk_norm_rope.py:16
_fused_qk_rmsnorm_rope_gate_kernel_0 = async_compile.triton('_fused_qk_rmsnorm_rope_gate_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 2}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'declared_constexpr_names': ['num_q_heads', 'num_kv_heads', 'head_dim', 'rotary_dim', 'half_rotary', 'eps', 'norm_beta', 'INPUT_DTYPE', 'HEAD_BLOCK', 'ROT_HALF_BLOCK', 'HAS_PASS', 'HAS_MROPE', 'MROPE_SECTION_H', 'MROPE_SECTION_W'], 'kernel_name': '_fused_qk_rmsnorm_rope_gate_kernel_0', 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'q_gate_ptr': '*bf16', 'k_ptr': '*bf16', 'q_out_ptr': '*bf16', 'k_out_ptr': '*bf16', 'gate_out_ptr': '*bf16', 'q_weight_ptr': '*bf16', 'k_weight_ptr': '*bf16', 'cos_sin_cache_ptr': '*bf16', 'positions_ptr': '*i64', 'q_gate_stride_t': 'i32', 'k_stride_t': 'i32', 'q_out_stride_t': 'i32', 'k_out_stride_t': 'i32', 'gate_out_stride_t': 'i32', 'cache_stride_p': 'i32', 'positions_stride_m': 'i32', 'positions_stride_t': 'constexpr', 'num_q_heads': 'constexpr', 'num_kv_heads': 'constexpr', 'head_dim': 'constexpr', 'rotary_dim': 'constexpr', 'half_rotary': 'constexpr', 'eps': 'constexpr', 'norm_beta': 'constexpr', 'INPUT_DTYPE': 'constexpr', 'HEAD_BLOCK': 'constexpr', 'ROT_HALF_BLOCK': 'constexpr', 'HAS_PASS': 'constexpr', 'HAS_MROPE': 'constexpr', 'MROPE_SECTION_H': 'constexpr', 'MROPE_SECTION_W': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'positions_stride_t': 1, 'num_q_heads': 16, 'num_kv_heads': 4, 'head_dim': 256, 'rotary_dim': 64, 'half_rotary': 32, 'eps': 1e-06, 'norm_beta': 1.0, 'INPUT_DTYPE': triton.language.bfloat16, 'HEAD_BLOCK': 256, 'ROT_HALF_BLOCK': 32, 'HAS_PASS': True, 'HAS_MROPE': True, 'MROPE_SECTION_H': 11, 'MROPE_SECTION_W': 10}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg14_1
        s7 = arg19_1
        assert_size_stride(arg0_1, (s18, 32, 128), (4096, 128, 1), 'input')
        assert_size_stride(arg2_1, (4096, 4096), (4096, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [flatten, linear], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(arg0_1, (s18, 4096), (4096, 1), 0), reinterpret_tensor(arg2_1, (4096, 4096), (1, 4096), 0), out=buf0)
            del arg0_1
            del arg2_1
            assert_size_stride(arg4_1, (s18, 4096), (4096, 1), 'input')
            assert_size_stride(arg3_1, (4096, ), (1, ), 'input')
            arg4_1 = copy_if_misaligned(arg4_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf1 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, steer_buffer], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_fused_add_rms_norm_0.run(buf0, arg4_1, arg3_1, buf1, buf5, s18, 4096, stream=raw_stream0)
            del arg3_1
            del arg4_1
            del buf0
            assert_size_stride(arg7_1, (8192, ), (1, ), 'input')
            assert_size_stride(arg8_1, (8192, ), (1, ), 'input')
            assert_size_stride(arg9_1, (512, 4096), (4096, 1), 'input')
            assert_size_stride(arg10_1, (512, ), (1, ), 'input')
            assert_size_stride(arg11_1, (8192, ), (1, ), 'input')
            arg7_1 = copy_if_misaligned(arg7_1)
            arg8_1 = copy_if_misaligned(arg8_1)
            arg9_1 = copy_if_misaligned(arg9_1)
            arg10_1 = copy_if_misaligned(arg10_1)
            arg11_1 = copy_if_misaligned(arg11_1)
            # Topologically Sorted Source Nodes: [steer_buffer], Original ATen: [mia.steer_buffer]
            torch.ops.mia.steer_buffer.default(buf1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1)
            del arg10_1
            del arg11_1
            del arg7_1
            del arg8_1
            del arg9_1
            assert_size_stride(arg5_1, (24576, 4096), (4096, 1), 'input')
            arg5_1 = copy_if_misaligned(arg5_1)
            buf6 = empty_strided_cuda((s18, 24576), (24576, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, linear_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf5, reinterpret_tensor(arg5_1, (4096, 24576), (1, 4096), 0), out=buf6)
            del arg5_1
            del buf5
            buf7 = empty_strided_cuda((s18, 12288), (12288, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
            triton_poi_fused_mul_silu_slice_1_xnumel = 12288*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_slice_1.run(buf6, buf7, triton_poi_fused_mul_silu_slice_1_xnumel, stream=raw_stream0)
            del buf6
            assert_size_stride(arg6_1, (4096, 12288), (12288, 1), 'input')
            arg6_1 = copy_if_misaligned(arg6_1)
            buf8 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul, linear_2], Original ATen: [aten.slice, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf7, reinterpret_tensor(arg6_1, (12288, 4096), (1, 12288), 0), out=buf8)
            del arg6_1
            del buf7
            assert_size_stride(arg12_1, (4096, ), (1, ), 'input')
            arg12_1 = copy_if_misaligned(arg12_1)
            buf9 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            buf11 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_2, add_1, fused_add_rms_norm_default_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_fused_add_rms_norm_2.run(buf8, buf1, arg12_1, buf9, buf11, s18, 4096, stream=raw_stream0)
            del arg12_1
            assert_size_stride(arg13_1, (10240, 4096), (4096, 1), 'input')
            arg13_1 = copy_if_misaligned(arg13_1)
            buf12 = empty_strided_cuda((s18, 10240), (10240, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_2, add_1, fused_add_rms_norm_default_1, linear_3], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf11, reinterpret_tensor(arg13_1, (4096, 10240), (1, 4096), 0), out=buf12)
            del arg13_1
            buf13 = buf11; del buf11  # reuse
            buf14 = empty_strided_cuda((s18, 1024), (1024, 1), torch.bfloat16)
            buf15 = buf8; del buf8  # reuse
            assert_size_stride(arg15_1, (256, ), (1, ), 'input')
            assert_size_stride(arg16_1, (256, ), (1, ), 'input')
            assert_size_stride(arg17_1, (1048576, 64), (64, 1), 'input')
            assert_size_stride(arg18_1, (3, s18), (s7, 1), 'input')
            arg15_1 = copy_if_misaligned(arg15_1)
            arg16_1 = copy_if_misaligned(arg16_1)
            arg17_1 = copy_if_misaligned(arg17_1)
            arg18_1 = copy_if_misaligned(arg18_1)
            # Topologically Sorted Source Nodes: [split, triton_kernel_wrapper_mutation], Original ATen: [aten.split_with_sizes]
            raw_stream0 = get_raw_stream(0)
            _fused_qk_rmsnorm_rope_gate_kernel_0.run(reinterpret_tensor(buf12, (s18, 8192), (10240, 1), 0), reinterpret_tensor(buf12, (s18, 1024), (10240, 1), 8192), buf13, buf14, buf15, arg15_1, arg16_1, arg17_1, arg18_1, 10240, 10240, 4096, 1024, 4096, 64, s7, 1, s18, 20, 1, stream=raw_stream0)
            del arg15_1
            del arg16_1
            del arg17_1
            del arg18_1
            buf19 = buf1; del buf1  # reuse
        return (reinterpret_tensor(buf14, (s18, 4, 256), (1024, 256, 1), 0), reinterpret_tensor(buf12, (s18, 4, 256), (10240, 256, 1), 9216), reinterpret_tensor(buf13, (s18, 16, 256), (4096, 256, 1), 0), reinterpret_tensor(buf19, (s18, 16, 256), (4096, 256, 1), 0), buf15, buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 32, 128), (4096, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((24576, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((4096, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg9_1 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg12_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((10240, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = 8192
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((1048576, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((3, 8192), (8193, 1), device='cuda:0', dtype=torch.int64)
    arg19_1 = 8193
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
