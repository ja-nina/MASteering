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


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/u6/cu6v2m2iwmqgdefjpqgzzyhsz364lbb2d653km2tyyjhvzd7nskl.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul], Original ATen: [aten.view, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul => mul_4
#   sigmoid => sigmoid
#   view => view
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 16, 256][4096, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 4096]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   return %mul_4
triton_poi_fused_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 268435456}, 'kernel_num_gb': 0.201326592, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/n3/cn3oar37efgsfgmqsj4dxdjupkumilvvyhsocpnjkj2ipa5qc3nj.py
# Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   add => add_12
#   float_1 => convert_element_type_2
#   fused_add_rms_norm_default => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %mm : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg5_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %buf2 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf2]
#   %arg4_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convert_element_type_2 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_12 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_12), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   return %buf2,%convert_element_type_default_7
triton_red_fused__to_copy_add_fused_add_rms_norm_1 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 268443648}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.201334784, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 4096.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/h4/ch4rud4xbgvxfplw2olcnsqomc56sx333ffyufvvj4mb2ikwfzxo.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul_1], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul_1 => mul_21
#   silu => add_25, convert_element_type_5, convert_element_type_6, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s18, 24576][24576, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 12288), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_5,), kwargs = {})
#   %exp : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_25 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_5, %add_25), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 12288, 9223372036854775807), kwargs = {})
#   %mul_21 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %slice_2), kwargs = {})
#   return %mul_21
triton_poi_fused_mul_silu_slice_2 = async_compile.triton('triton_poi_fused_mul_silu_slice_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_slice_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 805306368}, 'kernel_num_gb': 0.603979776, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/do/cdok6ik6nebye43c4jtnfiordjhq67jgterdor2xalgzwg2zpedv.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_default, capture_hs, float_2, add_1, fused_add_rms_norm_default_1], Original ATen: [vllm_ir.fused_add_rms_norm, mia.capture_hs, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_38
#   capture_hs => capture_hs_default
#   float_2 => convert_element_type_9
#   fused_add_rms_norm_default => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_default_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %mm_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg5_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %buf11 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf11]
#   %arg10_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convert_element_type_default_4 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.bfloat16), kwargs = {})
#   %capture_hs_default : [num_users=0] = call_function[target=torch.ops.mia.capture_hs.default](args = (%mm_2, %convert_element_type_default_6, %arg8_1, %arg9_1, 1), kwargs = {})
#   %convert_element_type_9 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.float32), kwargs = {})
#   %add_38 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_38), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf11,%buf7,%convert_element_type_default_2,%convert_element_type_default_3
triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 603987968}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.402661376, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp7, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 4096.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.full([1, 1], 1.0, tl.float32)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp30 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 4096*x0), tmp36, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/6h/c6hor3ruxdenjoe5tlui2osamslbfshvpcetwihojrug5sesqzzg.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 32, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_4 = async_compile.triton('triton_poi_fused_zeros_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 134217728}, 'kernel_num_gb': 0.067108864, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
arg0_1 = generate_example_value((8192, 16, 256), (4096, 256, 1), 'cuda:0', torch.bfloat16, 0, (8192, 16, 256))
arg2_1 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
buf0 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, 33554432, stream=raw_stream0)
del arg0_1, arg2_1, buf0

raw_stream0 = get_raw_stream(0)
buf1 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg5_1 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg4_1 = generate_example_value((4096,), (1,), 'cuda:0', torch.bfloat16, 0, (4096,))
buf3 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_fused_add_rms_norm_1.run(buf1, arg5_1, arg4_1, buf3, 8192, 4096, stream=raw_stream0)
del arg4_1, buf3

raw_stream0 = get_raw_stream(0)
buf4 = generate_example_value((8192, 24576), (24576, 1), 'cuda:0', torch.bfloat16, 0, (8192, 24576))
buf5 = generate_example_value((8192, 12288), (12288, 1), 'cuda:0', torch.bfloat16, 0, (8192, 12288))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_silu_slice_2.run(buf4, buf5, 100663296, stream=raw_stream0)
del buf4, buf5

raw_stream0 = get_raw_stream(0)
buf6 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg10_1 = generate_example_value((4096,), (1,), 'cuda:0', torch.bfloat16, 0, (4096,))
buf7 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
buf10 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
buf12 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3.run(buf6, buf1, arg5_1, arg10_1, buf7, buf10, buf12, 8192, 4096, stream=raw_stream0)
del buf1, arg5_1, buf6, arg10_1, buf7, buf10, buf12

raw_stream0 = get_raw_stream(0)
buf15 = generate_example_value((8192, 32, 128), (4096, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 32, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_zeros_4.run(buf15, 33554432, stream=raw_stream0)
del buf15

"""
# AOT ID: ['4_inference']
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


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/u6/cu6v2m2iwmqgdefjpqgzzyhsz364lbb2d653km2tyyjhvzd7nskl.py
# Topologically Sorted Source Nodes: [view, sigmoid, mul], Original ATen: [aten.view, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul => mul_4
#   sigmoid => sigmoid
#   view => view
# Graph fragment:
#   %arg0_1 : Tensor "bf16[s18, 16, 256][4096, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [-1, 4096]), kwargs = {})
#   %sigmoid : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg2_1,), kwargs = {})
#   %mul_4 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %sigmoid), kwargs = {})
#   return %mul_4
triton_poi_fused_mul_sigmoid_view_0 = async_compile.triton('triton_poi_fused_mul_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 268435456}, 'kernel_num_gb': 0.201326592, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/n3/cn3oar37efgsfgmqsj4dxdjupkumilvvyhsocpnjkj2ipa5qc3nj.py
# Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   add => add_12
#   float_1 => convert_element_type_2
#   fused_add_rms_norm_default => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %mm : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg5_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %buf2 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf2]
#   %arg4_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convert_element_type_2 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_12 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1.0), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor_2, %add_12), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_3, torch.bfloat16), kwargs = {})
#   return %buf2,%convert_element_type_default_7
triton_red_fused__to_copy_add_fused_add_rms_norm_1 = async_compile.triton('triton_red_fused__to_copy_add_fused_add_rms_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_fused_add_rms_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 268443648}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.201334784, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_fused_add_rms_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 4096.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/h4/ch4rud4xbgvxfplw2olcnsqomc56sx333ffyufvvj4mb2ikwfzxo.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul_1], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul_1 => mul_21
#   silu => add_25, convert_element_type_5, convert_element_type_6, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s18, 24576][24576, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 12288), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_5,), kwargs = {})
#   %exp : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_25 : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_5, %add_25), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s18, 12288][24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 12288, 9223372036854775807), kwargs = {})
#   %mul_21 : Tensor "bf16[s18, 12288][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %slice_2), kwargs = {})
#   return %mul_21
triton_poi_fused_mul_silu_slice_2 = async_compile.triton('triton_poi_fused_mul_silu_slice_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_slice_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 805306368}, 'kernel_num_gb': 0.603979776, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/do/cdok6ik6nebye43c4jtnfiordjhq67jgterdor2xalgzwg2zpedv.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_default, capture_hs, float_2, add_1, fused_add_rms_norm_default_1], Original ATen: [vllm_ir.fused_add_rms_norm, mia.capture_hs, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_38
#   capture_hs => capture_hs_default
#   float_2 => convert_element_type_9
#   fused_add_rms_norm_default => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_default_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %mm_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg5_1 : Tensor "bf16[s18, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %buf11 : Tensor "f32[s18, 1][1, s18]cuda:0" = PlaceHolder[target=buf11]
#   %arg10_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convert_element_type_default_4 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.bfloat16), kwargs = {})
#   %capture_hs_default : [num_users=0] = call_function[target=torch.ops.mia.capture_hs.default](args = (%mm_2, %convert_element_type_default_6, %arg8_1, %arg9_1, 1), kwargs = {})
#   %convert_element_type_9 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.float32), kwargs = {})
#   %add_38 : Tensor "f32[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1.0), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s18, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[s18, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_tensor, %add_38), kwargs = {})
#   %convert_element_type_default_3 : Tensor "bf16[s18, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_1, torch.bfloat16), kwargs = {})
#   return %buf11,%buf7,%convert_element_type_default_2,%convert_element_type_default_3
triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3 = async_compile.triton('triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 603987968}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.402661376, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp7, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 4096.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.full([1, 1], 1.0, tl.float32)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp30 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 4096*x0), tmp36, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /BS/ForNet/work/MASteering/.cache/vllm/torch_compile_cache/torch_aot_compile/6ec9428c9b91e774ece4c0a5d8e9e3069dbad9c1a47a6ba8c157a56002a79a3c/inductor_cache/6h/c6hor3ruxdenjoe5tlui2osamslbfshvpcetwihojrug5sesqzzg.py
# Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   zeros => full_default
# Graph fragment:
#   %full_default : Tensor "bf16[s18, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 32, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_4 = async_compile.triton('triton_poi_fused_zeros_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 134217728}, 'kernel_num_gb': 0.067108864, 'kernel_flop': 0, 'backend_hash': '5AC87545237B69B99BDD038D5349F4899684E13C8CDB69786C27F9D868C0C0F6', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1 = args
        args.clear()
        s59 = arg1_1
        s18 = arg13_1
        assert_size_stride(arg0_1, (s18, 16, 256), (4096, 256, 1), 'input')
        assert_size_stride(arg2_1, (s18, 4096), (4096, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul], Original ATen: [aten.view, aten.sigmoid, aten.mul]
            triton_poi_fused_mul_sigmoid_view_0_xnumel = 4096*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_sigmoid_view_0.run(arg0_1, arg2_1, buf0, triton_poi_fused_mul_sigmoid_view_0_xnumel, stream=raw_stream0)
            del arg0_1
            del arg2_1
            assert_size_stride(arg3_1, (4096, 4096), (4096, 1), 'input')
            arg3_1 = copy_if_misaligned(arg3_1)
            buf1 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view, sigmoid, mul, linear], Original ATen: [aten.view, aten.sigmoid, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf0, reinterpret_tensor(arg3_1, (4096, 4096), (1, 4096), 0), out=buf1)
            del arg3_1
            assert_size_stride(arg5_1, (s18, 4096), (4096, 1), 'input')
            assert_size_stride(arg4_1, (4096, ), (1, ), 'input')
            arg5_1 = copy_if_misaligned(arg5_1)
            arg4_1 = copy_if_misaligned(arg4_1)
            buf3 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_fused_add_rms_norm_1.run(buf1, arg5_1, arg4_1, buf3, s18, 4096, stream=raw_stream0)
            del arg4_1
            assert_size_stride(arg6_1, (24576, 4096), (4096, 1), 'input')
            arg6_1 = copy_if_misaligned(arg6_1)
            buf4 = empty_strided_cuda((s18, 24576), (24576, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, add, fused_add_rms_norm_default, linear_1], Original ATen: [aten._to_copy, aten.add, vllm_ir.fused_add_rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf3, reinterpret_tensor(arg6_1, (4096, 24576), (1, 4096), 0), out=buf4)
            del arg6_1
            del buf3
            buf5 = empty_strided_cuda((s18, 12288), (12288, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul_1], Original ATen: [aten.slice, aten.silu, aten.mul]
            triton_poi_fused_mul_silu_slice_2_xnumel = 12288*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_slice_2.run(buf4, buf5, triton_poi_fused_mul_silu_slice_2_xnumel, stream=raw_stream0)
            del buf4
            assert_size_stride(arg7_1, (4096, 12288), (12288, 1), 'input')
            arg7_1 = copy_if_misaligned(arg7_1)
            buf6 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul_1, linear_2], Original ATen: [aten.slice, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf5, reinterpret_tensor(arg7_1, (12288, 4096), (1, 12288), 0), out=buf6)
            del arg7_1
            assert_size_stride(arg10_1, (4096, ), (1, ), 'input')
            arg10_1 = copy_if_misaligned(arg10_1)
            buf7 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            buf10 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            buf12 = empty_strided_cuda((s18, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_default, capture_hs, float_2, add_1, fused_add_rms_norm_default_1], Original ATen: [vllm_ir.fused_add_rms_norm, mia.capture_hs, aten._to_copy, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_capture_hs_fused_add_rms_norm_3.run(buf6, buf1, arg5_1, arg10_1, buf7, buf10, buf12, s18, 4096, stream=raw_stream0)
            del arg10_1
            del arg5_1
            del buf1
            assert_size_stride(arg8_1, (32769, 4096), (4096, 1), 'input')
            assert_size_stride(arg9_1, (8192, ), (1, ), 'input')
            arg9_1 = copy_if_misaligned(arg9_1)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_default, capture_hs], Original ATen: [vllm_ir.fused_add_rms_norm, mia.capture_hs]
            torch.ops.mia.capture_hs.default(buf6, buf7, arg8_1, arg9_1, 1)
            del arg8_1
            del arg9_1
            del buf6
            del buf7
            assert_size_stride(arg11_1, (12288, 4096), (4096, 1), 'input')
            arg11_1 = copy_if_misaligned(arg11_1)
            buf13 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf12, reinterpret_tensor(arg11_1, (4096, 12288), (1, 4096), 0), out=buf13)
            del arg11_1
            assert_size_stride(arg12_1, (64, 4096), (4096, 1), 'input')
            arg12_1 = copy_if_misaligned(arg12_1)
            buf14 = empty_strided_cuda((s18, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf12, reinterpret_tensor(arg12_1, (4096, 64), (1, 4096), 0), out=buf14)
            del arg12_1
            buf15 = reinterpret_tensor(buf12, (s18, 32, 128), (4096, 128, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [zeros], Original ATen: [aten.zeros]
            triton_poi_fused_zeros_4_xnumel = 4096*s18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_zeros_4.run(buf15, triton_poi_fused_zeros_4_xnumel, stream=raw_stream0)
        return (buf13, buf14, buf15, buf10, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 16, 256), (4096, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 8192
    arg2_1 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((24576, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((4096, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((32769, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg10_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((64, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = 8192
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
