import math
import gc, itertools
import pytest
import torch
# from tests.test_helpers.utils_fp4 import (
#     cast_from_fp4,
#     recover_swizzled_scales,
#     ref_fp4_quant,
# )
# from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance
import einops
# from tests.test_helpers.sink_attention_reference import sink_attention_unified

import flashinfer
from flashinfer.utils import FP4Tensor, ceil_div, round_up, get_compute_capability


import torch

import flashinfer.utils as utils

import numpy as np
from flashinfer import testing
import pandas as pd

def benchmark_fn(fn, docstring: str):
    measured_times = testing.bench_gpu_time_with_cuda_event(fn)
    ms = np.average(measured_times)
    print("\n \033[1;32m" + docstring + f" Benchmark Time: {ms:.3f} ms" + "\033[0m")
    return ms


def save_test_results_to_excel(test_results, filename, test_name=None, show_stats=True, verbose=True):
    """
    将测试结果保存到Excel文件的通用函数
    
    Parameters:
    -----------
    test_results : list
        包含测试结果字典的列表，每个字典包含测试参数和结果
    filename : str
        Excel文件名，可以包含或不包含.xlsx扩展名
    test_name : str, optional
        测试名称，用于打印信息。如果为None，使用filename
    show_stats : bool, optional
        是否显示统计信息，默认True
    verbose : bool, optional
        是否显示详细进度信息，默认True
        
    Returns:
    --------
    dict
        包含保存结果的统计信息
    """
    if not test_results:
        if verbose:
            print("没有测试结果需要保存")
        return {"saved": False, "total_tests": 0, "successful_tests": 0, "failed_tests": 0}
    
    # 确保文件名有正确的扩展名
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    # 使用文件名作为默认测试名称
    if test_name is None:
        test_name = filename.replace('.xlsx', '')
    
    try:
        # 创建DataFrame
        df = pd.DataFrame(test_results)
        
        # 使用ExcelWriter来实现更好的格式控制
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='测试结果', index=False)
            
            # 获取工作表和工作簿对象
            worksheet = writer.sheets['测试结果']
            workbook = writer.book
            
            # 定义表头格式
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # 定义数据格式
            cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })
            
            # 应用表头格式
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # 计算每列的最佳宽度
            for i, col in enumerate(df.columns):
                # 计算列名长度
                column_len = len(str(col))
                
                # 计算该列数据的最大长度
                if len(df) > 0:
                    max_len = df[col].astype(str).str.len().max()
                    column_len = max(column_len, max_len)
                
                # 设置合理的列宽范围（最小8，最大50）
                column_len = min(max(column_len + 2, 8), 50)
                worksheet.set_column(i, i, column_len, cell_format)
            
            # 如果数据量大，启用自动筛选
            if len(df) > 1:
                worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
        
        if verbose:
            print(f"\n测试结果已保存到 {filename}")
            print(f"总共完成 {len(test_results)} 个测试用例")
        
        # 统计信息
        successful_tests = df[df['latency_ms'].notna()]
        failed_tests_count = len(test_results) - len(successful_tests)
        
        stats = {
            "saved": True,
            "filename": filename,
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": failed_tests_count
        }
        
        if show_stats and verbose:
            if len(successful_tests) > 0:
                print(f"成功测试: {len(successful_tests)} 个")
                if 'latency_ms' in successful_tests.columns:
                    stats["avg_latency"] = successful_tests['latency_ms'].mean()
                    stats["min_latency"] = successful_tests['latency_ms'].min()  
                    stats["max_latency"] = successful_tests['latency_ms'].max()
                    print(f"平均延迟: {stats['avg_latency']:.3f} ms")
                    print(f"延迟范围: {stats['min_latency']:.3f} - {stats['max_latency']:.3f} ms")
            
            if failed_tests_count > 0:
                print(f"失败测试: {failed_tests_count} 个")
        
        return stats
        
    except Exception as e:
        error_msg = f"保存Excel文件时出错: {str(e)}"
        if verbose:
            print(error_msg)
        return {
            "saved": False, 
            "error": error_msg,
            "total_tests": len(test_results),
            "successful_tests": 0,
            "failed_tests": len(test_results)
        }


def collect_test_result(test_params, latency=None, error=None, extra_data=None):
    """
    收集单个测试结果的辅助函数
    
    Parameters:
    -----------
    test_params : dict
        测试参数字典
    latency : float, optional
        测试延迟结果（毫秒）
    error : str, optional
        错误信息（如果测试失败）
    extra_data : dict, optional
        额外需要记录的数据
        
    Returns:
    --------
    dict
        格式化的测试结果字典
    """
    result = test_params.copy()
    
    if latency is not None:
        result['latency_ms'] = latency
    else:
        result['latency_ms'] = None
        
    if error is not None:
        result['error'] = error
        
    if extra_data:
        result.update(extra_data)
        
    return result


FLOAT4_E2M1_MAX = 6.0

# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def cast_from_fp4(x):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    new_shape = c.shape[:-2] + (
        c.shape[-2] * c.shape[-1],
    )  # fuse the dim added by stack
    lookup_table = torch.tensor(E2M1_TO_FLOAT32, device=c.device)
    out = lookup_table[c.to(torch.long)].reshape(new_shape).to(torch.float32)
    return out


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def ref_fp4_quant(x, global_scale, block_size, sf_use_ue8m0=False):
    assert isinstance(global_scale, (float, int)) or global_scale.dtype == torch.float32

    sliced_shape = x.shape[:-1] + (x.shape[-1] // block_size, block_size)
    sliced_x = torch.reshape(x, sliced_shape)
    vec_max = torch.max(torch.abs(sliced_x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    if sf_use_ue8m0:
        scale = (scale.view(torch.int32) + 0x007FFFFF) & 0x7F800000
        scale = scale.view(torch.float32)
    else:
        scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = sliced_x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(x.shape)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale, m, n, block_size, sf_start_index=0):
    assert sf_start_index + m <= scale.shape[0]
    full_m = scale.shape[0]
    scale_n = n // block_size
    rounded_n = utils.round_up(scale_n, 4)
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, full_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (full_m, rounded_n)).to(torch.float32)
    return result[sf_start_index : sf_start_index + m, :scale_n]



def assert_close_with_mismatch_tolerance(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_mismatched_elements: int = 0,
):
    """
    Asserts that two tensors are close, allowing for a specified number of mismatched elements.
    This function correctly implements the same logic as torch.isclose.
    """
    # Ensure tensors are float for comparison
    actual_float = actual.float()
    expected_float = expected.float()

    # This is the core logic from torch.isclose
    # A mismatch occurs if the difference is greater than the combined tolerance
    mismatched = torch.abs(actual_float - expected_float) > (
        atol + rtol * torch.abs(expected_float)
    )

    num_mismatched = torch.sum(mismatched).item()

    if num_mismatched > max_mismatched_elements:
        # For a helpful error message, let's find the worst offenders
        actual_flat = actual_float.flatten()
        expected_flat = expected_float.flatten()
        abs_diff = torch.abs(actual_flat - expected_flat)

        # Calculate relative difference only where expected is not zero to avoid division by zero
        # Add a small epsilon to the denominator for stability
        rel_diff = abs_diff / (torch.abs(expected_flat) + 1e-12)

        total_elements = actual_flat.numel()

        raise AssertionError(
            f"Tensors are not close enough!\n"
            f"Mismatched elements: {num_mismatched} / {total_elements} "
            f"({100.0 * num_mismatched / total_elements:.2f}%)\n"
            f"Allowed mismatched elements: {max_mismatched_elements}, but found {num_mismatched}.\n"
            f"Greatest absolute difference: {torch.max(abs_diff).item():.4g} (atol={atol})\n"
            f"Greatest relative difference: {torch.max(rel_diff).item():.4g} (rtol={rtol})"
        )


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "nvfp4": "nvfp4",
}

GPU_DEVICE = "cuda:0"

global_workspace_buffer = None  # can.be empty initialized
global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
workspace_size = 256 * 1024 * 1024


def flip_coin(*args, **kwargs):
    # Use any test parameters to deterministically decide branch
    # This makes test configurations go through different paths
    param_tuple = args + tuple(sorted(kwargs.items()))
    hash_value = hash(param_tuple)
    return (hash_value % 2) == 0


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def generate_seq_lens_prefill(batch_size, max_q_len, max_in_kv_len):
    # q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    # q_lens[-1] = max_q_len
    # in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int)
    # in_kv_lens[-1] = max_in_kv_len
    q_lens = torch.full((batch_size,), max_q_len, dtype=torch.int32)
    in_kv_lens = torch.full((batch_size,), max_in_kv_len, dtype=torch.int)
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens


def generate_seq_lens_decode(batch_size, q_len_per_req, max_in_kv_len):
    q_lens = torch.full((batch_size,), q_len_per_req, dtype=torch.int32)
    # in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int)
    # in_kv_lens[-1] = max_in_kv_len
    in_kv_lens = torch.full((batch_size,), max_in_kv_len, dtype=torch.int)
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens


def generate_cumsum_lens(lens):
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=GPU_DEVICE),
            torch.cumsum(lens.to(GPU_DEVICE), dim=0, dtype=torch.int32),
        ]
    )


def create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype):
    q = torch.randn(
        torch.sum(q_lens).item(),
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16 if q_dtype == "fp8" else DTYPE_MAP[q_dtype],
        device=GPU_DEVICE,
    )
    if q_dtype == "fp8":
        q, q_scale = to_float8(q)
        # Reference implementation have functional issue or low precision with fp8, use bfloat16 and fake-quantization instead.
        ref_q = q.bfloat16() * q_scale
    else:
        q_scale = 1.0
        ref_q = q

    return q, q_scale, ref_q


def create_kv_cache(
    batch_size, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype, ref_kv_dtype
):
    # Create separate K and V caches
    max_seq_len = torch.max(seq_lens).item()
    num_tokens = max_seq_len * batch_size
    num_pages = (num_tokens + page_size - 1) // page_size
    ref_kv_dtype_torch = DTYPE_MAP[ref_kv_dtype]
    if kv_dtype != "fp8":  # for fp8, create with high precision to generate scale.
        assert kv_dtype == ref_kv_dtype, (
            "kv_dtype and ref_kv_dtype must be the same for non-fp8 kv_cache"
        )

    k_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=ref_kv_dtype_torch,
        device=GPU_DEVICE,
    )
    v_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=ref_kv_dtype_torch,
        device=GPU_DEVICE,
    )

    # Convert K and V separately to fp8 if needed
    if kv_dtype == "fp8":
        k_cache, k_scale = to_float8(k_cache)
        v_cache, v_scale = to_float8(v_cache)
        # use high precision and fake-quantization for reference to avoid precision/functional issue
        ref_kv_cache = torch.stack(
            [
                k_cache.to(ref_kv_dtype_torch) * k_scale,
                v_cache.to(ref_kv_dtype_torch) * v_scale,
            ],
            dim=1,
        )
    else:
        k_scale = v_scale = 1.0
        ref_kv_cache = torch.stack([k_cache, v_cache], dim=1)
    # Combine K and V into interleaved format for the API
    kv_cache = torch.stack([k_cache, v_cache], dim=1)

    return kv_cache, k_scale, v_scale, ref_kv_cache


def create_page_table(batch_size, seq_lens, page_size):
    page_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_pages_per_seq = torch.max(page_per_seq).item()

    # Generate random but unique page IDs for all sequences
    total_pages_needed = torch.sum(page_per_seq).item()
    all_page_ids = torch.randperm(
        total_pages_needed, dtype=torch.int32, device=GPU_DEVICE
    )

    # Generate unique page IDs for all sequences
    page_tables = torch.zeros(
        (batch_size, max_num_pages_per_seq), dtype=torch.int32, device=GPU_DEVICE
    )

    # Populate page tables and track page assignments
    page_id = 0
    for i in range(batch_size):
        num_pages_needed = page_per_seq[i]
        page_tables[i, :num_pages_needed] = all_page_ids[
            page_id : page_id + num_pages_needed
        ]
        page_id += num_pages_needed
    return page_tables, all_page_ids, page_per_seq


def flatten_paged_kv(
    ref_kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    kv_last_page_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flat K/V and token-level indptr from paged KV cache and page table."""
    device = ref_kv_cache.device
    batch_size = int(page_table.shape[0])

    # Move loop-control tensors to CPU to avoid GPU sync in loops
    page_table_cpu = page_table.cpu()
    seq_lens_cpu = seq_lens.cpu()
    kv_last_page_len_cpu = kv_last_page_len.cpu()
    page_per_seq = (seq_lens_cpu + page_size - 1) // page_size
    k_list = []
    v_list = []
    for i in range(batch_size):
        pages_i = int(page_per_seq[i].item())
        last_len_i = int(kv_last_page_len_cpu[i].item())
        for j in range(pages_i):
            page_id = int(page_table_cpu[i, j].item())
            k_page = ref_kv_cache[page_id, 0]
            v_page = ref_kv_cache[page_id, 1]
            if j == pages_i - 1:
                k_page = k_page[:, :last_len_i, :]
                v_page = v_page[:, :last_len_i, :]
            k_list.append(einops.rearrange(k_page, "h p d -> p h d"))
            v_list.append(einops.rearrange(v_page, "h p d -> p h d"))
    k_flat = torch.cat(k_list, dim=0)
    v_flat = torch.cat(v_list, dim=0)
    kv_indptr_tokens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )
    return k_flat, v_flat, kv_indptr_tokens


def create_workspace_buffers(device):
    # Lazily initialize and reuse global workspace buffers
    global global_workspace_buffer, global_trtllm_gen_fmha_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    return global_trtllm_gen_fmha_workspace_buffer, global_workspace_buffer


def create_output(q, o_dtype, create_out_tensor, create_out_dtype):
    if o_dtype == "fp8":
        o_scale = torch.rand(1).item() * 0.5 + 0.5  # Scale range: 0.5 ~ 1.0
    else:
        o_scale = 1.0
    o_sf_scale = (
        300 if o_dtype == "nvfp4" else None
    )  # choose a value to make error smaller by testing.
    o_sf_vec_size = 16 if o_dtype == "nvfp4" else None

    if create_out_tensor:
        if o_dtype == "nvfp4":
            fp4_out_shape = q.shape[:-1] + (ceil_div(q.shape[-1], 2),)

            extra_size = torch.randint(0, 256, (1,)).item()

            fp4_out_scale_shape = (
                round_up(q.shape[0] + extra_size, 128),
                round_up(q.shape[1] * q.shape[2] // o_sf_vec_size, 4),
            )

            out_scale_factor = torch.empty(
                fp4_out_scale_shape, dtype=torch.float8_e4m3fn, device=q.device
            )
            rounded_extra_size = fp4_out_scale_shape[0] - q.shape[0]
            o_sf_start_index = (
                torch.randint(0, rounded_extra_size, (1,)).item()
                if rounded_extra_size > 0
                else 0
            )
            out_data = torch.empty(fp4_out_shape, dtype=torch.uint8, device=q.device)
            out = FP4Tensor(out_data, out_scale_factor, o_sf_start_index)
        else:
            out = torch.empty_like(q, dtype=DTYPE_MAP[o_dtype])
    else:
        out = None
    out_dtype = DTYPE_MAP[o_dtype] if create_out_dtype else None
    return out, out_dtype, o_scale, o_sf_scale, o_sf_vec_size


def get_last_page_len(seq_lens, page_size):
    kv_last_page_len = seq_lens % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size
    return kv_last_page_len


def unpack_compare_nvfp4(
    output: FP4Tensor,
    output_ref,
    o_sf_scale,
    o_sf_vec_size,
    sf_rtol=2e-1,
    sf_atol=2e-1,
    rmse_tol=0.3,
):
    output_ref, out_scale_factor_ref = ref_fp4_quant(
        output_ref, o_sf_scale, o_sf_vec_size
    )

    output_unpacked = cast_from_fp4(output.data)
    out_scale_factor = recover_swizzled_scales(
        output.scale,
        output_unpacked.shape[0],
        math.prod(list(output_unpacked.shape[1:])),
        o_sf_vec_size,
        output.scale_start_index,
    )

    torch.testing.assert_close(
        out_scale_factor.float().reshape(out_scale_factor_ref.shape),
        out_scale_factor_ref.float(),
        rtol=sf_rtol,
        atol=sf_atol,
    )
    rmse = torch.sqrt(torch.mean((output_unpacked.float() - output_ref.float()) ** 2))
    assert rmse.item() < rmse_tol
    return output_unpacked, output_ref


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (4, 16, 2, 1),
        (4, 32, 4, 5),
        (4, 64, 4, 8),
        (128, 16, 2, 5),
        (128, 32, 4, 1),
        (128, 64, 2, 8),
        (256, 16, 4, 8),
        (256, 32, 2, 8),
        (256, 64, 4, 1),
        (256, 64, 4, 5),
    ],
)
@pytest.mark.parametrize("window_left", [-1])  # todo(Siyuan): add 127 window_left
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("enable_sink", [True, False])
def test_trtllm_batch_prefill(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
    enable_sink,
    seq_len,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] in [11, 12]:
        pytest.skip("trtllm-gen does not support SM110/SM120/SM121 GPUs.")
    # Set up test parameters
    torch.manual_seed(0)
    head_dim = 128
    MAX_Q_LEN = seq_len
    MAX_IN_KV_LEN = seq_len

    # Generate random sequence lengths
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens_prefill(
        batch_size, MAX_Q_LEN, MAX_IN_KV_LEN
    )

    # Create query tensor and related data
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)

    # Create KV cache and related data
    kv_cache, k_scale, v_scale, ref_kv_cache = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(GPU_DEVICE)

    # Create output tensor and related data
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    can_infer_type = q.dtype == DTYPE_MAP[o_dtype] or create_out_tensor
    create_out_dtype = not can_infer_type or flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype, q_dtype
    )
    out, out_dtype, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor, create_out_dtype
    )

    sm_scale = float(1.0 / (head_dim**0.5))

    # # Build reference output
    # plan_params = {
    #     "qo_indptr": q_indptr,
    #     "paged_kv_indptr": kv_indptr,
    #     "paged_kv_indices": all_page_ids,
    #     "paged_kv_last_page_len": kv_last_page_len.to(GPU_DEVICE),
    #     "num_qo_heads": num_qo_heads,
    #     "num_kv_heads": num_kv_heads,
    #     "head_dim_qk": head_dim,
    #     "page_size": page_size,
    #     "causal": True,
    #     "pos_encoding_mode": "NONE",
    #     "logits_soft_cap": 0.0,
    #     "q_data_type": ref_q.dtype,
    #     "kv_data_type": ref_kv_cache.dtype,
    #     "window_left": window_left,
    # }
    # if not enable_sink:
    #     wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
    #         workspace_buffer_ref, kv_layout
    #     )
    #     wrapper_ref.plan(**plan_params)
    #     output_ref = wrapper_ref.run(ref_q, ref_kv_cache)
    # else:
    #     # Construct flat K/V via helper
    #     k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
    #         ref_kv_cache,
    #         page_table,
    #         seq_lens.to(GPU_DEVICE),
    #         page_size,
    #         kv_last_page_len,
    #     )
    #     sink = torch.rand(num_qo_heads, device=GPU_DEVICE, dtype=torch.float32) * 5
    #     output_ref = sink_attention_unified(
    #         ref_q,
    #         k_flat,
    #         v_flat,
    #         sink,
    #         window_left,
    #         True,
    #         sm_scale,
    #         mode="varlen",
    #         batch_size=batch_size,
    #         qo_indptr=q_indptr,
    #         kv_indptr=kv_indptr_tokens,
    #     )

    # Run trtllm-gen function call
    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens.to(GPU_DEVICE),
        torch.max(q_lens).item(),
        torch.max(seq_lens).item(),
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
        batch_size,
        q_indptr,
        kv_indptr,
        window_left,  # window_left
        out=out,
        out_dtype=out_dtype,
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        enable_pdl=enable_pdl,
        sinks=(sink if enable_sink else None),
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()

    def fn():
        _ = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                q.contiguous(),
                kv_cache,
                workspace_buffer,
                page_table,
                seq_lens.to(GPU_DEVICE),
                torch.max(q_lens).item(),
                torch.max(seq_lens).item(),
                q_scale * k_scale * sm_scale,  # bmm1_scale
                v_scale / o_scale,  # bmm2_scale
                batch_size,
                q_indptr,
                kv_indptr,
                window_left,  # window_left
                out=out,
                out_dtype=out_dtype,
                o_sf_scale=o_sf_scale,
                o_sf_vec_size=o_sf_vec_size,
                enable_pdl=enable_pdl,
                sinks=(sink if enable_sink else None),
            )
    latency = benchmark_fn(fn, f"trtllm_batch_context_with_kv_cache: batch: {batch_size}, seq_len: {seq_len}, q_dtype: {q_dtype}, kv_dtype: {kv_dtype}, o_dtype: {o_dtype}")
    return latency
    
    # if o_dtype == "nvfp4":
    #     output, output_ref = unpack_compare_nvfp4(
    #         output, output_ref, o_sf_scale, o_sf_vec_size
    #     )
    #     assert o_scale == 1.0
    #     rtol, atol = 4e-1, 1e0
    # elif q_dtype == "fp8" and o_dtype == "fp8":
    #     rtol, atol = 5e-2, 7e-2
    # elif q_dtype == "fp8" and o_dtype in ["bf16", "fp16"]:
    #     rtol, atol = 4e-2, 6e-2
    # else:
    #     rtol, atol = 1e-2, 1e-2

    # # Arbitary small mismatch rate
    # allowed_mismatch_rate = 1e-7
    # # Calculate max allowed mismatched elements based on tensor size
    # total_elements = (output.float() * o_scale).numel()
    # max_mismatched_elements = int(allowed_mismatch_rate * total_elements)

    # # convert to float32 for fp8 is not supported by assert_close
    # assert_close_with_mismatch_tolerance(
    #     output.float() * o_scale,
    #     output_ref.float(),
    #     rtol=rtol,
    #     atol=atol,
    #     max_mismatched_elements=max_mismatched_elements,
    # )

    # if o_dtype != "nvfp4":  # wrapper api does not support fp4 output yet.
    #     # test wrapper with trtllm-gen backend
    #     wrapper_trtllm_gen = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
    #         workspace_buffer, kv_layout, backend="trtllm-gen"
    #     )
    #     plan_params["q_data_type"] = q.dtype
    #     plan_params["kv_data_type"] = kv_cache.dtype
    #     wrapper_trtllm_gen.plan(**plan_params)
    #     output_wrapper = wrapper_trtllm_gen.run(
    #         q.contiguous(),
    #         kv_cache,
    #         q_scale=q_scale,
    #         k_scale=k_scale,
    #         v_scale=v_scale / o_scale,
    #         enable_pdl=enable_pdl,
    #         sinks=(sink if enable_sink else None),
    #     )
    #     # v_scale, o_scale in wrapper is emulated by multiplying output by v_scale instead of fused into kernel.
    #     if v_scale == o_scale == 1.0:
    #         assert (output_wrapper == output).all()
    #     else:
    #         torch.testing.assert_close(
    #             output.float(), output_wrapper.float(), rtol=1e-1, atol=1e-1
    #         )
    #     # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    #     # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    #     assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


@pytest.mark.parametrize("kv_layout", ["HND"])  # trtllm-gen only support HND
@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 1, 16, 2, 1),
        (4, 1, 32, 2, 5),
        (4, 2, 64, 2, 5),
        (4, 3, 32, 2, 5),
        (4, 3, 64, 2, 1),
        (4, 4, 64, 4, 1),
        (4, 5, 64, 4, 8),
        (128, 1, 64, 2, 5),
        (128, 2, 32, 4, 1),
        (128, 3, 16, 4, 8),
        (128, 4, 16, 2, 5),
        (128, 5, 16, 2, 5),
        (256, 1, 64, 4, 8),
        (256, 2, 16, 2, 8),
        (256, 3, 64, 4, 5),
        (256, 4, 32, 2, 8),
        (256, 5, 32, 2, 1),
    ],
)
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("bf16", "fp8", "bf16"),
        ("fp16", "fp8", "fp16"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("enable_sink", [True, False])
def test_trtllm_batch_decode(
    kv_layout,
    batch_size,
    q_len_per_req,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
    enable_sink,
    kv_len,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    if o_dtype == "nvfp4" and q_len_per_req > 1:
        # todo(Yingyi): add support for nvfp4 with speculative decoding
        pytest.skip("nvfp4 is not supported for q_len_per_req > 1")

    # Set up test parameters
    torch.manual_seed(0)
    head_dim = 128
    MAX_IN_KV_LEN = kv_len

    # Generate random sequence lengths
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens_decode(
        batch_size, q_len_per_req, MAX_IN_KV_LEN
    )

    # Create query tensor and related data
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)

    # Create KV cache and related data
    kv_cache, k_scale, v_scale, ref_kv_cache = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(GPU_DEVICE)

    # Create output tensor and related data
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    can_infer_type = q.dtype == DTYPE_MAP[o_dtype] or create_out_tensor
    create_out_dtype = not can_infer_type or flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype, q_dtype
    )
    out, out_dtype, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor, create_out_dtype
    )

    sm_scale = float(1.0 / (head_dim**0.5))

    # # Build reference output
    # plan_params = {
    #     "indptr": kv_indptr,
    #     "indices": all_page_ids,
    #     "last_page_len": kv_last_page_len.to(GPU_DEVICE),
    #     "num_qo_heads": num_qo_heads,
    #     "num_kv_heads": num_kv_heads,
    #     "head_dim": head_dim,
    #     "page_size": page_size,
    #     "pos_encoding_mode": "NONE",
    #     "kv_data_type": ref_kv_cache.dtype,
    #     "q_data_type": ref_q.dtype,
    #     "window_left": window_left,
    # }
    # if not enable_sink:
    #     if q_len_per_req == 1:
    #         wrapper_ref = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
    #             workspace_buffer_ref, kv_layout, use_tensor_cores=True
    #         )
    #         wrapper_ref.plan(**plan_params)
    #         output_ref = wrapper_ref.run(ref_q, ref_kv_cache)

    #     else:
    #         # speculative decoding test
    #         wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
    #             workspace_buffer_ref, kv_layout
    #         )
    #         plan_params_prefill = plan_params.copy()
    #         plan_params_prefill.update(
    #             {
    #                 "qo_indptr": q_indptr,
    #                 "paged_kv_indptr": plan_params_prefill.pop("indptr"),
    #                 "paged_kv_indices": plan_params_prefill.pop("indices"),
    #                 "paged_kv_last_page_len": plan_params_prefill.pop("last_page_len"),
    #                 "head_dim_qk": plan_params_prefill.pop("head_dim"),
    #                 "causal": True,
    #                 "logits_soft_cap": 0.0,
    #             }
    #         )
    #         wrapper_ref.plan(**plan_params_prefill)
    #         output_ref = wrapper_ref.run(ref_q, ref_kv_cache)
    # else:
    #     # Construct flat K/V via helper
    #     k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
    #         ref_kv_cache,
    #         page_table,
    #         seq_lens.to(GPU_DEVICE),
    #         page_size,
    #         kv_last_page_len,
    #     )
    #     sink = torch.rand(num_qo_heads, device=GPU_DEVICE, dtype=torch.float32) * 5
    #     output_ref = sink_attention_unified(
    #         ref_q,
    #         k_flat,
    #         v_flat,
    #         sink,
    #         window_left,
    #         True,
    #         sm_scale,
    #         mode="varlen",
    #         batch_size=batch_size,
    #         qo_indptr=q_indptr,
    #         kv_indptr=kv_indptr_tokens,
    #     )

    # Run trtllm-gen function call
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens.to(GPU_DEVICE),
        torch.max(seq_lens).item(),
        q_scale * k_scale * sm_scale,  # bmm1_scale
        v_scale / o_scale,  # bmm2_scale
        window_left,  # window_left
        out=out,
        out_dtype=out_dtype,
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        enable_pdl=enable_pdl,
        sinks=(sink if enable_sink else None),
        q_len_per_req=q_len_per_req,
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()

    def fn():
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q.contiguous(),
            kv_cache,
            workspace_buffer,
            page_table,
            seq_lens.to(GPU_DEVICE),
            torch.max(seq_lens).item(),
            q_scale * k_scale * sm_scale,  # bmm1_scale
            v_scale / o_scale,  # bmm2_scale
            window_left,  # window_left
            out=out,
            out_dtype=out_dtype,
            o_sf_scale=o_sf_scale,
            o_sf_vec_size=o_sf_vec_size,
            enable_pdl=enable_pdl,
            sinks=(sink if enable_sink else None),
            q_len_per_req=q_len_per_req,
        )
    latency = benchmark_fn(fn, f"trtllm_batch_decode_with_kv_cache: batch: {batch_size}, kv_len: {kv_len}, q_dtype: {q_dtype}, kv_dtype: {kv_dtype}, o_dtype: {o_dtype}")
    return latency

    # if o_dtype == "nvfp4":
    #     output, output_ref = unpack_compare_nvfp4(
    #         output, output_ref, o_sf_scale, o_sf_vec_size
    #     )
    #     assert o_scale == 1.0
    #     rtol, atol = 3e-1, 1e0
    # elif q_dtype == "fp8" and o_dtype == "fp8":
    #     rtol, atol = 5e-2, 7e-2
    # elif q_dtype == "fp8" and o_dtype in ["bf16", "fp16"]:
    #     rtol, atol = 4e-2, 7e-2
    # else:
    #     rtol, atol = 1e-2, 1e-2

    # # convert to float32 for fp8 is not supported by assert_close
    # # relax rtol and atol for speculative decoding test
    # if q_len_per_req > 1:
    #     rtol, atol = rtol * 2, atol * 2

    # # Arbitary small mismatch rate
    # allowed_mismatch_rate = 5e-5
    # # Calculate max allowed mismatched elements based on tensor size
    # total_elements = (output.float() * o_scale).numel()
    # max_mismatched_elements = int(allowed_mismatch_rate * total_elements)

    # assert_close_with_mismatch_tolerance(
    #     output.float() * o_scale,
    #     output_ref.float(),
    #     rtol=rtol,
    #     atol=atol,
    #     max_mismatched_elements=max_mismatched_elements,
    # )

    # if o_dtype != "nvfp4":  # wrapper api does not support fp4 output yet.
    #     # test wrapper with trtllm-gen backend
    #     wrapper_trtllm_gen = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
    #         workspace_buffer, kv_layout, backend="trtllm-gen"
    #     )
    #     plan_params["q_data_type"] = q.dtype
    #     plan_params["kv_data_type"] = kv_cache.dtype
    #     wrapper_trtllm_gen.plan(**plan_params)
    #     output_wrapper = wrapper_trtllm_gen.run(
    #         q.contiguous(),
    #         kv_cache,
    #         q_scale=q_scale,
    #         k_scale=k_scale,
    #         v_scale=v_scale / o_scale,
    #         enable_pdl=enable_pdl,
    #         sinks=(sink if enable_sink else None),
    #         q_len_per_req=q_len_per_req,
    #     )
    #     # v_scale, o_scale in wrapper is emulated by multiplying output by v_scale instead of fused into kernel.
    #     if v_scale == o_scale == 1.0:
    #         assert (output_wrapper == output).all()
    #     else:
    #         # todo(Yingyi): fix precision issue with this test
    #         if not (
    #             q_dtype == "fp8"
    #             and kv_dtype == "fp8"
    #             and o_dtype == "fp8"
    #             and batch_size == 256
    #             and q_len_per_req == 3
    #             and page_size == 64
    #             and num_kv_heads == 4
    #             and head_grp_size == 5
    #         ):
    #             torch.testing.assert_close(
    #                 output.float(),
    #                 output_wrapper.float(),
    #                 rtol=1e-1,
    #                 atol=1e-1,
    #             )
    #         else:
    #             assert_close_with_mismatch_tolerance(
    #                 output.float(),
    #                 output_wrapper.float(),
    #                 rtol=1e-1,
    #                 atol=1e-1,
    #                 max_mismatched_elements=5,
    #             )
    #     # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    #     # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    #     assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("s_qo", [32, 64, 87])
@pytest.mark.parametrize("s_kv", [32, 64, 87])
@pytest.mark.parametrize("num_kv_heads", [16, 32])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_gen_prefill_deepseek(
    batch_size, s_qo, s_kv, num_kv_heads, head_grp_size, causal
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] in [11, 12]:
        pytest.skip("trtllm-gen does not support SM110/SM120/SM121 GPUs.")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    num_qo_heads = num_kv_heads * head_grp_size
    head_dim_qk = 192
    head_dim_vo = 128

    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    cumsum_s_kv = torch.sum(actual_seq_lens_kv)

    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )

    k_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_qk),
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        (cumsum_s_kv, num_kv_heads, head_dim_vo),
        device=device,
        dtype=torch.bfloat16,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim_qk**0.5))

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(device)

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # kv_indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * s_kv

    # Create kv_indptr as cumulative sum of actual_seq_lens_kv
    kv_indptr = torch.cat(
        [
            torch.tensor(
                [0],
                device=device,
            ),
            torch.cumsum(actual_seq_lens_kv.view(-1), dim=0),
        ]
    ).int()

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer_ref,
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    output_ref, lse_ref = wrapper.run(q, k_cache, v_cache, return_lse=True)
    output = torch.empty_like(output_ref)

    bmm1_scale = scale
    bmm2_scale = 1.0
    output_trtllm, lse_trtllm = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv,
        s_qo,
        s_kv,
        bmm1_scale,
        bmm2_scale,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        out=output,
    )
    torch.testing.assert_close(
        output_trtllm,
        output_ref,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        lse_trtllm,
        lse_ref,
        atol=1e-3,
        rtol=1e-3,
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()


def get_prefill_params():
    kv_layout = ["HND"]
    batch_size = [1, 4, 16, 64, 128]
    page_size = 16,
    num_kv_heads = 128,
    head_grp_size = 1,
    window_left = -1,
    q_o_kv_dtype = [["bf16", "bf16", "bf16"], ["fp8", "fp8", "fp8"], ["fp8", "nvfp4", "fp8"]]
    enable_pdl = [False]
    enable_sink = [False]
    seq_len = [1024, 4*1024, 16*1024, 64*1024, 256*1024]
    params = []
    for kv, bs, ps, nkh, hgs, wl, dtype, pdl, sink, s in itertools.product(kv_layout, batch_size, page_size, num_kv_heads, head_grp_size, window_left, q_o_kv_dtype, enable_pdl, enable_sink, seq_len):
        params.append((kv, bs, ps, nkh, hgs, wl, *dtype, pdl, sink, s))
    return params


def benchmark_trtllm_batch_context_with_kv_cache():
    params = get_prefill_params()

    # 创建数据收集列表
    test_results = []
    
    for kv, bs, ps, nkh, hgs, wl, q_dtype, o_dtype, kv_dtype, pdl, sink, seq_len in params:
        # 准备测试参数字典
        test_params = {
            'kv_layout': kv,
            'batch_size': bs,
            'page_size': ps,
            'num_kv_heads': nkh,
            'head_grp_size': hgs,
            'window_left': wl,
            'q_dtype': q_dtype,
            'o_dtype': o_dtype,
            'kv_dtype': kv_dtype,
            'enable_pdl': pdl,
            'enable_sink': sink,
            'seq_len': seq_len
        }
        
        try:
            latency = test_trtllm_batch_prefill(kv, bs, ps, nkh, hgs, wl, q_dtype, o_dtype, kv_dtype, pdl, sink, seq_len)
            
            # 使用辅助函数收集成功的测试结果
            result_data = collect_test_result(test_params, latency=latency)
            test_results.append(result_data)
            
            print(f"测试完成 - batch_size: {bs}, seq_len: {seq_len}, latency: {latency:.3f} ms")
            
        except Exception as e:
            error_msg = str(e)
            print(f"测试失败 - batch_size: {bs}, seq_len: {seq_len}, error: {error_msg}")
            
            # 使用辅助函数收集失败的测试结果
            result_data = collect_test_result(test_params, error=error_msg)
            test_results.append(result_data)
        
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 使用通用函数保存结果到Excel文件
    save_test_results_to_excel(test_results, "test_trtllm_batch_prefill", show_stats=True)


def get_decode_params():
    kv_layout = ["HND"]
    batch_size = [1, 4, 16, 64, 128]
    q_len_per_req = 1,
    page_size = 16,
    num_kv_heads = 128,
    head_grp_size = 1,
    window_left = -1,
    q_o_kv_dtype = [["bf16", "bf16", "bf16"], ["fp8", "fp8", "fp8"], ["fp8", "nvfp4", "fp8"]]
    enable_pdl = [False]
    enable_sink = [False]
    kv_len = [1024, 4*1024, 16*1024, 64*1024, 256*1024]
    params = []
    for kv, bs, q_len_per_req, ps, nkh, hgs, wl, dtype, pdl, sink, kv_seq_len in itertools.product(kv_layout, batch_size, q_len_per_req, page_size, num_kv_heads, head_grp_size, window_left, q_o_kv_dtype, enable_pdl, enable_sink, kv_len):
        params.append((kv, bs, q_len_per_req, ps, nkh, hgs, wl, *dtype, pdl, sink, kv_seq_len))
    return params

def benchmark_trtllm_batch_decode_with_kv_cache():
    params = get_decode_params()

    # test_params = {'kv_layout': 'HND', 'batch_size': 4, 'q_len_per_req': 1, 'page_size': 16, 'num_kv_heads': 128, 'head_grp_size': 1, 'window_left': -1, 'q_dtype': 'bf16', 'o_dtype': 'bf16', 'kv_dtype': 'bf16', 'enable_pdl': False, 'enable_sink': False, 'kv_len': 1024}
    # test_params['kv_len'] = 1024 # 1024*1024
    # print(test_params)
    # test_trtllm_batch_decode(**test_params)
    # exit(0)

    # 创建数据收集列表
    test_results = []
    
    for kv, bs, q_len_per_req, ps, nkh, hgs, wl, q_dtype, o_dtype, kv_dtype, pdl, sink, kv_len in params:
        # 准备测试参数字典
        test_params = {
            'kv_layout': kv,
            'batch_size': bs,
            'q_len_per_req': q_len_per_req,
            'page_size': ps,
            'num_kv_heads': nkh,
            'head_grp_size': hgs,
            'window_left': wl,
            'q_dtype': q_dtype,
            'o_dtype': o_dtype,
            'kv_dtype': kv_dtype,
            'enable_pdl': pdl,
            'enable_sink': sink,
            'kv_len': kv_len
        }

        try:
            latency = test_trtllm_batch_decode(**test_params)
            
            # 使用辅助函数收集成功的测试结果
            result_data = collect_test_result(test_params, latency=latency)
            test_results.append(result_data)
            
            print(f"测试完成 - batch_size: {bs}, kv_len: {kv_len}, latency: {latency:.3f} ms")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            error_msg = str(e)
            print(f"测试失败 - batch_size: {bs}, kv_len: {kv_len}, error: {error_msg}")
            
            # 使用辅助函数收集失败的测试结果
            result_data = collect_test_result(test_params, error=error_msg)
            test_results.append(result_data)

    
    # 使用通用函数保存结果到Excel文件
    save_test_results_to_excel(test_results, "test_trtllm_batch_decode", show_stats=True)


if __name__ == "__main__":
    benchmark_trtllm_batch_context_with_kv_cache()
    benchmark_trtllm_batch_decode_with_kv_cache()
    
