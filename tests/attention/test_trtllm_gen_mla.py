import gc
import math

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

import numpy as np
from flashinfer import testing
import pandas as pd

def benchmark_fn(fn, docstring: str):
    measured_times = testing.bench_gpu_time_with_cuda_event(fn)
    ms = np.average(measured_times)
    print("\n \033[1;32m" + docstring + f" Benchmark Time: {ms:.3f} ms" + "\033[0m")
    return ms


global_workspace_buffer = None  # can.be empty initialized
global_trtllm_gen_fmha_workspace_buffer = None  # must be zero initialized
workspace_size = 128 * 1024 * 1024


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize(
    "q_len_per_request", [1, 2]
)  # todo(Yingyi): verify larger q_len_per_request
@pytest.mark.parametrize("dynamic_scale", [False])
@pytest.mark.parametrize("enable_pdl", [True, False, None])
def test_trtllm_batch_decode_mla(
    batch_size: int,
    scale: float,
    dtype: torch.dtype,
    page_size: int,
    q_len_per_request: int,
    dynamic_scale: bool,
    enable_pdl: bool,
    seq_len: int,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] in [11, 12]:
        pytest.skip("trtllm-gen does not support SM110/SM120/SM121 GPUs.")
    if dynamic_scale and dtype != torch.float8_e4m3fn:
        pytest.skip("Dynamic scale is not supported for non-fp8 dtype")

    torch.manual_seed(42)
    device = "cuda:0"

    # Fixed max sequence length
    MAX_SEQ_LEN = seq_len

    # Deepseek attention config (decode-MLA)
    num_q_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512

    # Initialize tensors
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
    ).to(dtype)

    num_tokens = MAX_SEQ_LEN * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size

    # Sequence lengths and block tables
    # seq_lens = [torch.randint(1, MAX_SEQ_LEN, (1,)).item() for _ in range(batch_size)]
    # seq_lens[-1] = MAX_SEQ_LEN
    seq_lens = [MAX_SEQ_LEN] * batch_size
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()

    # Generate random but unique block IDs for all sequences
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = torch.randperm(
        total_blocks_needed, device=device
    )  # Random permutation

    # Generate unique block IDs for all sequences
    block_id = 0
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int, device=device
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # Create interleaved KV cache
    # Allocate more than needed blocks, block_id is just enough, to mimick real-world cases
    kv_cache = torch.randn(
        size=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim), device=device
    ).to(dtype)
    # (num_blocks, 1, page_size, kv_lora_rank + qk_rope_head_dim)

    # Allocate workspace buffer
    # todo(Yingyi): calculate the actual size of workspace buffer
    global global_workspace_buffer, global_trtllm_gen_fmha_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_trtllm_gen_fmha_workspace_buffer is None:
        global_trtllm_gen_fmha_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    workspace_buffer = global_trtllm_gen_fmha_workspace_buffer
    workspace_buffer_ref = global_workspace_buffer

    bmm1_log2_scale_tensor = (
        torch.tensor(
            [scale / ((128 + 64) ** 0.5 * math.log2(math.e))],
            dtype=torch.float32,
            device=device,
        )
        if dynamic_scale
        else None
    )
    bmm2_scale_tensor = (
        torch.tensor([1.0], dtype=torch.float32, device=device)
        if dynamic_scale
        else None
    )

    # Run decode-MLA
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale / ((128 + 64) ** 0.5),
        bmm2_scale=1.0,
        bmm1_scale_log2_tensor=bmm1_log2_scale_tensor,
        bmm2_scale_tensor=bmm2_scale_tensor,
        enable_pdl=enable_pdl,
    )
    # check if the first 8192 * 256 * 4 bytes of workspace_buffer is zero
    # note(Yingyi): the first 8192 * 256 * 4 bytes of workspace_buffer is the counter workspace, size might change in the future
    assert (workspace_buffer[: 8192 * 256 * 4].cpu().numpy() == 0).all()

    def fn():
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=max_seq_len,
            bmm1_scale=scale / ((128 + 64) ** 0.5),
            bmm2_scale=1.0,
            bmm1_scale_log2_tensor=bmm1_log2_scale_tensor,
            bmm2_scale_tensor=bmm2_scale_tensor,
            enable_pdl=enable_pdl,
        )
    latency = benchmark_fn(fn, f"trtllm_batch_decode_with_kv_cache_mla: batch: {batch_size}, seq_len: {seq_len}, q_dtype: {dtype}, kv_dtype: {dtype}, o_dtype: {dtype}")
    return latency

    # # Run reference attention and align output
    # sm_scale = scale / (
    #     (128 + 64) ** 0.5
    # )  # use head dimension before matrix absorption
    # wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    #     workspace_buffer_ref,
    #     backend="fa2",
    # )

    # if dtype == torch.float8_e4m3fn:
    #     # convert query and kv_cache to bfloat16
    #     query = query.to(torch.bfloat16)
    #     kv_cache = kv_cache.to(torch.bfloat16)

    # q_indptr = (
    #     torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
    #     * q_len_per_request
    # )
    # kv_indptr = torch.zeros_like(q_indptr)
    # kv_indptr[1:] = torch.cumsum(blocks_per_seq, dim=0)
    # kv_indices = all_block_ids.int()

    # wrapper.plan(
    #     q_indptr,
    #     kv_indptr,
    #     kv_indices,
    #     seq_lens_tensor,
    #     num_q_heads,
    #     kv_lora_rank,
    #     qk_rope_head_dim,
    #     page_size,
    #     True,
    #     sm_scale,
    #     query.dtype,
    #     kv_cache.dtype,
    # )
    # q_nope = query[..., :kv_lora_rank].view(
    #     batch_size * q_len_per_request, num_q_heads, kv_lora_rank
    # )
    # q_pe = query[..., kv_lora_rank:].view(
    #     batch_size * q_len_per_request, num_q_heads, qk_rope_head_dim
    # )

    # # todo: fix kv_cache
    # ckv = kv_cache[..., :kv_lora_rank]
    # kpe = kv_cache[..., kv_lora_rank:]

    # o_ref = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    # # check is nan
    # assert not torch.isnan(o_ref).any(), "o_ref is nan"
    # assert not torch.isnan(output).any(), "output is nan"

    # if dtype == torch.float8_e4m3fn:
    #     try:
    #         torch.testing.assert_close(
    #             output,
    #             o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
    #             rtol=1e-1,
    #             atol=1e-1,
    #         )  # todo: do reference with normal attention?
    #     except AssertionError as e:
    #         print("output:", output)
    #         print("o_ref:", o_ref)
    #         raise e
    # else:
    #     try:
    #         torch.testing.assert_close(
    #             output,
    #             o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
    #             rtol=1e-2,
    #             atol=1e-2,
    #         )
    #     except AssertionError as e:
    #         print("output:", output)
    #         print("o_ref:", o_ref)
    #         raise e


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


def get_mla_decode_params():
    """
    Generate parameter combinations for MLA decode benchmarking
    """
    import itertools
    
    batch_size = [1, 4, 16, 64, 128]
    scale = [1.0]
    dtype = [torch.float8_e4m3fn, torch.bfloat16]
    page_size = [32]
    q_len_per_request = [1]
    dynamic_scale = [False]  # True only supported for fp8
    enable_pdl = [False]
    seq_len = [1024, 4*1024, 16*1024, 64*1024, 256*1024]

    # Add dynamic_scale=True for fp8 dtype
    params = []
    for bs, s, dt, ps, q_len, dyn_scale, pdl, sl in itertools.product(
        batch_size, scale, dtype, page_size, q_len_per_request, dynamic_scale, enable_pdl, seq_len
    ):
        params.append((bs, s, dt, ps, q_len, dyn_scale, pdl, sl))
    
    return params


def benchmark_trtllm_batch_decode_mla():
    """
    Benchmark function for MLA decode that mimics benchmark_trtllm_batch_context_with_kv_cache
    """
    params = get_mla_decode_params()

    # 创建数据收集列表
    test_results = []
    
    for bs, scale, dtype, ps, q_len, dyn_scale, pdl, seq_len in params:
        # 准备测试参数字典
        test_params = {
            'batch_size': bs,
            'scale': scale,
            'dtype': str(dtype),
            'page_size': ps,
            'q_len_per_request': q_len,
            'dynamic_scale': dyn_scale,
            'enable_pdl': pdl,
            'seq_len': seq_len
        }
        
        try:
            latency = test_trtllm_batch_decode_mla(bs, scale, dtype, ps, q_len, dyn_scale, pdl, seq_len)
            
            # 使用辅助函数收集成功的测试结果
            result_data = collect_test_result(test_params, latency=latency)
            test_results.append(result_data)
            
            print(f"测试完成 - batch_size: {bs}, q_len: {q_len}, dtype: {dtype}, latency: {latency:.3f} ms")
            
        except Exception as e:
            error_msg = str(e)
            print(f"测试失败 - batch_size: {bs}, q_len: {q_len}, dtype: {dtype}, error: {error_msg}")
            
            # 使用辅助函数收集失败的测试结果
            result_data = collect_test_result(test_params, error=error_msg)
            test_results.append(result_data)
        
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 使用通用函数保存结果到Excel文件
    save_test_results_to_excel(test_results, "test_trtllm_batch_decode_mla", show_stats=True)


if __name__ == "__main__":
    # Run the benchmark when script is executed directly
    benchmark_trtllm_batch_decode_mla()
