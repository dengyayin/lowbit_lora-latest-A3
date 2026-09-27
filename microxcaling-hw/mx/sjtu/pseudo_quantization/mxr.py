from awq.quantize.qmodule_mxfp import get_quant_weight_mxfp, get_quant_act_mxfp
import torch
import pdb
from mx.mx_ops import quantize_mx_op, quantize_mx_op_scale

torch.manual_seed(42)

def find_nan_positions(tensor):
    nan_indices = torch.where(torch.isnan(tensor))
    num_nans = len(nan_indices[0]) if nan_indices else 0

    if num_nans == 0:
        print("✅ Tensor 中没有 NaN 值。")
        return

    print(f"⚠️ 发现 {num_nans} 个 NaN 值，位置如下：")
    for i in range(num_nans):
        # 获取每个维度上的索引
        idx = [dim[i].item() for dim in nan_indices]
        val = tensor[idx].item()
        print(f"  - 位置 {idx} → 值 = NaN")


mx_specs = {
    'block_size': 32,
    'fp': 16,
    'bfloat': -1,
    'w_elem_format': 'fp4_e1m2',
    'a_elem_format': 'fp4_e1m2', 
    'custom_cuda': False,
    # For quantization-aware finetuning, do backward pass in FP32
    'scale_bits': 8,
    'shared_exp_method': 'max',
    'quantize_backprop': False,
    "mx_flush_fp32_subnorms": True,
    'bfloat_subnorms': True,
    "round_output": "nearest",    
    "round_mx_output": "nearest",        
    "round_weight": "nearest",
    "round_output": "nearest",     
}

def split_last_dim_into_blocks(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    *dims, last_dim = x.shape  # 支持任意前面的维度，如 B, S, ...
    assert last_dim % block_size == 0, (
        f"最后一维大小 {last_dim} 必须能被 block_size={block_size} 整除"
    )
    num_blocks = last_dim // block_size
    return x.view(*dims, num_blocks, block_size)


quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

quant_grid = torch.tensor(quant_grid).to(dtype=torch.half)


M, K = 128, 128
input_tensor = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
inp_tensor = input_tensor.clone()

ori_shape = input_tensor.shape
# M,K => M,K//32,32
input_tensor = split_last_dim_into_blocks(input_tensor)

qinput, quant_mse_sum, _, scales = get_quant_act_mxfp(input_tensor, 'fp8', x_clip_r=0.99, round_method="up")
# qinput = qinput.reshape(ori_shape)
dqinput = qinput*scales
dqinput = dqinput.reshape(ori_shape)

dqa = quantize_mx_op(inp_tensor.to(torch.float32), mx_specs=mx_specs, elem_format='fp8_e4m3', axes=[-1], round='nearest', expand_and_reshape=False)
qa, scale = quantize_mx_op_scale(inp_tensor.to(torch.float32), mx_specs=mx_specs, elem_format='fp8_e4m3', axes=[-1], round='nearest', expand_and_reshape=False)


pdb.set_trace()

