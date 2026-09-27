
import math
from copy import deepcopy
from typing import Optional, OrderedDict, Union

import tqdm
import torch

# from .fp8_nvf4_kernel.quant_cy import QType, quant_dequant_float
from .hif4_gpu_e6m2_bf16.quant_cy import QType, quant_dequant_float, quant_func
# from .fp8_nvf4_kernel_deprecated.HiFX import to_HiFX_V12
from .mx_ops import quantize_nv4_op
# import pdb


def quantize_hif4_kernel(x_torch, N=4, axe=-1, force=False):
    # pytorch 实现
    # quant_type = QType(qtype_str).dim(axe)
    from .hif4_gpu_e6m2_bf16.quant_cy import QType, quant_dequant_float
    qtype_str = 'hifx%d'%(N)
    quant_type = QType(qtype_str).dim(axe)
    y2 = quant_dequant_float(x_torch, quant_type, force_py=True, force_fp32=False)
    return y2

def quantize_hif4_kernel_ste(x_torch, N=4, axe=-1, force=False):
    """hif4 量化的 STE (Straight-Through Estimator) 包装，用于训练时激活量化。

    前向：与 quantize_hif4_kernel 相同的量化逻辑
    反向：梯度直通（quant_func 内部 QuantFunc.apply 已实现 custom_bwd 直接返回 grad）
    """
    from .hif4_gpu_e6m2_bf16.quant_cy import QType, quant_func
    qtype_str = 'hifx%d'%(N)
    quant_type = QType(qtype_str).dim(axe)
    y2 = quant_func(x_torch, quant_type, force_py=True, force_fp32=False)
    return y2

def quantize_nvf4_scaled_kernel(x_torch, N=4, axe=-1, force=False):
    # pytorch 实现    
    qtype = QType('nvf4').dim(axe)
    y2 = quant_dequant_float(x_torch, qtype, force_py=force, force_fp32=False)
    return y2

from .fp8_nvf4_kernel_test.quant_cy import QType, quant_dequant_float_test

def quantize_nvf4_kernel(x_torch, N=4, axe=-1, force=False):
    # pytorch 实现    
    qtype = QType('nvf4').dim(axe)
    y2 = quant_dequant_float_test(x_torch, qtype, force_py=force, force_fp32=False)
    return y2

def quantize_nvf4_to_hif4_kernel(x_torch, N=4, axe=-1, force=False):
    y1 = quantize_nvf4_kernel(x_torch)
    y2 = quantize_hif4_kernel(y1)
    return y2

def quantize_nvf4s_to_hif4_kernel(x_torch, N=4, axe=-1, force=False):
    mx_specs = {
        'block_size': 32,
        'fp': 16,
        'bfloat': -1,
        'w_elem_format': 'fp4_e2m1',
        'a_elem_format': 'fp4_e2m1', 
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
    y1 = quantize_nv4_op(x_torch.to(torch.float32), mx_specs=mx_specs, elem_format='fp4_e2m1', axes=[-1], round='nearest', expand_and_reshape=False)
    y2 = quantize_hif4_kernel(y1)
    return y2

NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

FLOAT_MAPPING = {"nf4": NF4, "fp4": FP4_BNB, "fp4_e2m1_bnb": FP4_BNB, "fp4_e2m1": FP4_E2M1}
INT_MAPPING = {"nf4": NF4_BIT, "fp4": FP4_BNB_BIT, "fp4_e2m1_bnb": FP4_BNB_BIT, "fp4_e2m1": FP4_E2M1_BIT}

def quantize_afp4(tensor, quantile=1.0, data_type="fp4", return_int=False):
    """Quantize tensor to NF4/FP4 data type.
    """
    ori_shape = tensor.shape
    
    group_size = 32
    tensor = tensor.reshape(-1, group_size)
    
    assert data_type in FLOAT_MAPPING, "unexpected data type."
    allow_data = FLOAT_MAPPING[data_type]
    allow_data_bit = INT_MAPPING[data_type]
    # get scale and update tensor
    tmp = torch.zeros(tensor.shape[0], device=tensor.device)
    xmin = torch.minimum(tensor.min(1)[0], tmp)
    xmax = torch.maximum(tensor.max(1)[0], tmp)
    maxq = max(allow_data) - min(allow_data)
    scale_pos = (torch.abs(xmax) * quantile / (maxq / 2)).unsqueeze_(dim=-1)
    scale_neg = (torch.abs(xmin) * quantile / (maxq / 2)).unsqueeze_(dim=-1)

    x_pos = torch.zeros_like(tensor)
    x_neg = torch.zeros_like(tensor)
    x_pos = torch.where(tensor >= 0, tensor, x_pos)
    x_neg = torch.where(tensor < 0, tensor, x_neg)
    q_pos = x_pos / scale_pos
    q_neg = x_neg / scale_neg
    #tensor.div_(scale)
    mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
    q_pos_tensor = torch.zeros_like(q_pos)
    q_neg_tensor = torch.zeros_like(q_neg)
    for i in range(len(allow_data)):
        
        data = allow_data_bit[i] if return_int else allow_data[i]
        
        if i == 0:
            q_pos_tensor += torch.where(q_pos <= mid_data[i], data, 0)
            q_neg_tensor += torch.where(q_neg <= mid_data[i], data, 0)
        elif i == len(allow_data) - 1:
            q_pos_tensor += torch.where(q_pos > mid_data[i - 1], data, 0)
            q_neg_tensor += torch.where(q_neg > mid_data[i - 1], data, 0)
        else:
            q_pos_tensor += torch.where((mid_data[i - 1] < q_pos) & (q_pos <= mid_data[i]), data, 0)
            q_neg_tensor += torch.where((mid_data[i - 1] < q_neg) & (q_neg <= mid_data[i]), data, 0)

    xq = q_pos_tensor * scale_pos + q_neg_tensor * scale_neg

    xq = xq.reshape(ori_shape)
    
    return xq
    
def quantize_afp4_protect(tensor, quantile=1.0, data_type="fp4", return_int=False):
    """Quantize tensor to NF4/FP4 data type.
    """

    ori_shape = tensor.shape
    
    group_size = 32
    tensor = tensor.reshape(-1, group_size)
    
    assert data_type in FLOAT_MAPPING, "unexpected data type."
    allow_data = FLOAT_MAPPING[data_type]
    allow_data_bit = INT_MAPPING[data_type]
    
    # 找到每一行中绝对值最大的值及其索引
    max_abs_values, max_abs_indices = torch.max(tensor.abs(), dim=1)
    
    # 创建一个掩码，用于剔除每一行中绝对值最大的值
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask[torch.arange(tensor.shape[0]), max_abs_indices] = False
    
    # 剔除每一行中绝对值最大的值
    tensor_filtered = tensor[mask].reshape(tensor.shape[0], -1)
    
    # 继续原来的计算
    tmp = torch.zeros(tensor_filtered.shape[0], device=tensor.device)
    xmin = torch.minimum(tensor_filtered.min(1)[0], tmp)
    xmax = torch.maximum(tensor_filtered.max(1)[0], tmp)
    maxq = max(allow_data) - min(allow_data)
    scale_pos = (torch.abs(xmax) * quantile / (maxq / 2)).unsqueeze_(dim=-1)
    scale_neg = (torch.abs(xmin) * quantile / (maxq / 2)).unsqueeze_(dim=-1)
    
    x_pos = torch.zeros_like(tensor_filtered)
    x_neg = torch.zeros_like(tensor_filtered)
    x_pos = torch.where(tensor_filtered >= 0, tensor_filtered, x_pos)
    x_neg = torch.where(tensor_filtered < 0, tensor_filtered, x_neg)
    q_pos = x_pos / scale_pos
    q_neg = x_neg / scale_neg
    
    mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
    q_pos_tensor = torch.zeros_like(q_pos)
    q_neg_tensor = torch.zeros_like(q_neg)
    for i in range(len(allow_data)):
        
        data = allow_data_bit[i] if return_int else allow_data[i]
        
        if i == 0:
            q_pos_tensor += torch.where(q_pos <= mid_data[i], data, 0)
            q_neg_tensor += torch.where(q_neg <= mid_data[i], data, 0)
        elif i == len(allow_data) - 1:
            q_pos_tensor += torch.where(q_pos > mid_data[i - 1], data, 0)
            q_neg_tensor += torch.where(q_neg > mid_data[i - 1], data, 0)
        else:
            q_pos_tensor += torch.where((mid_data[i - 1] < q_pos) & (q_pos <= mid_data[i]), data, 0)
            q_neg_tensor += torch.where((mid_data[i - 1] < q_neg) & (q_neg <= mid_data[i]), data, 0)
    
    xq_filtered = q_pos_tensor * scale_pos + q_neg_tensor * scale_neg
    
    # 将剔除的绝对值最大值贴回原来的位置
    xq = torch.zeros_like(tensor)
    xq[mask] = xq_filtered.reshape(-1)
    xq[torch.arange(tensor.shape[0]), max_abs_indices] = tensor[torch.arange(tensor.shape[0]), max_abs_indices]
    
    xq = xq.reshape(ori_shape)
    
    return xq

def quantize_afp4a(tensor, quantile=1.0, data_type="fp4", return_int=False):
    """Quantize tensor to NF4/FP4 data type.

    Args:
        tensor: input tensor
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): data type. Defaults to 'nf4'.
        return_int (bool, optional): whether return int data. Defaults to False.

    Returns:
        q_tensor: fake quantized tensor
    """
    # import pdb;pdb.set_trace()
    group_size = 32
    ori_shape = tensor.shape

    tensor = tensor.reshape(-1, group_size)

    assert data_type in FLOAT_MAPPING, "unexpected data type."
    allow_data = FLOAT_MAPPING[data_type]
    allow_data_bit = INT_MAPPING[data_type]
    # get scale and update tensor
    
    tmp = torch.zeros(tensor.shape[0], device=tensor.device)
    xmin = torch.minimum(tensor.min(1)[0], tmp)
    xmax = torch.maximum(tensor.max(1)[0], tmp)
    maxq = max(allow_data) - min(allow_data)
    scale_pos = (torch.abs(xmax) * quantile / (maxq / 2)).unsqueeze_(dim=-1)
    scale_neg = (torch.abs(xmin) * quantile / (maxq / 2)).unsqueeze_(dim=-1)
    
    x_pos = torch.zeros_like(tensor)
    x_neg = torch.zeros_like(tensor)
    x_pos = torch.where(tensor >= 0, tensor, x_pos)
    x_neg = torch.where(tensor < 0, tensor, x_neg)
    q_pos = x_pos / scale_pos
    q_neg = x_neg / scale_neg
    mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
    q_pos_tensor = torch.zeros_like(q_pos)
    q_neg_tensor = torch.zeros_like(q_neg)
    for i in range(len(allow_data)):
        
        data = allow_data_bit[i] if return_int else allow_data[i]
        
        if i == 0:
            q_pos_tensor += torch.where(q_pos <= mid_data[i], data, 0)
            q_neg_tensor += torch.where(q_neg <= mid_data[i], data, 0)
        elif i == len(allow_data) - 1:
            q_pos_tensor += torch.where(q_pos > mid_data[i - 1], data, 0)
            q_neg_tensor += torch.where(q_neg > mid_data[i - 1], data, 0)
        else:
            q_pos_tensor += torch.where((mid_data[i - 1] < q_pos) & (q_pos <= mid_data[i]), data, 0)
            q_neg_tensor += torch.where((mid_data[i - 1] < q_neg) & (q_neg <= mid_data[i]), data, 0)
    xq = q_pos_tensor * scale_pos + q_neg_tensor * scale_neg
    xq = xq.reshape(ori_shape)

    return xq

import numpy as np

def To_HiFX(x, N = 4, G = 64):
    # {E8 + 8*E1 + 16*E1} + 1PN: Three-level scaling
    # EXP = [-127， 128]， column-wise convert
    x= np.array(x)
    Mi = x.shape[0]
    Ni =x.shape[1]
    Mcnt= np.ceil(Mi/G).astype(int)
    res = np.zeros((Mi, Ni))
    Ng = N-2
    for i in range(Mcnt):
        for j in range(Ni):
            ori = x[i*G:i*G+G,j]
            S= np.ones(G) # sign of grp values
            S[ori<0]= -1
            S = S.T
            tmpG = np.abs(ori) # abs of grp values
            EG =np.floor(np.log2(tmpG + 2**(-1000)))
            E16 = np.zeros(16)
            for k in range(16):
                E16[k]=np.max(EG[k*4:k*4+4])
            E8=np.zeros(8)
            for k in range(8): 
                E8[k]=np.max(E16[k*2:k*2+2])
            Emax = np.max(E8)
            # level-1 exp
            E8_1 = Emax - 2  #〔-127,126〕
            # level-2 exp
            E1_8= E8-E8_1-1 # <=1
            E1_8[E1_8<0]=0  # [0,1]
            # level-3 exp
            E1_8x2 = np.zeros(16)
            for k in range(8):
                E1_8x2[k*2:k*2+2]= E1_8[k]
            E1_16=E16- E1_8x2 -E8_1#<=1
            E1_16[E1_16<0]=0#[0,1]
            # Restore grp vals
            E16G = E1_16 + E1_8x2 + E8_1 #_Fused 16 exp vals
            EG = np.zeros(G)
            for k in range(16):
                EG[k*4:k*4+4]=E16G[k]
            in_grp = np.floor(tmpG*2**(-EG + Ng)+0.5)*2.0**(-Ng)
            in_grp[in_grp >= 2] = 2- 2**(-Ng)
            grp = S * in_grp * 2.0**EG
            res[i*G:i*G+G,j]= grp
    return res


def To_NVFP4(x, G=16):
    """
    PyTorch implementation of NVFP4 conversion (E4M3 + E2M1)
    Operates entirely on GPU if input is on GPU
    
    Args:
        x: Input tensor (PyTorch Tensor)
        G: Group size (default=16)
    
    Returns:
        Converted tensor with same device as input
    """
    Mi, Ni = x.shape[0], x.shape[1]
    Mcnt = (Mi + G - 1) // G  # Equivalent to ceil(Mi/G)
    res = torch.zeros_like(x)
    
    # Small constant to avoid log2(0)
    eps = torch.tensor(2**(-1000), device=x.device)
    
    for i in range(Mcnt):
        for j in range(Ni):
            # Process each group
            start = i * G
            end = min(start + G, Mi)
            ori = x[start:end, j]
            
            # Sign handling
            S = torch.where(ori < 0, -1.0, 1.0)
            tmp = torch.abs(ori)
            
            # Scale factor calculation
            SF = torch.max(tmp) / 6
            SF = torch.where(SF > 448, torch.tensor(448.0, device=x.device), SF)
            
            # E4M3 part
            E_SF = torch.floor(torch.log2(SF + eps))
            E_SF = torch.where(E_SF < -6, torch.tensor(-6.0, device=x.device), E_SF)
            E4M3 = torch.round(SF * 2**(-E_SF + 3)) * 2**(-3 + E_SF)
            
            # E2M1 part
            igv = tmp / E4M3
            E2 = torch.floor(torch.log2(igv + eps))
            E2 = torch.where(E2 < 0, torch.tensor(0.0, device=x.device), E2)
            
            # Round half to even
            M1 = torch.round(igv * 2**(-E2 + 1)) * 2**(-1)
            E2M1 = 2**E2 * M1
            E2M1 = torch.where(E2M1 > 6, torch.tensor(6.0, device=x.device), E2M1)
            
            # Combine results
            grp = S * E2M1 * E4M3
            res[start:end, j] = grp
            
    return res



