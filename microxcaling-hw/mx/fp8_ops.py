"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch

from typing import Optional, Tuple
from .intel_ops import quantize_fp4
from mx.mx_ops import quantize_mx_op

def fake_quantize_fp8(tensor,
                      scale,
                      zero_point,
                      fp8_format):
    dtype = tensor.dtype
    scale = scale.to(tensor.device)
    # scale = scale.to(torch.bfloat16)
    # print("########", scale.dtype)
    zero_point = zero_point.to(tensor.device)
    # scale = scale.view(-1, 1)
    # zero_point = zero_point.view(-1, 1)
    finfo = torch.finfo(fp8_format)

    tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max)
    tensor = tensor.to(fp8_format)
    # tensor = tensor.to(dtype) * scale #FP8
    tensor = tensor.to(dtype)
    return tensor, scale

def init_weight_quant_fp8(
        weight,
        fp8_format,
        mm_tensor = False
):
    finfo = torch.finfo(fp8_format)
    if mm_tensor:
        max_, min_ = weight, weight
        max_shape = (-1, )

        for i in range(len(weight.shape) - 1):
            max_ = max_.max(1)[0]
            min_ = min_.min(1)[0]
            max_shape += (1, )

        max_val = max_.reshape(max_shape)
        min_val = min_.reshape(max_shape)
    else:
        min_val, max_val = weight.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / finfo.max
    scale = scale.to(weight.dtype)
    zero_point = torch.tensor(float(0.0)).type_as(min_val).expand(scale.shape)
    return scale, zero_point

def per_channel_quantize(tensor, fp8_format) -> torch.Tensor:
    scale, zero_point = init_weight_quant_fp8(tensor, fp8_format, True)
    dq_weight, scale = fake_quantize_fp8(tensor, scale, zero_point, fp8_format)
    return dq_weight*scale

def fake_quantize_fp8_16(tensor,
                      scale,
                      zero_point,
                      fp8_format):
    dtype = tensor.dtype
    scale = scale.to(tensor.device)
    zero_point = zero_point.to(tensor.device)
    # scale = scale.view(-1, 1)
    # zero_point = zero_point.view(-1, 1)
    finfo = torch.finfo(fp8_format)

    tensor = (tensor / scale)
    # tensor = tensor.to(dtype) * scale #FP8
    tensor = tensor.to(dtype)
    return tensor, scale

def init_weight_quant_fp8_16(
        weight,
        fp8_format,
        mm_tensor = False
):
    finfo = torch.finfo(fp8_format)
    if mm_tensor:
        max_, min_ = weight, weight
        max_shape = (-1, )

        for i in range(len(weight.shape) - 1):
            max_ = max_.max(1)[0]
            min_ = min_.min(1)[0]
            max_shape += (1, )

        max_val = max_.reshape(max_shape)
        min_val = min_.reshape(max_shape)
    else:
        min_val, max_val = weight.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax / finfo.max
    scale = scale.to(weight.dtype)
    zero_point = torch.tensor(float(0.0)).type_as(min_val).expand(scale.shape)
    return scale, zero_point

def per_channel_quantize_16(tensor, fp8_format) -> torch.Tensor:
    scale, zero_point = init_weight_quant_fp8_16(tensor, fp8_format, True)
    dq_weight, scale = fake_quantize_fp8_16(tensor, scale, zero_point, fp8_format)
    return dq_weight, scale


def linear_quantization_params_fp8(
        min_val,
        max_val,
        fp8_format,
):
    finfo = torch.finfo(fp8_format)
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / finfo.max
    scale = scale.to(min_val.dtype)
    zero_point = torch.tensor(float(0.0)).type_as(min_val).expand(scale.shape)
    return scale, zero_point

def per_token_quantize(tensor: torch.Tensor, fp8_format) -> torch.Tensor:
    x_min = tensor.min(-1, keepdim=True)[0]  # keepdim=True 保持维度对齐
    x_max = tensor.max(-1, keepdim=True)[0]
    # x_min = x_min.view(-1,x_min.shape[1],1)
    # x_max = x_max.view(-1,x_max.shape[1],1)
    input_scale, input_offset = linear_quantization_params_fp8(x_min, x_max, fp8_format)
    dq_tensor, scale = fake_quantize_fp8(tensor, input_scale, input_offset, fp8_format)
    return dq_tensor*scale

def linear_quantization_params_fp8_16(
        min_val,
        max_val,
        fp8_format,
):
    finfo = torch.finfo(fp8_format)
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax / finfo.max
    scale = scale.to(min_val.dtype)
    zero_point = torch.tensor(float(0.0)).type_as(min_val).expand(scale.shape)
    return scale, zero_point

def per_token_quantize_16(tensor: torch.Tensor, fp8_format) -> torch.Tensor:
    x_min = tensor.min(2)[0]
    x_max = tensor.max(2)[0]
    x_min = x_min.view(-1,x_min.shape[1],1)
    x_max = x_max.view(-1,x_max.shape[1],1)
    input_scale, input_offset = linear_quantization_params_fp8_16(x_min, x_max, fp8_format)
    dq_tensor, scale = fake_quantize_fp8_16(tensor, input_scale, input_offset, fp8_format)
    return dq_tensor, scale


def per_tensor_quantize(tensor: torch.Tensor, fp8_format) -> torch.Tensor:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
        fp8_format: torch.float8_e5m2; torch.float8_e4m3fn
    """
    finfo = torch.finfo(fp8_format)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(fp8_format)
    scale = scale.float().reciprocal()
    dqweight = qweight.to(torch.float16) * scale
    # qweight = qweight.to(torch.float16)
    return dqweight
    return qweight, scale

# def quantize_fp8(tensor: torch.Tensor, method, fp8_format=torch.float8_e4m3fn) -> torch.Tensor:
def quantize_fp8(tensor: torch.Tensor, method, fp8_format=torch.float8_e4m3fn):
    if method == "per-token":
        return per_token_quantize(tensor, fp8_format)
    elif method == "per-tensor":
        return per_tensor_quantize(tensor, fp8_format)
    elif method == "per-channel":
        return per_channel_quantize(tensor, fp8_format)
    else:
        print(" [quantize_fp8] Error: Check Activation Method Input")


def quantize_fp8_16(tensor: torch.Tensor, fp8_format) -> torch.Tensor:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
        fp8_format: torch.float8_e5m2; torch.float8_e4m3fn
    """
    finfo = torch.finfo(fp8_format)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    # qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    qweight = (tensor * scale)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    # qweight = qweight.to(fp8_format)
    scale = scale.float().reciprocal()
    # dqweight = qweight.to(torch.float16) * scale
    dqweight = qweight.to(torch.float16)
    return dqweight, scale


def quantize_fp4_(tensor: torch.Tensor, group_size, first_method) -> torch.Tensor:
    first, scale = quantize_fp8(tensor, first_method, torch.float8_e5m2)
    second = quantize_fp4(first, num_bits=4, group_size=group_size, scheme="asym", quantile=1.0, data_type="fp4_e2m1", return_int=False, full_range=False)
    return second*scale

def quantize_fp4_vanila(tensor: torch.Tensor, group_size) -> torch.Tensor:
    first, scale = quantize_fp8_16(tensor, torch.float8_e4m3fn)
    second = quantize_fp4(first, num_bits=4, group_size=group_size, scheme="asym", quantile=1.0, data_type="fp4_e2m1", return_int=False, full_range=False)
    return second*scale

def quantize_fp4_mocha(tensor: torch.Tensor, group_size) -> torch.Tensor:
    first, scale = per_channel_quantize_16(tensor, torch.float8_e4m3fn)
    second = quantize_fp4(first, num_bits=4, group_size=group_size, scheme="asym", quantile=1.0, data_type="fp4_e2m1", return_int=False, full_range=False)
    return second*scale

def quantize_mxfp4_(tensor: torch.Tensor, first_method, mx_specs) -> torch.Tensor:
    first, scale = quantize_fp8(tensor, first_method, torch.float8_e4m3fn)
    # second = quantize_fp4(first, num_bits=4, group_size=group_size, scheme="asym", quantile=1.0, data_type="fp4_e2m1", return_int=False, full_range=False)
    second = quantize_mx_op(first.to(torch.float32), mx_specs=mx_specs, elem_format='fp4_e2m1', axes=[-1], round='nearest', expand_and_reshape=False).to(torch.float16)
    return second*scale

def quantize_mxfp4_vanila(tensor: torch.Tensor, mx_specs) -> torch.Tensor:
    first, scale = quantize_fp8_16(tensor, torch.float8_e4m3fn)
    # second = quantize_fp4(first, num_bits=4, group_size=group_size, scheme="asym", quantile=1.0, data_type="fp4_e2m1", return_int=False, full_range=False)
    second = quantize_mx_op(first.to(torch.float32), mx_specs=mx_specs, elem_format='fp4_e2m1', axes=[-1], round='nearest', expand_and_reshape=False).to(torch.float16)
    return second*scale

def quantize_mxfp4_mocha(tensor: torch.Tensor, mx_specs) -> torch.Tensor: # default per-channel
    first, scale = per_channel_quantize_16(tensor, torch.float8_e4m3fn)
    # second = quantize_fp4(first, num_bits=4, group_size=group_size, scheme="asym", quantile=1.0, data_type="fp4_e2m1", return_int=False, full_range=False)
    second = quantize_mx_op(first.to(torch.float32), mx_specs=mx_specs, elem_format='fp4_e2m1', axes=[-1], round='nearest', expand_and_reshape=False).to(torch.float16)
    return second*scale

def quantize_mxfp4_mocha_(tensor: torch.Tensor, mx_specs) -> torch.Tensor: # default per-channel
    first, scale = per_token_quantize_16(tensor, torch.float8_e4m3fn)
    # second = quantize_fp4(first, num_bits=4, group_size=group_size, scheme="asym", quantile=1.0, data_type="fp4_e2m1", return_int=False, full_range=False)
    second = quantize_mx_op(first.to(torch.float32), mx_specs=mx_specs, elem_format='fp4_e2m1', axes=[-1], round='nearest', expand_and_reshape=False).to(torch.float16)
    return second*scale