import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation

from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock
import os
from .ant_quant import ant_quantization, ant_quantization_search, meta_flint_set
from ..utils.make_distribution import group_dist_outlier, group_dist, make_heat_map, outlier_ratio_stat, outlier_count
import math

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import functools
from collections import defaultdict
import numpy as np
import json

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]



def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif 'mptblock' in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif 'falcon' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'bigcode' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'neox' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
        
        
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        # assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
        
@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            # m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            # m.cpu()


@torch.no_grad()
def make_quant_linear(
    model, w_bit, q_config, ant_config=None, outlier_config=None, quant_mode_config=None, mxfp_config=None,
    init_only=False
):
    from .pre_quant import get_blocks, get_named_linears
    # assert q_config["zero_point"], "We only support zero_point quantization now."
    
    layers = get_blocks(model)

    for i in tqdm(range(len(layers)), desc=" make quant linear..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)
        for name, module in named_linears.items():
            # module.cuda()
            if quant_mode_config['quant_method'] == 'mxfp':
                from .qmodule_mxfp import MXFP_Linear
                q_linear = MXFP_Linear.from_linear(
                    module, w_bit, 32, i, name, init_only=False, ant_config=ant_config, quant_mode=mxfp_config["mx_type"], mxfp_config=mxfp_config)
            else:
                pass
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
            # module.cpu()


    torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def save_x_clip_cw(
    model, file_name
):
    from .pre_quant import get_blocks, get_named_linears
    layers = get_blocks(model)
    x_clip_cw_dict = {}
    for i in tqdm(range(len(layers)), desc=" save the x_clip_cw ..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        for name, module in named_linears.items():
            x_clip_cw_dict[f"layer_{i}_{name}"] = module.x_clip_cw

    with open(file_name, 'w') as f:
        json.dump(x_clip_cw_dict, f)

@torch.no_grad()
def load_x_clip_cw(model, file_name):
    from .pre_quant import get_blocks, get_named_linears
    # 读取保存的 x_clip_cw 数据
    with open(file_name, 'r') as f:
        x_clip_cw_dict = json.load(f)
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="Loading x_clip_cw into model..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        for name, module in named_linears.items():
            key = f"layer_{i}_{name}"
            if key in x_clip_cw_dict:
                module.x_clip_cw = x_clip_cw_dict[key]
                module.weight_quant_grid = True
            else:
                print(f"Warning: {key} not found in the JSON file.")