#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
import torch
import torch_npu

from . import print_sim_info_rank_0


def _npu_moe_token_unpermute_with_routing_map_mock(permuted_tokens, sorted_indices, restore_shape, *, probs=None,
                                                   routing_map=None, drop_and_pad=False):
    out_token = permuted_tokens.shape[0]
    hidden_size = permuted_tokens.shape[1]
    tokens_num = routing_map.shape[0]

    # unpermuted_tokens的shape：(tokens_num, hidden_size)
    unpermuted_tokens = torch.zeros([tokens_num, hidden_size], dtype=permuted_tokens.dtype,
                                    device=permuted_tokens.device)
    # out_index的shape：addedMode为false：(tokens_num * topK_num) paddedMode为true：(experts_num* capacity)
    out_index = torch.zeros([out_token], dtype=sorted_indices.dtype, device=sorted_indices.device)
    # permuted_token_id的shape： addedMode为false：(tokens_num * topK_num) paddedMode为true：(experts_num* capacity)
    permuted_token_id = torch.zeros([out_token], dtype=sorted_indices.dtype, device=sorted_indices.device)
    permute_probs = None
    # probsOptional
    if probs is not None:
        permute_probs = probs
    return unpermuted_tokens, out_index, permuted_token_id, permute_probs


def npu_moe_token_unpermute_with_routing_map_grad_mock(unpermuted_tokens_grad, out_index, permuted_token_id,
                                                       routing_map, permuted_tokens, probs, drop_and_pad,
                                                       restore_shape):
    permuted_tokens_grad_out = torch.zeros([out_index.shape[0], unpermuted_tokens_grad.shape[1]],
                                           dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
    if probs is not None:
        probs_grad_out = torch.zeros(probs.shape, dtype=unpermuted_tokens_grad.dtype,
                                     device=unpermuted_tokens_grad.device)
        return permuted_tokens_grad_out, probs_grad_out
    else:
        return permuted_tokens_grad_out, None


def npu_moe_token_permute_with_routing_map_grad_mock(permuted_token_out_grad, probs_grad, sorted_indices, routing_map,
                                                     experts_num, tokens_num, drop_and_pad):
    tokens_grad_out = torch.zeros([tokens_num, permuted_token_out_grad.size(1)], dtype=permuted_token_out_grad.dtype,
                                device=permuted_token_out_grad.device)
    probs_grad_out_optional = None
    if probs_grad is not None:
        probs_grad_out_optional = torch.zeros([tokens_num, experts_num], dtype=probs_grad.dtype,
                                           device=permuted_token_out_grad.device)
    return tokens_grad_out, probs_grad_out_optional


torch_npu._npu_moe_token_unpermute_with_routing_map = _npu_moe_token_unpermute_with_routing_map_mock
print_sim_info_rank_0(
    "[SIM_INFO] torch_npu._npu_moe_token_unpermute_with_routing_map "
    "to _npu_moe_token_unpermute_with_routing_map_mock......",
    flush=True)

torch_npu.npu_moe_token_unpermute_with_routing_map_grad = npu_moe_token_unpermute_with_routing_map_grad_mock
print_sim_info_rank_0(
    "[SIM_INFO] torch_npu.npu_moe_token_unpermute_with_routing_map_grad "
    "to npu_moe_token_unpermute_with_routing_map_grad_mock......",
    flush=True)

torch_npu.npu_moe_token_permute_with_routing_map_grad = npu_moe_token_permute_with_routing_map_grad_mock
print_sim_info_rank_0(
    "[SIM_INFO] torch_npu.npu_moe_token_permute_with_routing_map_grad "
    "to npu_moe_token_permute_with_routing_map_grad_mock......",
    flush=True)
