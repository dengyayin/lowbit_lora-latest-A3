#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

from functools import wraps
from typing import List, Optional, Tuple
import torch
import torch_npu
# 主要替换 megatron.core.transformer.moe.moe_utils 中方法
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules import token_dispatcher as mindspeed_token_dispatcher
from megatron.core.transformer.moe import token_dispatcher as megatron_token_dispatcher
from megatron.core.transformer.moe import legacy_a2a_token_dispatcher as megatron_legacy_a2a_token_dispatcher
from mindspeed.core.fusions import fused_moe_permute as mindspeed_fused_moe_permute
from megatron.core.transformer.moe.moe_utils import permute as megatron_permute
from mindspeed.te.pytorch.permutation import MoePermuteMaskMap
from mindspeed.te.pytorch.permutation import MoeUnpermuteMaskMap

def sort_chunks_by_idxs_wrapper(sort_chunks_by_idxs):
    @wraps(sort_chunks_by_idxs)
    def wrapper(
            input_tensor: torch.Tensor,
            split_sizes: torch.Tensor,
            sorted_idxs: torch.Tensor,
            probs: Optional[torch.Tensor] = None,
            fused: bool = False,
    ):
        split_sizes = torch.zeros_like(split_sizes, device='cpu')
        split_sizes.view(-1)[0] = input_tensor.size(0)
        return sort_chunks_by_idxs(input_tensor, split_sizes, sorted_idxs, probs, fused)

    return wrapper


class MockMoePermuteMaskMap(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            tokens,
            routing_map,
            probs: Optional[torch.Tensor] = None,
            num_out_tokens: Optional[int] = None,
            drop_and_pad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        permuted_input, permuted_probs, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
            tokens, routing_map, probs=probs, num_out_tokens=num_out_tokens, drop_and_pad=drop_and_pad)
        num_tokens, _ = tokens.shape
        num_experts = routing_map.shape[1]
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.drop_and_pad = drop_and_pad
        if sorted_indices.dtype == torch.int64:
            sorted_indices = sorted_indices.to(torch.int32)
        ctx.sorted_indices = sorted_indices
        ctx.routing_map = routing_map
        return megatron_permute(tokens, routing_map, probs, num_out_tokens, False, drop_and_pad)

    @staticmethod
    def backward(
            ctx,
            permuted_act_grad: torch.Tensor,
            permuted_probs_grad: torch.Tensor,
            _,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            sorted_indices = ctx.sorted_indices
            routing_map = ctx.routing_map
            num_tokens = ctx.num_tokens
            num_experts = ctx.num_experts
            drop_and_pad = ctx.drop_and_pad

            act_grad, probs_grad = torch_npu.npu_moe_token_permute_with_routing_map_grad(
                permuted_act_grad,
                permuted_probs_grad,
                sorted_indices,
                routing_map,
                num_experts,
                num_tokens,
                drop_and_pad
            )
        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None, None

def moe_unpermute_mask_map_forward(forward):
    @wraps(forward)
    def wrapper(
            ctx,
            permuted_tokens: torch.Tensor,
            sorted_indices: torch.Tensor,
            restore_shape: torch.Size,
            probs: Optional[torch.Tensor] = None,
            routing_map: Optional[torch.Tensor] = None,
            drop_and_pad: bool = False,
    ) -> torch.Tensor:
        if sorted_indices.dtype == torch.int64:
            sorted_indices = sorted_indices.to(torch.int32)
        return forward(ctx,permuted_tokens,sorted_indices,restore_shape,probs,routing_map,drop_and_pad)
    return wrapper

mindspeed_token_dispatcher.sort_chunks_by_idxs = \
    sort_chunks_by_idxs_wrapper(mindspeed_token_dispatcher.sort_chunks_by_idxs)
megatron_token_dispatcher.sort_chunks_by_idxs = \
    sort_chunks_by_idxs_wrapper(megatron_token_dispatcher.sort_chunks_by_idxs)
megatron_legacy_a2a_token_dispatcher.sort_chunks_by_idxs = \
    sort_chunks_by_idxs_wrapper(megatron_legacy_a2a_token_dispatcher.sort_chunks_by_idxs)
mindspeed_fused_moe_permute.MoePermuteMaskMap = MockMoePermuteMaskMap

MoeUnpermuteMaskMap.forward = moe_unpermute_mask_map_forward(MoeUnpermuteMaskMap.forward)