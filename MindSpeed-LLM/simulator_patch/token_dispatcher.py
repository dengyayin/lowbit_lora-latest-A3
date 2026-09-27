#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import time
from typing import Optional
from functools import wraps
import fcntl
import numpy as np
import torch
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.token_dispatcher \
    import MindSpeedMOEAlltoAllFbOverlapTokenDispatcher
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import All2AllSeqTp2epDispatcherImpl
from mindspeed.core.transformer.moe.moe_feature.overlap.token_dispatcher import MoEAlltoAllSeqOverLapDispatcher
from mindspeed.core.transformer.moe.moe_feature.overlap import token_dispatcher as mindspeed_overlap_token_dispatcher
from mindspeed.core.fusions import fused_moe_permute as mindspeed_fused_moe_permute
from megatron.core.transformer.moe import token_dispatcher as megatron_token_dispatcher
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep import \
    token_dispatcher as mindspeed_tp_extend_ep_token_dispatcher
from megatron.training import get_args
from megatron.core import parallel_state
from . import print_sim_info_rank_0

curr_token_size = {}
SAVE_DIR = "/root/simulator/moe/"


def generate_indices_array(indices: torch.Tensor, num_moe_experts: int) -> np.ndarray:
    num_token = indices.shape[0]
    top_k = get_args().moe_router_topk
    token_total_num = num_token * top_k
    np.random.seed(int(time.time()))
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tokens_count = get_args().world_size // tp_size

    # 生成原始数组
    one_array = np.ones(tokens_count)
    indices_array = (one_array * token_total_num).astype(np.int32)
    # 需要是专家的倍数
    for i in range(indices_array.size):
        indices_array[i] -= (indices_array[i] % num_moe_experts)
    # 总量为tokens_count * indices.numel()
    if indices_array.sum() != tokens_count * token_total_num:
        indices_array[-1] = tokens_count * token_total_num - indices_array[:-1].sum()
    save_path = os.path.join(SAVE_DIR, "indices_array.txt")

    with open(save_path, 'w+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if os.path.getsize(save_path) > 0:
            fcntl.flock(f, fcntl.LOCK_UN)
            return
        np.savetxt(f, indices_array, fmt="%d")
        fcntl.flock(f, fcntl.LOCK_UN)


# rules: sum(input_split) equal indices.numel() &&
#        sum(output_split) equal self.global_input_tokens_local_experts_indices.numel()
def preprocess_simulator_wrapper(preprocess):
    @wraps(preprocess)
    def wrapper(self, indices: torch.Tensor) -> torch.Tensor:
        os.makedirs(SAVE_DIR, exist_ok=True)
        num_tokens_per_local_expert = preprocess(self, indices)
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        device_rank = torch.distributed.get_rank() // tp_size

        save_path = os.path.join(SAVE_DIR, "indices_array.txt")
        if not os.path.exists(save_path):
            generate_indices_array(indices, self.config.num_moe_experts)

        if device_rank in curr_token_size:
            tokens_num = curr_token_size[device_rank]
        else:
            num_token = indices.shape[0]
            top_k = get_args().moe_router_topk
            tokens_num = num_token * top_k
            curr_token_size[device_rank] = tokens_num

        if self.input_splits is not None:
            if isinstance(self.input_splits, torch.Tensor):
                split_size = self.input_splits.numel()
            else:
                split_size = self.input_splits.size

            # input_splits和output_splits不能均分
            self.input_splits = torch.tensor(
                [tokens_num] + [0] * (split_size - 1),
                dtype=num_tokens_per_local_expert.dtype,
                device=torch.device("cpu")
            ).numpy()

            self.output_splits = torch.tensor(
                [tokens_num] + [0] * (split_size - 1),
                dtype=num_tokens_per_local_expert.dtype,
                device=torch.device("cpu")
            ).numpy()
        if hasattr(self, 'output_splits_tp'):
            if self.output_splits_tp is not None:
                if isinstance(self.output_splits_tp, torch.Tensor):
                    self.output_splits_tp = self.output_splits_tp.cpu().numpy()
                self.output_splits_tp[0] = tokens_num

        num_tokens_per_local_expert = torch.tensor(
            [tokens_num // self.config.num_moe_experts * self.num_local_experts * 8] + \
            [0] * (num_tokens_per_local_expert.numel() - 1),
            dtype=num_tokens_per_local_expert.dtype,
            device=num_tokens_per_local_expert.device,
        )
        return num_tokens_per_local_expert

    return wrapper


def preprocess_simulator_for_moe_wrapper(preprocess):
    @wraps(preprocess)
    def wrapper(self, indices: torch.Tensor) -> torch.Tensor:
        os.makedirs(SAVE_DIR, exist_ok=True)
        num_tokens_per_local_expert = preprocess(self, indices)
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        device_rank = torch.distributed.get_rank() // tp_size

        save_path = os.path.join(SAVE_DIR, "indices_array.txt")
        if not os.path.exists(save_path):
            generate_indices_array(indices, self.config.num_moe_experts)

        if device_rank in curr_token_size:
            tokens_num = curr_token_size[device_rank]
        else:
            with open(save_path, "r") as f:
                lines = f.readlines()
                cnt = 0
                while (len(lines) <= device_rank) and (cnt < 1000):
                    lines = f.readlines()
                    cnt = cnt + 1
                tokens_num = int(lines[device_rank].strip())
                curr_token_size[device_rank] = tokens_num
        if self.input_splits is not None:
            if isinstance(self.input_splits, torch.Tensor):
                split_size = self.input_splits.numel()
            else:
                split_size = self.input_splits.size
            self.input_splits = torch.tensor(
                [indices.numel()] + [0] * (split_size - 1),
                dtype=num_tokens_per_local_expert.dtype,
                device=torch.device("cpu")
            ).numpy()

            self.output_splits = torch.tensor(
                [tokens_num] + [0] * (split_size - 1),
                dtype=num_tokens_per_local_expert.dtype,
                device=torch.device("cpu")
            ).numpy()
        self.output_splits_tp = None
        num_tokens_per_local_expert = torch.tensor(
            [tokens_num // self.config.num_moe_experts * self.num_local_experts] + \
            [0] * (num_tokens_per_local_expert.numel() - 1),
            dtype=num_tokens_per_local_expert.dtype,
            device=num_tokens_per_local_expert.device,
        )
        return num_tokens_per_local_expert

    return wrapper


def async_comm_sort_chunks_by_idxs_mock(
        input_tensor: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        fused: bool = False,
        prob_handle=None
):
    if fused:
        raise AssertionError('async sort_chunks_by_idxs not support fused now.')
    mockshape = [input_tensor.shape[0] // split_sizes.shape[0]] * split_sizes.shape[0]
    input_tensor = torch.split(input_tensor, mockshape, dim=0)
    # 手动切分tensor
    output = torch.cat([input_tensor[i] for i in sorted_idxs.tolist()], dim=0)
    if probs is not None:
        if prob_handle:
            prob_handle.wait()
        probs = torch.split(probs, mockshape, dim=0)
        permuted_probs = torch.cat([probs[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


def sort_chunks_by_idxs_mock(
        input_tensor: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        fused: bool = False,
):
    mockshape = [input_tensor.shape[0] // split_sizes.shape[0]] * split_sizes.shape[0]
    input_tensor = torch.split(input_tensor, mockshape, dim=0)
    output = torch.cat([input_tensor[i] for i in sorted_idxs.tolist()], dim=0)
    if probs is not None:
        probs = torch.split(probs, mockshape, dim=0)
        permuted_probs = torch.cat([probs[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


def permute_mock(
        tokens,
        routing_map,
        probs: Optional[torch.Tensor] = None,
        num_out_tokens: Optional[int] = None,
        fused: bool = False,
        drop_and_pad: bool = False,
):
    top_k_num = get_args().moe_router_topk
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
                         :, :capacity
                         ].contiguous()
        sorted_indices = sorted_indices.view(-1)
        if probs is not None:
            routing_map = routing_map.bool()
    else:
        routing_map = routing_map.bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = torch.zeros(num_tokens * top_k_num, dtype=torch.int32, device=routing_map.device)
    permuted_input = tokens.index_select(0, sorted_indices)
    if probs is not None:
        permuted_probs = probs.T.contiguous().masked_select(routing_map).index_select(0, sorted_indices)
    else:
        permuted_probs = None
    return permuted_input, permuted_probs, sorted_indices


def permute_wrapper(permute):
    @wraps(permute)
    def wrapper(
            tokens,
            routing_map,
            probs: Optional[torch.Tensor] = None,
            num_out_tokens: Optional[int] = None,
            fused: bool = False,
            drop_and_pad: bool = False,
    ):
        permuted_input, permuted_probs, sorted_indices = permute(tokens, routing_map, probs, num_out_tokens, fused,
                                                                 drop_and_pad)
        if sorted_indices.dtype == torch.int64:
            sorted_indices = sorted_indices.to(torch.int32)
        return permuted_input, permuted_probs, sorted_indices

    return wrapper


try:
    os.remove("/root/simulator/moe/indices_array.txt")
except OSError as e:
    print_sim_info_rank_0(f"[SIM_INFO] No file indices_array.txt: /root/simulator/moe/indices_array.txt")

MindSpeedMOEAlltoAllFbOverlapTokenDispatcher.preprocess = preprocess_simulator_for_moe_wrapper(
    MindSpeedMOEAlltoAllFbOverlapTokenDispatcher.preprocess)
MoEAlltoAllTokenDispatcher.preprocess = preprocess_simulator_for_moe_wrapper(MoEAlltoAllTokenDispatcher.preprocess)
MoEAlltoAllSEQTokenDispatcher.preprocess = preprocess_simulator_for_moe_wrapper(
    MoEAlltoAllSEQTokenDispatcher.preprocess)
All2AllSeqTp2epDispatcherImpl.preprocess = preprocess_simulator_for_moe_wrapper(
    All2AllSeqTp2epDispatcherImpl.preprocess)
MoEAlltoAllSeqOverLapDispatcher.preprocess_overlap = preprocess_simulator_wrapper(
    MoEAlltoAllSeqOverLapDispatcher.preprocess_overlap)
mindspeed_overlap_token_dispatcher.async_comm_sort_chunks_by_idxs = async_comm_sort_chunks_by_idxs_mock
mindspeed_overlap_token_dispatcher.permute = permute_mock
megatron_token_dispatcher.permute = permute_wrapper(megatron_token_dispatcher.permute)
mindspeed_overlap_token_dispatcher.sort_chunks_by_idxs = sort_chunks_by_idxs_mock
mindspeed_fused_moe_permute.sort_chunks_by_idxs = sort_chunks_by_idxs_mock
mindspeed_tp_extend_ep_token_dispatcher.sort_chunks_by_idxs = sort_chunks_by_idxs_mock
print_sim_info_rank_0("[SIM_INFO] megatron.core.transformer.moe.token_dispatcher to preprocess_simulator_patch......",
                      flush=True
                      )
