#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
from typing import Optional

import torch
from megatron.core import mpu
from megatron.training import get_args
from mindspeed_mm.models.transformers.cp_utils import get_seq_len, set_seq_len, split_visual_seqs_with_cp
from mindspeed_mm.models.transformers.qwen3vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLModel
from mindspeed_mm.models.common.communications import (
    cal_split_sizes,
    gather_forward_split_backward,
    cal_split_sizes_multi
)
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ring_rank
)
from . import print_sim_info_rank_0


def forward_mock(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    if self.data_balance is not None:
        hidden_states, grid_thw = self.data_balance.get_image_balance_data(
            {'pixel_values': hidden_states, 'image_grid_thw': grid_thw}
        )
    hidden_states = self.patch_embed(hidden_states)  # s1+s2+s3..., h
    seq_len, _ = hidden_states.size()
    sequence_lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cpu()
    set_seq_len("per_visual", sequence_lengths)
    set_seq_len("visual", seq_len)
    hidden_states = hidden_states.reshape(seq_len, -1)
    if mpu.get_context_parallel_world_size() > 1:
        hidden_states = split_visual_seqs_with_cp(hidden_states, dim=0)
    # 此处跳过处理hidden_states
    deepstack_feature_lists = []
    hidden_states = self.merger(hidden_states)
    if self.data_balance is not None:
        hidden_states, deepstack_feature_lists = self.data_balance.reverse_img_balance_data(
            hidden_states, deepstack_feature_lists
        )
    set_seq_len("visual", seq_len // self.spatial_merge_size ** 2)
    if mpu.get_context_parallel_world_size() > 1:
        gather_sizes = cal_split_sizes(get_seq_len("visual"), mpu.get_context_parallel_world_size())
        hidden_states = gather_forward_split_backward(
            hidden_states,
            mpu.get_context_parallel_group(),
            dim=0,
            grad_scale="up",
            gather_sizes=gather_sizes
        )
    return hidden_states, deepstack_feature_lists


def get_image_features_mock(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
    if hasattr(self.visual,
               "_get_fsdp_state") and self.visual._get_fsdp_state()._mp_policy.param_dtype != pixel_values.dtype:
        param_dtype = self.visual._get_fsdp_state()._mp_policy.param_dtype
        pixel_values = pixel_values.type(param_dtype) if param_dtype is not None else pixel_values
    image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

    product = 1
    for i in image_grid_thw[-1]:
        product *= i.item()
    # 此处直接计算split size
    split_sizes = [product // self.visual.spatial_merge_size ** 2]

    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds


def get_placeholder_mask_mock(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
):
    if input_ids is None:
        special_image_mask = inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_image_mask = special_image_mask.all(-1)
        special_video_mask = inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_video_mask = special_video_mask.all(-1)
    else:
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)  # b s h
    # 去掉校验
    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    return special_image_mask, special_video_mask


def get_rope_index_mock(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_length: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if sequence_length is None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            attention_mask = attention_mask.to(total_input_ids.device)
        else:
            total_input_ids = input_ids[0].split(sequence_length.tolist())
            max_input_ids_len = max(sequence_length)
            position_ids = [None] * len(sequence_length)
        for i, input_ids in enumerate(total_input_ids):
            if sequence_length is None:
                input_ids = input_ids[attention_mask[i] == 1]
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            # 给一个初值
            st = 0
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if sequence_length is None:
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            else:
                position_ids[i] = llm_positions.to(input_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - max_input_ids_len)
        if sequence_length is not None:
            position_ids = torch.cat(position_ids, dim=-1).unsqueeze(1)
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        return raw_get_rope_index_method(attention_mask, input_ids)


def raw_get_rope_index_method(attention_mask, input_ids):
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas


Qwen3VLVisionModel.forward = forward_mock
Qwen3VLModel.get_image_features = get_image_features_mock
Qwen3VLModel.get_placeholder_mask = get_placeholder_mask_mock
Qwen3VLModel.get_rope_index = get_rope_index_mock
print_sim_info_rank_0(
    "[SIM_INFO] mindspeed_mm.models.transformers.qwen3vl.modeling_qwen3_vl to Qwen3VLModel patch......",
    flush=True)
