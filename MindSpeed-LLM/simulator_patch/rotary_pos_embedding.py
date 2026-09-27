#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
import torch
from megatron.training import get_args
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindspeed.args_utils import get_mindspeed_args


def get_rotary_seq_len_mock(
        self,
        inference_params,
        transformer,
        transformer_input,
        transformer_config,
        packed_seq_params=None
) -> float:
    if inference_params is not None:
        rotary_seq_len = inference_params.max_sequence_length
    else:
        if transformer.input_tensor is not None:
            rotary_seq_len = transformer.input_tensor.size(0)
        else:
            rotary_seq_len = transformer_input.size(0)

        if transformer_config.sequence_parallel:
            if not transformer_config.variable_seq_lengths:
                rotary_seq_len = get_args().seq_length
            else:
                # 这里要先cpu再npu，否则转整数会变0
                rotary_seq_len = torch.Tensor([rotary_seq_len]).cpu().int().npu()
                torch.distributed.all_reduce(rotary_seq_len, op=torch.distributed.ReduceOp.SUM,
                                             group=get_tensor_model_parallel_group())
                rotary_seq_len = rotary_seq_len.item()

    rotary_seq_len *= transformer_config.context_parallel_size
    return rotary_seq_len


if getattr(get_mindspeed_args(), "unaligned-linear".lower().strip().replace('-', '_'), None):
    RotaryEmbedding.get_rotary_seq_len = get_rotary_seq_len_mock
