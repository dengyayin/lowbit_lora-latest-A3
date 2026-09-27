#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
from typing import Callable, Iterable, List, Optional, Tuple, Union
from functools import wraps

import torch
from mindspeed.core.pipeline_parallel.variable_seq_length import adaptor as mindspeed_variable_seq_length_adaptor
from mindspeed.core.pipeline_parallel.variable_seq_length.common import Config

Shape = Union[List[int], torch.Size]


def communicate_impl_wrapper(communicate_impl):
    @wraps(communicate_impl)
    def wrapper(
            tensor_send_next: Optional[torch.Tensor],
            tensor_send_prev: Optional[torch.Tensor],
            recv_prev: bool,
            recv_next: bool,
            tensor_shape: Shape,
            config: Config,
            get_pipeline_model_parallel_group: Callable,
            get_pipeline_model_parallel_next_rank: Callable,
            get_pipeline_model_parallel_prev_rank: Callable,
            batched_p2p_ops: Callable,
            p2p_ops: Callable,
            original_batched_p2p_ops: Callable,
            original_p2p_ops: Callable,
            wait_on_reqs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Iterable[torch.distributed.P2POp]]:
        # 强制让其走等长处理逻辑，规避通信处理（通信处理isend irevc存在问题）导致tensor size全0的问题
        bak_val = config.variable_seq_lengths
        if config.variable_seq_lengths:
            config.variable_seq_lengths = False
        res = communicate_impl(tensor_send_next, tensor_send_prev, recv_prev, recv_next, tensor_shape, config,
                               get_pipeline_model_parallel_group, get_pipeline_model_parallel_next_rank,
                               get_pipeline_model_parallel_prev_rank, batched_p2p_ops, p2p_ops,
                               original_batched_p2p_ops,
                               original_p2p_ops, wait_on_reqs)
        config.variable_seq_lengths = bak_val
        return res

    return wrapper


mindspeed_variable_seq_length_adaptor.communicate_impl = communicate_impl_wrapper(
    mindspeed_variable_seq_length_adaptor.communicate_impl)
