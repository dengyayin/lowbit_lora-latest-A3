#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from functools import wraps

import torch
from torch import Tensor
from . import print_sim_info_rank_0


# A6 要用AICore去做concat操作，但仿真不支持
def _flatten_dense_tensors_wrapper(flatten_dense_tensors):
    @wraps(flatten_dense_tensors)
    def wrapper(tensors: list[Tensor]) -> Tensor:
        if tensors is None or len(tensors) == 0:
            return flatten_dense_tensors(tensors)
        cnt = 0
        device = None
        dtype = None
        for t in tensors:
            device = t.device
            dtype = t.dtype
            cnt = cnt + t.numel()
        return torch.zeros([cnt]).to(dtype).to(device)

    return wrapper

torch._C._nn.flatten_dense_tensors = _flatten_dense_tensors_wrapper(torch._C._nn.flatten_dense_tensors)
print_sim_info_rank_0(
    "[SIM_INFO] torch._flatten_dense_tensors to _flatten_dense_tensors_wrapper......",
    flush=True
)
