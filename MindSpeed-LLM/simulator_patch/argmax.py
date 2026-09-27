#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

from functools import wraps
import torch
from . import print_sim_info_rank_0


def argmax_wrapper(argmax):
    @wraps(argmax)
    def wrapper(input_tensor, dim=None, keepdim=False):
        result = argmax(input_tensor, dim, keepdim)

        if dim is None:
            max_index = input_tensor.numel()
        else:
            max_index = input_tensor.size(dim)
        # 仿真不计算，所以要cpu上置零发到npu上
        # 如果仿真仍出现问题，可以考虑去除掉.npu
        result = torch.randint(
            low=0,
            high=max_index,
            size=result.shape,
            dtype=result.dtype,
            device="cpu"
        ).npu()

        return result
    return wrapper


torch.argmax = argmax_wrapper(torch.argmax)
print_sim_info_rank_0(
    "[SIM_INFO] torch.argmax to argmax_wrapper......",
    flush=True
)
