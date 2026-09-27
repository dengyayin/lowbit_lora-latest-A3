#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.

from functools import wraps
import torch
from . import print_sim_info_rank_0


# 将rand方法的dtype从bfloat16改成float16, 因为A6似乎还不支持bloat16类型的rand 带后续测试完成后，需要解桩
def tensor_rand_wrapper(rand):
    @wraps(rand)
    def wrapper(*args, **kwargs):
        # 处理 dtype 参数
        if 'dtype' in kwargs:
            if kwargs['dtype'] == torch.bfloat16:
                kwargs['dtype'] = torch.float16
        elif args:
            # dtype 参数通常在第二个位置
            if len(args) >= 2 and isinstance(args[1], torch.dtype) and args[1] == torch.bfloat16:
                # 替换 bfloat16 为 float16
                new_args = list(args)
                new_args[1] = torch.float16
                args = tuple(new_args)
        return rand(*args, **kwargs)

    return wrapper


torch.rand = tensor_rand_wrapper(torch.rand)
print_sim_info_rank_0(
    "[SIM_INFO] torch.rand to rand_wrapper......",
    flush=True)
