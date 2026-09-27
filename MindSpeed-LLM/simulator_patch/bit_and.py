#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.

from functools import wraps
from torch import Tensor
from . import print_sim_info_rank_0


# 位运算只能在cpu上进行，否则报错 Parse dynamic kernel config fail. 可能和PTA有关,若后续修复，则需要解桩
def tensor___rand___wrapper(__and__):
    @wraps(__and__)
    def wrapper(self, other):
        if isinstance(other, Tensor):
            return __and__(self, other)
        else:
            return __and__(self.cpu(), other).npu()
    return wrapper


Tensor.__and__ = tensor___rand___wrapper(Tensor.__and__)
print_sim_info_rank_0(
    "[SIM_INFO] Tensor.__and__ to tensor___rand___wrapper ......",
    flush=True)
