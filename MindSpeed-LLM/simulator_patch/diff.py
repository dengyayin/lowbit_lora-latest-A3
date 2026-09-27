#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
from functools import wraps
from torch import Tensor
from torch.types import (
    _int
)
from . import print_sim_info_rank_0


def tensor_diff_wrapper(diff):
    @wraps(diff)
    def wrapper(
            self,
            n: _int = 1,
            dim: _int = -1,
            prepend: Tensor | None = None,
            append: Tensor | None = None,
    ) -> Tensor:
        return diff(self.cpu(), n, dim, prepend, append).to(self.device)

    return wrapper


Tensor.diff = tensor_diff_wrapper(Tensor.diff)
print_sim_info_rank_0(
    "[SIM_INFO] torch.diff to diff_wrapper......",
    flush=True)
