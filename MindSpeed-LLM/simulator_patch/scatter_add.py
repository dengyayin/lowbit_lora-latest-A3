#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
from torch import Tensor

from torch.types import (
    _int
)

from . import print_sim_info_rank_0


def tensor_scatter_add__mock(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
    return self


Tensor.scatter_add_ = tensor_scatter_add__mock
print_sim_info_rank_0(
    "[SIM_INFO] torch.diff to diff_wrapper......",
    flush=True)
