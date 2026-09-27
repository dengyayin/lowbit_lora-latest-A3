#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
from torch import Tensor
from torch.types import (
    _bool
)
from . import print_sim_info_rank_0


def tensor_copy__mock(self, other: Tensor, non_blocking: _bool = False) -> Tensor:
    return self


Tensor.copy_ = tensor_copy__mock
print_sim_info_rank_0(
    "[SIM_INFO] torch.copy to copy_wrapper......",
    flush=True)
