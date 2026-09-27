#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.

import torch
from torch import Tensor
from torch.types import (
    _bool
)
from . import print_sim_info_rank_0


# A6 CANN 不支持算子 ForeachCopy 若后续CANN修复，则需要解桩
def tensor__foreach_copy__mock(
        self: tuple[Tensor, ...] | list[Tensor] | None,
        src: tuple[Tensor, ...] | list[Tensor] | None,
        non_blocking: _bool = False,
) -> None:
    self = src
    return None


torch._foreach_copy_ = tensor__foreach_copy__mock
print_sim_info_rank_0(
    "[SIM_INFO] torch._foreach_copy_ to _foreach_copy__mock......",
    flush=True)
