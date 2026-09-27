#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.

import torch
from mindspeed.core.optimizer.fused_ema_adamw import fused_ema_adamw as ms_fused_ema_adamw
from . import print_sim_info_rank_0


# A6 CANN 不支持算子 fused_ema_adamw 若后续CANN修复，则需要解桩
def npu_apply_fused_ema_adamw_mock(grad: torch.Tensor,
                                   var: torch.Tensor,
                                   m: torch.Tensor,
                                   v: torch.Tensor,
                                   s: torch.Tensor,
                                   step: torch.Tensor,
                                   lr: float = 1e-3,
                                   ema_decay: float = 0.9999,
                                   beta1: float = 0.9,
                                   beta2: float = 0.999,
                                   eps: float = 1e-8,
                                   mode: int = 1,
                                   bias_correction: bool = True,
                                   weight_decay: float = 0.0):
    return grad, m, v, s


ms_fused_ema_adamw.npu_apply_fused_ema_adamw = npu_apply_fused_ema_adamw_mock

print_sim_info_rank_0(
    "[SIM_INFO] npu_apply_fused_ema_adamw to npu_apply_fused_ema_adamw_mock......",
    flush=True)
