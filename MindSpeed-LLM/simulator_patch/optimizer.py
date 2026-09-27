#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.

from typing import List, Optional, Union

import torch

from megatron.core.optimizer import optimizer


# def get_grad_norm_fp32_mock(
#         grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
#         norm_type: Union[int, float] = 2,
#         grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
# ) -> float:
#     return 0.0
def get_grad_norm_fp32_mock(*args, **kwargs) -> float:
      return 0.0


optimizer.get_grad_norm_fp32 = get_grad_norm_fp32_mock

def clip_grad_by_total_norm_fp32_mock(*args, **kwargs):
    return 


optimizer.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32_mock