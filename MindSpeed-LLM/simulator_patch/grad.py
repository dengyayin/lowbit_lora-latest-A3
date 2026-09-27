#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
import torch
from . import print_sim_info_rank_0


def grad_getter_mock(self):
    try:
        return self.clone()
    except:
        return torch.zeros(self.shape).to(self.dtype).to(self.device)


def grad_setter_mock(self, value):
    return


def grad_del_mock(self):
    return


torch.Tensor._grad = torch.Tensor.grad
torch.Tensor.grad = property(grad_getter_mock, grad_setter_mock, grad_del_mock)

print_sim_info_rank_0(
    "[SIM_INFO] Tensor.grad to Tensor.grad_mock ......",
    flush=True)
