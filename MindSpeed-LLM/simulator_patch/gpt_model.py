#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.

import torch
from torch import Tensor

from megatron.core.models.gpt.gpt_model import GPTModel


def set_input_tensor_mock(self, input_tensor: Tensor) -> None:
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]

    # 规避因通信导致的全0tensor问题，如果遇到了全0tensor，则直接根据已有配置生成一个
    if input_tensor[0] is not None and input_tensor[0].shape[0] == 0:
        if hasattr(self.config, "seq_length") and hasattr(self.config, "micro_batch_size") and hasattr(self.config,
                                                                                                       "hidden_size"):
            self.decoder.set_input_tensor(
                torch.zeros([self.config.seq_length, self.config.micro_batch_size, self.config.hidden_size],
                            dtype=input_tensor[0].dtype, requires_grad=True).npu())
        elif hasattr(self.config, "pipeline_tensor_shapes"):
            self.decoder.set_input_tensor(
                torch.zeros(self.config.pipeline_tensor_shapes[0]['shape'], dtype=input_tensor[0].dtype,
                            requires_grad=True).npu())
        else:
            self.decoder.set_input_tensor(input_tensor[0])
    else:
        self.decoder.set_input_tensor(input_tensor[0])


GPTModel.set_input_tensor = set_input_tensor_mock
