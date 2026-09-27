#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.

from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.utils.global_vars import get_args

from mindspeed_llm.fsdp2.data.megatron_data import megatron_dataset_generate

logger = get_logger(__name__)


def is_shared_path_mock(path: str, retry: int = 3, wait: float = 0.5) -> bool:
    return True


megatron_dataset_generate.is_shared_path = is_shared_path_mock
