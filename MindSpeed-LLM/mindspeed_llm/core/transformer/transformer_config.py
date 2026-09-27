# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from functools import wraps

from megatron.training import get_args


def transformer_config_post_init_mtp_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        fn(self)
        args = get_args()

        self.mtp_num_layers = args.mtp_num_layers
        self.mtp_loss_scaling_factor = args.mtp_loss_scaling_factor

    return wrapper
