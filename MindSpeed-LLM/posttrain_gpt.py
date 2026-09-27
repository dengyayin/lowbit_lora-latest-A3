# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from simulator_patch import pre_patch
from mindspeed_llm import megatron_adaptor
from simulator_patch import post_patch
from mindspeed_llm.tasks.posttrain.launcher import AutoTrainer

def launch():
    trainer = AutoTrainer()
    trainer.train()


if __name__ == '__main__':
    launch()
