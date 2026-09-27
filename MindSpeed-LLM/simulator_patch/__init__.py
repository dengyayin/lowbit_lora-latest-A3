#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Hints: put this module in Megatron-LM or Mindspeed-LLM
#        Ensure that the 'megatron' folder exists in the same directory.
#        Add 'from simulator_patch import pre_patch' before 'import mindspeed.megatron_adaptor' in pretrain_gpt.py
#        Add 'from simulator_patch import post_patch' after 'import mindspeed.megatron_adaptor' in pretrain_gpt.py
import os


def print_sim_info_rank_0(fstring, flush=True):
    if int(os.getenv('RANK', '0')) == 0:
        print(fstring, flush=flush)
