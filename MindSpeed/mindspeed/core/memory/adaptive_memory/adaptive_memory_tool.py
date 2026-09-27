# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from enum import Enum, IntEnum
from typing import Set, Dict
import pickle
import torch
from megatron.core import parallel_state as ps
from megatron.training import print_rank_0

BYTES_PER_MB = 1024 * 1024


class LayerAction(IntEnum):
    FULL_RECOMPUTE = 0
    FULL_SWAP = 1
    ADAPTIVE = 2
    NONE = 3


class ModuleAction(IntEnum):
    RECOMPUTE = 0
    SWAP = 1
    NONE = 2


class SingletonBase(type):
    singleton_instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.singleton_instances:
            instance = super().__call__(*args, **kwargs)
            cls.singleton_instances[cls] = instance
        return cls.singleton_instances[cls]


class ContextKey(SingletonBase):
    NAME = "name" # module name
    DEEP = "deep" # module depth
    PREFIX_NAME = "prefix_name" # module parent name
    MODULE = "module" # module field
    SUBMODULES = "submodules" # children modules
    INPUT = "input" # input args total size
    MEMORY = "memory" # current module's activation memory + input + output
    OUTPUT = "output" # output total size
    FORWARD_CNT = "forward_cnt" # forward called times
    PRE_TOTAL_TIME = "pre_total_time" # forward called total time
    AVG_TIME = "avg_time" # forward called avg time
    ALLOWED_ADAPT = "allowed_adapt" # allowed adapted modules, user init and set name at startup
    IS_FUNCTION = "is_function" # mark module if it is a torch.autograd.Function
    IS_MODULE_LIST = "is_module_list" # mark module if it is a torch.nn.ModuleList
    IS_ADAPT_LAYER = "is_adapt_layer" # mark self or parent if self is ALLOWED_ADAPT
    USED_MEM = "used_mem" # model memory consumption, as allocated memory
    DEVICE_MEMORY = "device_memory" # device total memory
    # adaptive
    MODULE_FORWARD_TOTAL_TIME = "module_forward_total_time" # module forward called total time
    MODULE_FORWARD_AVG_TIME = "module_forward_avg_time" # module forward called avg time
    MODULE_FORWARD_CNT = "module_forward_cnt" # module forward called times
    MODULE_SWAP_TOTAL_TIME = "module_swap_total_time" # module forward called total swap time
    MODULE_SWAP_AVG_TIME = "module_swap_avg_time" # module forward called avg swap time
    MODULE_SWAP_CNT = "module_swap_cnt" # module forward called swap times
    MODULE_SWAP_TOTAL_MEMORY = "module_swap_total_memory" # module forward swap total memory
    MODULE_SWAP_AVG_MEMORY = "module_swap_avg_memory" # module forward swap avg memory
    IS_SWAP = "is_swap" # mark module if it is swap
    IS_LAYER0_OF_MODULE0 = "is_layer0_of_module0" # mark module if it is layer0 of module0
    IS_MODLUE_OF_LAYER0 = "is_modlue_of_layer0" # mark module if it belongs to layer0 of module0


class FuncLocationMgr(metaclass=SingletonBase):
    def __init__(self):
        self._module_names = []
        self._function_in_stack = None
        self._function_child = None
        self.is_first_layer = False

    def push_name(self, prefix, name):
        self._module_names.append(f"{prefix}.{name}")
        if self._function_in_stack and not self._function_child:
            self._function_child = f"{prefix}.{name}"

    def pop_name(self, prefix, name):
        last_name = self._module_names.pop()
        if f"{prefix}.{name}" != last_name:
            raise ValueError(f"unexpected module name in stack, expect:{prefix}.{name}, find:{last_name}")

    def get_latest_name(self):
        return self._module_names[-1]

    def set_function_in_stack(self):
        self._function_in_stack = True

    def get_function_location(self, parent):
        if not self._function_child:
            direct_child = ""
        else:
            first_child = self._function_child[len(parent):]
            direct_child = first_child.split(".")[1]
        self._function_child = None
        self._function_in_stack = False
        return direct_child


class AdaptiveStepMgr(metaclass=SingletonBase):
    def __init__(self):
        self.cur_step = 1
        self.skip_steps = 3
        self.recompute_profiling_steps = 0
        se