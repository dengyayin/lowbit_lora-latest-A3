#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import pickle
import acl
import torch
from megatron.core import parallel_state as ps
from megatron.training import print_rank_0
from . import can_apply_patches, apply_patches, print_sim_info_rank_0


def bind_cpus_patch(world_size, rank_id, device_id, ratio=0.5):
    return


def update_whole_model_memory_patch(self):
    _, all_memory, _ = acl.rt.get_mem_info(1)
    all_memory = 62432 * 1024 * 1024
    self.context[Key.USED_MEM] = torch.npu.memory_allocated() / BYTES_PER_MB
    self.context[Key.DEVICE_MEMORY] = all_memory / BYTES_PER_MB


def broadcast_obj_patch(obj):
    try:
        mp = ps.get_tensor_model_parallel_world_size()
        dp = ps.get_data_parallel_world_size(True)
    except:
        print_rank_0("WARNING: mp, dp is not defined")
    global_rank = torch.distributed.get_rank()
    src = (global_rank // (mp * dp)) * dp * mp
    obj_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
    obj_shape_tensor = torch.tensor(obj_tensor.shape, device=torch.npu.current_device())
    broadcast_in_mp_dp(obj_shape_tensor, src, mp, dp)
    obj_len = obj_shape_tensor.cpu().tolist()

    # 同步 src rank 的 shape
    final_tensor = torch.empty(obj_len, dtype=torch.uint8)
    split_size = 100
    # 将原始的 obj_tensor 拆分成不大于 100 字节的小张量
    if global_rank == src:
        small_tensors = torch.split(obj_tensor, split_size)
    else:
        small_tensors = torch.split(final_tensor, split_size)

    synced_tensors = []
    
    for small_tensor in small_tensors:
        if global_rank == src:
            obj_tensor_npu = small_tensor.npu()
        else:
            obj_tensor_npu = torch.empty(small_tensor.numel(), dtype=torch.uint8, device=torch.npu.current_device())
        broadcast_in_mp_dp(obj_tensor_npu, src, mp, dp)
        synced_tensors.append(obj_tensor_npu.cpu())
        del obj_tensor_npu

    final_tensor = torch.cat(synced_tensors)
    result = pickle.loads(final_tensor.numpy().tobytes())
    return result

need_patches = {
    'bind_cpus': bind_cpus_patch,
    'update_whole_model_memory': update_whole_model_memory_patch,
    'broadcast_obj': broadcast_obj_patch,
}


if can_apply_patches():

    print_sim_info_rank_0(f'[SIM_INFO] exec patch function', flush=True)

    import mindspeed.core.memory.adaptive_memory.adaptive_memory_tool as adaptive_memory_tool

    BYTES_PER_MB = adaptive_memory_tool.BYTES_PER_MB
    broadcast_in_mp_dp = adaptive_memory_tool.broadcast_in_mp_dp
    Key = adaptive_memory_tool.ContextKey

    for func_name, patch_func in need_patches.items():
        apply_patches(func_name, patch_func)

    from mindspeed.core.memory.adaptive_memory.adaptive_memory_profiling import AdaptiveMemoryProfiling
    AdaptiveMemoryProfiling.update_whole_model_memory = update_whole_model_memory_patch
    print_sim_info_rank_0(f'[SIM_INFO] patched AdaptiveMemoryProfiling.update_whole_model_memory', flush=True)

    print_sim_info_rank_0(f'[SIM_INFO] exec patch function done', flush=True)