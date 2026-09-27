import torch
from typing import Tuple

from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction
from mindspeed.lite.ops.triton.sort_chunks_by_idx import _moe_chunk_sort
from functools import wraps

from . import print_sim_info_rank_0


def autotuner_run_mock(self, *args, **kwargs):
    return


def jit_run_mock(self, *args, grid, warmup, **kwargs):
    return


def _moe_chunk_sort_forward_wrapper(forward):
    @wraps(forward)
    def wrapper(
            ctx,
            inp: torch.Tensor,
            split_sizes: torch.Tensor,
            sorted_idxs: torch.Tensor,
            probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if split_sizes.device.type != 'npu':
            split_sizes = split_sizes.npu()
        if sorted_idxs.device.type != 'npu':
            sorted_idxs = sorted_idxs.npu()
        return forward(ctx, inp, split_sizes, sorted_idxs, probs)

    return wrapper


Autotuner.run = autotuner_run_mock
JITFunction.run = jit_run_mock
_moe_chunk_sort.forward = _moe_chunk_sort_forward_wrapper(_moe_chunk_sort.forward)

print_sim_info_rank_0(
    "[SIM_INFO] triton mocked ......",
    flush=True)
