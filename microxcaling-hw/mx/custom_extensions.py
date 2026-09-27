"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Python interface for custom CUDA implementations of functions.
"""

import os
from torch.utils.cpp_extension import load

sources = [
    "funcs.cpp",
    "mx.cu",
    "elemwise.cu",
    "reduce.cu",
]
file_dir = os.path.dirname(__file__)
sources = [os.path.join(file_dir, "cpp", x) for x in sources]


import torch
cap = torch.cuda.get_device_capability()   # (10,1)
arch = f"{cap[0]}{cap[1]}"                 # '101'
# 只保留 sm_100，避开 PyTorch 不认识的 101
funcs = load(
    name="custom_extensions",
    sources=sources,
    extra_cuda_cflags=['-gencode=arch=compute_100,code=sm_100'],
)