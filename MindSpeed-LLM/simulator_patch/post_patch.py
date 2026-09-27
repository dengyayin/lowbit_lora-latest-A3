#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

from . import print_sim_info_rank_0


from . import optimizer

try:
    from . import moe_utils
    from . import token_dispatcher
    from . import gpt_model
    from . import rotary_pos_embedding
    # 更新大禹3.8B后就不需要 from . import communicate 了 因为isend irecv功能正常了

    from . import fused_ema_adamw
except ImportError:
    print_sim_info_rank_0(f"[SIM_INFO] not install Megatron And MindSpeed, skip patch it.")

try:
    from . import modeling_qwen3_vl
except ImportError:
    print_sim_info_rank_0(f"[SIM_INFO] not install MindSpeed_MM, skip patch it.")

try:
    from . import megatron_utils
except ImportError:
    print_sim_info_rank_0(f"[SIM_INFO] not install MindSpeed_LLM skip patch it.")

# 新CANN没有npuc 导致无法编译triton算子，需要打桩
try:
    from . import triton_mock
except ImportError:
    print_sim_info_rank_0(f"[SIM_INFO] not install triton skip patch it.")
