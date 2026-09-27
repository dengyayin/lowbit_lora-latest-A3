# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.training import get_args


def pai_megatron_aux_loss(self, logits: torch.Tensor):
    routing_weights = torch.softmax(logits, dim=1, dtype=torch.float32).type_as(logits)
    scores, indices = torch.topk(routing_weights, k=self.topk, dim=-1)

    # TopK without capacity
    num_experts = logits.shape[1]
    tokens_per_expert = torch.histc(indices, bins=num_experts, min=0, max=num_experts)

    # Apply load balancing loss
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    scores = self.apply_load_balancing_loss(probs, tokens_per_expert, activation=scores)

    args = get_args()
    global_indices = indices
    if args.moe_token_dispatcher_type == "allgather":
        if args.moe_permutation_async_comm and (
                self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1)):
            from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
            with torch.no_grad():
                global_indices = gather_from_sequence_parallel_region_to_moe_async(indices)
    return scores, global_indices

