# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Sample Generate LLAMA"""
from mindspeed_llm import megatron_adaptor

import os
import sys
import time
import logging
import torch
import tqdm
from typing import Union

import megatron
from torch import distributed as dist

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, print_rank_0, get_model
from megatron.core.models.gpt import GPTModel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.checkpointing import load_checkpoint
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches,
)
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.moe.router import TopKRouter

from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM
from mindspeed_llm.tasks.evaluation.utils import add_text_generate_args
from mindspeed_llm.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from mindspeed_llm.tasks.posttrain.utils import train_valid_test_datasets_provider
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank, generate_actual_seq_len

import pdb
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

def model_provider(pre_process, post_process):
    """
    Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                    args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model

    
def save_router_cache(router_cache, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    
    for layer_id, data in router_cache.items():
        save_path = (
            f"{save_dir}/"
            f"dp{dp_rank}_pp{pp_rank}_layer{layer_id}_router_cache.pt"
        )

        torch.save( 
            {
                "indices": data["indices"],  # List[Tensor]
            },
            save_path
        )
        
def main():
    initialize_megatron(args_defaults={'no_load_rng': True,
                                    'no_load_optim': True})
    args = get_args()
    args.iteration = 0
    
    # Build model.
    model = get_model(model_provider, wrap_with_ddp=False)
            
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_world_size = mpu.get_pipeline_model_parallel_world_size()

    layers_per_rank = args.num_layers // pp_world_size
            
    if args.load:
        load_checkpoint(model, None, None, 'load')
    else:
        raise ValueError("Please specify the model checkpoint path with --load")
    
    model = model[0].module
    print_rank_0(f"========model: {model}")
    
    weight_records = defaultdict(dict)
    for idx, layer in enumerate(model.decoder.layers):
        global_layer_id = idx + pp_rank * layers_per_rank
        for name, param in layer.named_parameters():
            mean_val = float(param.data.mean().item())
            var_val = float(param.data.var().item())
            l2_norm = float(torch.norm(param.data, p=2).item())
            shape = list(param.data.shape)
            dtype = str(param.data.dtype)
            param_stats = {
                "mean": mean_val,
                "var": var_val,
                "l2_norm": l2_norm,
                "shape": shape,
                "dtype": dtype
            }
            weight_records[global_layer_id][name] = param_stats
    
    # rank0 gather and save
    pp_group = mpu.get_pipeline_model_parallel_group()
    gathered_records = [None for _ in range(pp_world_size)]
    
    # 将本地的 weight_records (defaultdict 转为 dict) 收集到所有 rank 的 gathered_records 列表中
    dist.all_gather_object(gathered_records, dict(weight_records), group=pp_group)
    
    # 仅由第一个数据并行副本的第一个 pipeline 阶段 (即全局 rank 0) 负责保存文件
    if mpu.get_pipeline_model_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
        import json
        all_weight_records = {}
        for record in gathered_records:
            if record:
                all_weight_records.update(record)
        
        save_path = "no-grouped-gemm-online-hif4-weight_statistics_no_fp32_hw.json"
        with open(save_path, "w") as f:
            json.dump(all_weight_records, f, indent=4)
        print_rank_0(f"Weight statistics saved to {save_path}")
        


if __name__ == "__main__":
    main()

