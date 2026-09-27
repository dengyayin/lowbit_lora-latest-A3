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
from transformers import AutoTokenizer

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, print_rank_0, get_model
from megatron.core.models.gpt import GPTModel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.utils import get_batch_on_this_cp_rank
from megatron.core.pipeline_parallel import get_forward_backward_func
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

def get_batch(data_iterator):
    """Generate a batch."""
    # Items and their type.
    keys = ['input_ids', 'attention_mask', 'labels']
    args = get_args()
    if args.reset_position_ids:
        keys += ['position_ids']
    data_type = torch.int64

    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
            tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
            return tokens, None, None, attention_mask, None
        else:
            # Broadcast data.
            data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
            if args.reset_position_ids:
                generate_actual_seq_len(data_b)
                batch = {'attention_mask': None}
            else:
                attention_mask_1d = data_b.get('attention_mask').long()
                attention_mask = get_tune_attention_mask(attention_mask_1d)
                batch = {'attention_mask': attention_mask}
            batch = get_batch_on_this_cp_rank(batch)
            return None, None, None, batch['attention_mask'], None

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

    # Unpack
    labels = data_b.get('labels').long()
    tokens = data_b.get('input_ids').long()
    attention_mask_1d = data_b.get('attention_mask').long()
    # ignored label -100

    if get_args().spec is not None and args.spec[0] == "mindspeed_llm.tasks.models.spec.hunyuan_spec":
        input_ids = tokens
        pad_id = 127961

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        attention_mask = input_ids.ne(pad_id)

        position_ids = None
        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': None,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
    else:
        if args.reset_position_ids:
            position_ids = data_b.get('position_ids').long()
            generate_actual_seq_len(data_b)

            batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': None,
            }
            batch = get_batch_on_this_cp_rank(batch)
            batch['attention_mask'] = None
            batch['position_ids'] = position_ids
            return batch.values()

        attention_mask = get_tune_attention_mask(attention_mask_1d)
        position_ids = None
        batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': None,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
    batch = get_batch_on_this_cp_rank(batch)
    # print_rank_0(f"========batch['tokens']: {batch['tokens']}, shape: {None if batch['tokens'] is None else batch['tokens'].shape}")
    return batch.values()
    
def forward_only_step_func(data_iterator, model):
    args = get_args()
    
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    
    if args.use_mcore_models:
        output_tensor = model(tokens, position_ids, attention_mask,
                            labels=labels, loss_mask=loss_mask)
    def noop_func(out, non_loss_data=False):
        if non_loss_data:
            return {}
        else:
            return out, torch.tensor(0., device=out.device)
        
    return output_tensor, noop_func
    
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
    print_rank_0(f"========args.is_instruction_dataset: {args.is_instruction_dataset}")
    
    # Build model.
    model = get_model(model_provider, wrap_with_ddp=False)
    print_rank_0(f"========model: {model}")
    for module in model:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
            
    router_cache = defaultdict(lambda: {"indices": []})
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_world_size = mpu.get_pipeline_model_parallel_world_size()

    layers_per_rank = args.num_layers // pp_world_size

    def make_hook(global_layer_id):
        def hook(module, inputs, output):
            scores, indices = output
            
            router_cache[global_layer_id]["indices"].append(
                indices.detach().cpu()
            )

        return hook
    
    for local_layer_idx, layer in enumerate(model[0].module.decoder.layers):
        global_layer_id = pp_rank * layers_per_rank + local_layer_idx
        for _, module in layer.named_modules():
            if isinstance(module, TopKRouter):
                module.register_forward_hook(
                    make_hook(global_layer_id)
                )
            
    if args.load:
        load_checkpoint(model, None, None, 'load')
    else:
        raise ValueError("Please specify the model checkpoint path with --load")
    
    # pdb.set_trace()
    # Build data iterators.
    train_valid_test_datasets_provider.is_distributed = True
    train_data_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
    num_micro_batch_size = get_num_microbatches()
    print_rank_0(f'num micro batch size: {num_micro_batch_size}')
    
    is_rank_0 = torch.distributed.get_rank() == 0
    if is_rank_0:
        pbar = tqdm.tqdm(total=args.train_iters, desc=f"Recording router selection")
            
    with torch.no_grad():
        iteration = 0
        while iteration < args.train_iters:
            iteration += 1
            forward_backward_func = get_forward_backward_func()
            loss = forward_backward_func(
                forward_step_func=forward_only_step_func,
                data_iterator=train_data_iterator,
                model=model,
                num_microbatches=num_micro_batch_size,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                collect_non_loss_data=True,
                forward_only=True,
            )
            if is_rank_0:
                pbar.update(1)
    
    if is_rank_0:
        pbar.close()
    
    # print_rank_0(f"========router_cache: {router_cache}")
    print_rank_0(f"========Saving router cache to {args.save} ...")
    save_router_cache(router_cache, save_dir=args.save)
    print_rank_0(f"========Save done.")


if __name__ == "__main__":
    main()

