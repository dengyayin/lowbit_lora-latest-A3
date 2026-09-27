# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from typing import Union
from functools import partial
import logging
import torch
import torch.distributed as dist
from megatron.training import get_args, print_rank_0
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core import mpu, tensor_parallel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.core.models.gpt import GPTModel
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.orm.orm_model import GPTRewardModel
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank

logger = logging.getLogger(__name__)


class ORMTrainer(BaseTrainer):
    """
    A trainer class for Outcome Reward Model (ORM).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """

    def __init__(self):
        """
        Initializes the ORMTrainer instance.

        Sets up the instance variables for the model provider, actual micro batch size,
        and initializes the RM model.
        """
        super().__init__()

        # Similar to dpo, set actual_micro_batch_size to change the recv/send shape when using PP
        self.args.actual_micro_batch_size = self.args.micro_batch_size * 2

    @staticmethod
    def model_provider(pre_process=True, post_process=True):
        """Builds the model.

        Currently supports only the mcore GPT model.

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

        if not args.use_mcore_models:
            raise ValueError("Outcome Reward model training currently supports mcore only.")
        if (not args.untie_embeddings_and_output_weights) and (args.pipeline_model_parallel_size > 1):
            args.untie_embeddings_and_output_weights = True
            logger.warning(
                "untie_embeddings_and_output_weights is set to True, "
                "since output_layer is not used in Outcome Reward model training."
            )
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                    args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTRewardModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            post_layer_norm=not args.no_post_layer_norm,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

        return model

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""

        args = get_args()

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                return tokens, None, None, attention_mask, None
            else:
                batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        data_type = torch.int64

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        loss_mask = attention_mask_1d

        # ORM uses only the last token to compute loss, so also keeps only the last 1 in loss mask,
        # for example, [1, 1, 1, 1, 1, 1, 0, 0] -> [0, 0, 0, 0, 0, 1, 0, 0]
        last_token_position = torch.sum(loss_mask, dim=1) - 1
        loss_mask_new = torch.zeros_like(loss_mask)
        fill = torch.ones(loss_mask.size(0)).to(loss_mask.dtype).to(loss_mask.device)
        loss_mask_new.scatter_(1, last_token_position.unsqueeze(1), fill.unsqueeze(1))
        loss_mask = loss_mask_new

        attention_mask = get_tune_attention_mask(attention_mask_1d)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': None
        }
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    @staticmethod
    def loss_func(input_ids: torch.Tensor, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """RM Loss function.
        """
        args = get_args()

        batch_size = input_ids.size(0) // 2
        chosen_masks, rejected_masks = torch.split(loss_mask, batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(output_tensor, batch_size, dim=0)
        chosen_rewards, rejected_rewards = chosen_rewards.squeeze(-1), rejected_rewards.squeeze(-1)
        chosen_scores = (chosen_masks * chosen_rewards).sum(dim=1)
        rejected_scores = (rejected_masks * rejected_rewards).sum(dim=1)
        if args.context_parallel_size > 1:
            dist.all_reduce(chosen_scores, group=mpu.get_context_parallel_group())
            dist.all_reduce(rejected_scores, group=mpu.get_context_parallel_group())

        loss = -torch.log(torch.sigmoid(chosen_scores.float() - rejected_scores.float())).mean() 
        with torch.no_grad():
            acc = (chosen_scores > rejected_scores).sum() / len(chosen_scores)
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss * args.context_parallel_size, {'lm loss': averaged_loss[0], 'acc': acc}


    def forward_step(self, data_iterator, model):
        """RM Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)
        self.timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask)

        return output_tensor, partial(self.loss_func, tokens, loss_mask)
    