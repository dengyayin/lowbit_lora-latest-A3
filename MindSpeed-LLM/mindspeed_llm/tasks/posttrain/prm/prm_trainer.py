# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from typing import Union
from functools import partial
import torch
import torch.nn as nn
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args, get_tokenizer
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.utils import convert_token_to_id
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank


class PRMTrainer(BaseTrainer):
    """
    A trainer class for Process Reward Model (PRM).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """

    def __init__(self):
        """
        Initializes the PRMTrainer instance.

        Sets up the instance variables for the model provider, actual micro batch size,
        and initializes the PRM model.
        """
        super().__init__()
        
        args = get_args()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.tokenizer = get_tokenizer().tokenizer
        # set placeholder token
        self.placeholder_token_id = convert_token_to_id(args.placeholder_token, self.tokenizer)
        self.reward_token_ids = args.reward_tokens
        if self.reward_token_ids is not None:
            self.reward_token_ids = sorted(
                [convert_token_to_id(token, self.tokenizer) for token in self.reward_token_ids]
            )

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""

        args = get_args()

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)

                return tokens, None, None, attention_mask, None
            else:
                return None, None, None, None, None
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

        attention_mask = get_tune_attention_mask(attention_mask_1d)

        return tokens, labels, loss_mask, attention_mask, None


    def loss_func(self, input_ids: torch.Tensor, labels: torch.Tensor, output_tensor: torch.Tensor):
        """PRM Loss function.
        """
        placeholder_mask = input_ids == self.placeholder_token_id

        output_tensor = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(output_tensor)
        logits = output_tensor[placeholder_mask]
        labels = labels[placeholder_mask]

        if self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.cross_entropy_loss(logits, labels)
        averaged_loss = average_losses_across_data_parallel_group([loss])
        
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()

        return loss * self.args.context_parallel_size, {'lm loss': averaged_loss[0], 'acc': acc}


    def forward_step(self, data_iterator, model):
        """PRM Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, _, attention_mask, position_ids = self.get_batch(
            data_iterator)
        self.timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask)

        return output_tensor, partial(self.loss_func, tokens, labels)