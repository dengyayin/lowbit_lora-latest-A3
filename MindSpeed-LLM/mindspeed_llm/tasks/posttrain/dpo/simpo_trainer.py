# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
from typing import Dict, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from megatron.training import get_args
from megatron.core import mpu
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.dpo import DPOTrainer
from mindspeed_llm.tasks.posttrain.utils import vocab_parallel_log_softmax


class SimPOTrainer(BaseTrainer):
    """
    A trainer class for Simple Preference Optimization (SimPO).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """
    IGNORE_INDEX = -100

    def __init__(self):
        """
        Initializes the SimPOTrainer instance.

        Sets up the instance variables for the model provider and initializes the SimPO model.
        """
        super().__init__()

        self.args.actual_micro_batch_size = self.args.micro_batch_size * 2

    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """SimPO Loss function.

        Args:
            input_tensor (torch.Tensor): The tensor with the labels (repeated in double)
            output_tensor (torch.Tensor): The tensor with the Policy Model's Logits
        """
        args = get_args()

        all_policy_logits = output_tensor
        labels = input_tensor

        loss, metrics = self.get_batch_loss_metrics(
            all_policy_logits,
            labels
        )

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        metrics['lm loss'] = average_losses_across_data_parallel_group([loss])
        for key in metrics.keys():
            metrics[key] = average_losses_across_data_parallel_group([metrics[key]])

        return loss, metrics
    
    @staticmethod
    def get_batch(data_iterator):
        return DPOTrainer.get_batch(data_iterator)

    def forward_step(self, data_iterator, model):
        """SimPO Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, attention_mask, position_ids = self.get_batch(data_iterator)
        self.timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask)

        return output_tensor, partial(self.loss_func, labels)

    def simpo_loss(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        ) -> Tuple[torch.Tensor, ...]:
        """
        Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_log_probs:
            Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_log_probs:
            Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs

        logits = pi_log_ratios - self.args.gamma_beta_ratio

        # The beta is a temperature parameter for the SimPO loss.
        # The gamma_beta_ratio is a target reward margin to help separating the winning and losing responses.
        # The label_smoothing parameter encodes our uncertainty about the labels and calculates a conservative SimPO loss.
        if self.args.simpo_loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.args.simpo_beta * logits) * (1 - self.args.simpo_label_smoothing)
                - F.logsigmoid(-self.args.simpo_beta * logits) * self.args.simpo_label_smoothing
            )
        elif self.args.simpo_loss_type == "hinge":
            losses = torch.relu(1 - self.args.simpo_beta * logits)
        elif self.args.simpo_loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter
            # for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.args.simpo_beta)) ** 2
        else:
            raise ValueError(
                f"Unknown loss type: {self.args.simpo_loss_type}."
                f" Should be one of ['sigmoid', 'hinge', 'ipo']"
            )

        chosen_rewards = (self.args.simpo_beta * policy_chosen_log_probs.detach())
        
        rejected_rewards = (self.args.simpo_beta * policy_rejected_log_probs.detach())

        return losses, chosen_rewards, rejected_rewards

    def compute_preference_loss(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the preference loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_log_probs: Log probabilities of the policy model for the chosen responses.
            policy_rejected_log_probs: Log probabilities of the policy model for the rejected responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the preference loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs
        )
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
            self,
            all_policy_logits,
            label
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Computes the sum log probabilities of the labels under the given logits.

        Otherwise, the average log probabilities.

        Args:
            all_policy_logits: Logits of the policy model.
            all_reference_logits: Logits of the reference model.
            label: The label tensor.

        Returns:
            A tuple containing the loss tensor and a dictionary of metrics.
        """
        metrics = {}

        (
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            policy_chosen_log_probs_avg,
        ) = self._compute_log_probs(all_policy_logits, label)


        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
        )

        sft_loss = -policy_chosen_log_probs_avg
        if self.args.pref_ftx > 1e-6:
            losses += self.args.pref_ftx * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = ""
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.detach().mean()
        if self.args.simpo_loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean()
            metrics["{}odds_ratio_loss".format(prefix)] = (
                    (losses - sft_loss) / self.args.simpo_beta).detach().mean()

        return losses.mean(), metrics


    def _compute_log_probs(self, all_logits, label) -> Tuple[torch.Tensor, ...]:
        """
        Computes the sum log probabilities of the labels under given logits if loss_type.
        Otherwise, the average log probabilities.
        Assuming IGNORE_INDEX is all negative numbers, the default is -100.

        Args:
            all_logits: The logits tensor.
            label: The label tensor.

        Returns:
            A tuple containing the log probabilities and other tensors.
        """
  
        label = label[:, 1:].clone()
        all_logits = all_logits[:, :-1, :]
        batch_size = all_logits.size(0) // 2


        all_log_probs, valid_length = self._get_batch_log_probs(
            all_logits,
            label
        )
        
        all_log_probs = all_log_probs / torch.clamp(valid_length, min=1)

        chosen_log_probs, rejected_log_probs = all_log_probs.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        all_results = (chosen_log_probs, rejected_log_probs, chosen_log_probs / chosen_length)

        return all_results

    def _get_batch_log_probs(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """ 
        Computes the log probabilities of the given labels under the given logits.

        In the tensor parallelism case, it takes into account the vocab parallelism and
        performs the necessary adjustments to the labels and logits.

        Args:
            logits: The logits tensor.
            labels: The label tensor.

        Returns:
            A tuple containing the log probabilities and the valid length.
        """
        if mpu.get_tensor_model_parallel_world_size() > 1:
            tp_vocab_size = logits.size(2)

            labels -= mpu.get_tensor_model_parallel_rank() * tp_vocab_size
            labels = labels.masked_fill(torch.logical_or(labels < 0, labels >= tp_vocab_size), 0)
            loss_mask = labels != 0

            per_token_log_probs = torch.gather(
                vocab_parallel_log_softmax(logits), dim=2, index=labels.unsqueeze(2)).squeeze(2)

            all_log_probs = (per_token_log_probs * loss_mask).sum(-1)
            valid_length = loss_mask.sum(-1)

            torch.distributed.all_reduce(
                all_log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_tensor_model_parallel_group()
            )

            torch.distributed.all_reduce(
                valid_length,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_tensor_model_parallel_group()
            )

        else:
            label_pad_token_id = self.IGNORE_INDEX
            loss_mask = labels != label_pad_token_id
            labels[labels == label_pad_token_id] = 0  # dummy token
            per_token_log_probs = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            all_log_probs = (per_token_log_probs * loss_mask).sum(-1)
            valid_length = loss_mask.sum(-1)
            
        if mpu.get_context_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                valid_length,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_context_parallel_group()
            )

            torch.distributed.all_reduce(
                all_log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_context_parallel_group()
            )

        return all_log_probs, valid_length