# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
import apex
import torch.nn.functional as F
from megatron.training import get_args
from megatron.legacy.model.transformer import ParallelTransformer
from megatron.legacy.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from megatron.legacy.model.module import MegatronModule

CLASS_TOKEN_LENGTH = 8

class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Args:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, config, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.config = config
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if(x >= 0):
        sr = math.sqrt(x)
        return (int(sr) * int(sr) == x)
    return False


def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    args = get_args()
    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    hidden_size = args.hidden_size

    key = prefix + "weight"

    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert(isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - CLASS_TOKEN_LENGTH))
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - CLASS_TOKEN_LENGTH if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = args.class_token_present

        # update input_param and load it to state_dict[key]
        if input_has_class_token:
            input_param_tok = input_param[:CLASS_TOKEN_LENGTH, :]
            input_param_grid = input_param[CLASS_TOKEN_LENGTH:, :]
        else:
            input_param_tok = torch.zeros(CLASS_TOKEN_LENGTH, hidden_size)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:

            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert (
            input_param.shape[0] == num_tok_output
            and input_param.shape[1] == hidden_size
        )

        if output_has_class_token