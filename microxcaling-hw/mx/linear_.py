"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from functools import partial

import torch
import torch.nn.functional as F

from .intel_ops import quantize_fp4
from .mx_ops import quantize_mx_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test


f_linear = F.linear
torch_matmul = torch.matmul

from typing import Optional, Tuple

from .fp8_ops import quantize_fp4_, quantize_fp8

def quantize_none(t):
    return t

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        mx_specs=None,
        act_method=None,
        act_func=None,
        wei_func=None,
        name=None,
    ):

        quantize_map = {
            'fp8': (quantize_fp8, {'method': act_method, 'fp8_format': torch.float8_e4m3fn}),
            'fp4': (quantize_fp4, {'num_bits': 4, 'group_size': mx_specs['block_size'], 'scheme': 'asym', 'quantile': 1.0, 'data_type': 'fp4_e2m1', 'return_int': False, 'full_range': False}),
            'fp4_': (quantize_fp4_, {'group_size': mx_specs['block_size'], 'first_method': "per-channel"}),
            'mxfp4': (quantize_mx_op, {'mx_specs': mx_specs, 'elem_format': 'fp4_e2m1', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False}),
            'mxfp8': (quantize_mx_op, {'mx_specs': mx_specs, 'elem_format': 'fp8_e4m3', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False}),
            'fp16': (lambda x: x, {})
        }
        
        def set_quantize_func(func_name, func_type):
            quantize_func, kwargs = quantize_map[func_name]
            # if func_type == 'activation':
            #     pass
            return partial(quantize_func, **kwargs)
        
        wei_func = set_quantize_func(wei_func, 'weight')
        act_func = set_quantize_func(act_func, 'activation')
        
        bf_in = input
        bf_weight = weight
        
        # print("MX INPUT RANGE: ", input.max().item(), input.min().item())
        # print("MX INPUT: \n", input)
        # print("MX WEIGHT RANGE: ", weight.max().item(), weight.min().item())
        # print("MX WEIGHT: \n", weight)

        # element-wise quantize for input
        # bf_in = quantize_elemwise_op(
        #     input, mx_specs=mx_specs, round=mx_specs["round_output"]
        # )
        # # element-wise quantize for weight and bias
        # bf_weight = quantize_elemwise_op(
        #     weight, mx_specs=mx_specs, round=mx_specs["round_weight"]
        # )

        # mse_input = torch.nn.MSELoss()(input, bf_in)
        # mse_weight = torch.nn.MSELoss()(weight, bf_weight)

        # print("***MSE***:", mse_input, mse_weight)

        #-------------------Caculate--------------------#
        qis_weight = wei_func(bf_weight)
        qis_input = act_func(bf_in)

        if bias is not None:
            print("!!!BIAS")
            ctx.has_bias = True
            bf_bias = quantize_elemwise_op(
                bias, mx_specs=mx_specs, round=mx_specs["round_weight"]
            )
        else:
            ctx.has_bias = False
            bf_bias = None

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)        

        #----------------------FOR MX PRINT------------------------*
        # MX quantize everything along input size
        # print("****************QUANTIZING****************")
        # qis_input, act_shared_exp, mse_act = quantize_mx_op(
        #     bf_in.to(torch.float32),
        #     mx_specs,
        #     elem_format=mx_specs['a_elem_format'],
        #     axes=[-1],
        #     round=mx_specs["round_mx_output"],
        # )
        # qis_input = qis_input.to(torch.float16)

        # qis_weight, weight_shared_exp, mse_wei = quantize_mx_op(
        #     bf_weight.to(torch.float32),
        #     mx_specs,
        #     elem_format=mx_specs['w_elem_format'],
        #     axes=[-1],
        #     round=mx_specs["round_mx_output"],
        # )
        # qis_weight = qis_weight.to(torch.float16)

        # min_val, max_val = act_shared_exp.aminmax()
        # print("act mse: ", mse_act)
        # print("act maxmin: ", max_val, min_val)
        
        # max_value = np.max(weight_shared_exp)
        # min_value = np.min(weight_shared_exp)
        # min_val, max_val = weight_shared_exp.aminmax()
        # print("wei mse: ", mse_wei)
        # print("wei maxmin: ", max_val, min_val)   

        # print("MX QINPUT RANGE: ", qis_input.max().item(), qis_input.min().item())
        # print("MX QINPUT: \n", qis_input)
        # print("MX QWEIGHT RANGE: ", qis_weight.max().item(), qis_weight.min().item())
        # print("MX QWEIGHT: \n", qis_weight)
        
        # print("MSE:", torch.nn.MSELoss()(qis_input, input), torch.nn.MSELoss()(qis_weight, weight) )
        
        # compute output
        output = F.linear(qis_input, qis_weight)
        # output = quantize_elemwise_op(
        #     output, mx_specs=mx_specs, round=mx_specs["round_output"]
        # )

        if bias is not None:
            print("!!!BIAS!!!")
            output = output + bf_bias
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.name = name

        # print("MX OUTPUT RANGE: ", output.max().item(), output.min().item())
        # print("MX OUTPUT: \n", output)
        
        return output
        # return output, weight_shared_exp, act_shared_exp, mse_wei, mse_act

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        input, weight = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        grad_output = quantize_elemwise_op(
            grad_output,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_weight, grad_bias
        #####################################################
        # if the input is 2D, quantize everything along examples (batches)
        # if the input is 3D, quantize everything along the first axis
        qex_input = quantize_mx_op(
            input,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_weight"],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_weight"],
        )

        # compute grad_weight [out_features, in_features]
        qex_grad_output = qex_grad_output.reshape(-1, out_dim)
        qex_input = qex_input.reshape(-1, in_dim)

        # Compute grad_weight
        grad_weight = torch_matmul(qex_grad_output.transpose(0, 1), qex_input)
        grad_weight = quantize_elemwise_op(
            grad_weight,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_weight"],
        )

        #####################################################
        # perform madtile operation for grad_input
        #####################################################
        # compute grad_input, quantize everything along output size
        qos_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['w_elem_format_bp'],
            axes=[0],
            round=ctx.mx_specs["round_mx_weight_grad_input"],
        )
        # grad_output shape is (B, seq, out_dim)
        qos_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_os'],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # Compute grad_input
        grad_input = torch_matmul(qos_grad_output, qos_weight)
        grad_input = quantize_elemwise_op(
            grad_input,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_input, grad_weight, grad_bias, None, None, None, None)


def linear(
    input,
    weight,
    bias=None,
    mx_specs=None,
    act_method=None,
    act_func=None,
    wei_func=None,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_linear(input, weight, bias=bias)

    mx_specs = apply_mx_specs(mx_specs)

    return LinearFunction.apply(input, weight, bias, mx_specs, act_method, act_func, wei_func, name)

class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mx_specs=None,
        name=None,
        weight_shared_exp=None,
        act_shared_exp=None,
        mse_a = None,
        mse_w = None,
        act_method = None,
        act_func = None,
        wei_func=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)
        self.act_method = act_method
        self.act_func = act_func
        self.wei_func = wei_func
        super().__init__(in_features, out_features, bias)

    def set_weight_bias(self, weight, bias):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        # print("Linear: ", self.weight)
        self.bias = bias

    def apply_mx_specs(self, mx_specs):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        # if self.mx_none:
        #     return super().forward(inputs)
        # output, weight_shared_exp, act_shared_exp, mse_wei, mse_act =  linear(
        output =  linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            mx_specs=self.mx_specs,
            act_method = self.act_method,
            act_func = self.act_func,
            wei_func = self.wei_func,
            name=self.name,
        )

        # self.weight_shared_exp = weight_shared_exp
        # self.act_shared_exp = act_shared_exp
        # self.mse_w = mse_wei
        # self.mse_a = mse_act
        
        return output
