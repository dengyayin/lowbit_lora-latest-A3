"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from functools import partial

import torch
import torch.nn.functional as F

from .intel_ops import quantize_fp4
from .fp4_ops import quantize_afp4, quantize_afp4_protect, To_HiFX, To_NVFP4, quantize_hif4_kernel, quantize_nvf4_kernel, quantize_nvf4_scaled_kernel, quantize_nvf4_to_hif4_kernel, quantize_nvf4s_to_hif4_kernel
from .mx_ops import quantize_mx_op, quantize_mx_op_protect, quantize_nv4_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test


f_linear = F.linear
torch_matmul = torch.matmul

from typing import Optional, Tuple
from .fp8_ops import quantize_fp4_, quantize_fp8, quantize_fp4_vanila, quantize_mxfp4_, quantize_mxfp4_vanila, quantize_fp4_mocha, quantize_mxfp4_mocha, quantize_mxfp4_mocha_

# import flashinfer
# from flashinfer.fp4_quantization import fp4_quantize
import pdb
# from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float
# from flashinfer.fp4_quantization import mxfp4_quantize, mxfp4_dequantize
# from flashinfer import nvfp4_quantize, mm_fp4, SfLayout

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
        perm=None,
        clip=1.0,
    ):
        mx_specs_fp8 = {
            'block_size': 32,
            'fp': 16,
            'bfloat': -1,
            # 'bfloat': 16,
            # 'fp': -1,
            'w_elem_format': 'fp8_e4m3',
            'a_elem_format': 'fp8_e4m3', 
            'custom_cuda': True,
            # For quantization-aware finetuning, do backward pass in FP32
            'scale_bits': 8,
            'shared_exp_method': 'max',
            'quantize_backprop': False,
            "mx_flush_fp32_subnorms": True,
            'bfloat_subnorms': True,
            "round_output": "nearest",    
            "round_mx_output": "nearest",        
            "round_weight": "nearest",
            "round_output": "nearest",     
        }
        
        # quantize_mx_op(key_states.to(torch.float32), mx_specs=mx_specs_fp8, elem_format='fp8_e4m3', axes=[-1], round='nearest', expand_and_reshape=False)
        quantize_map = {
            'fp8': (quantize_fp8, {'method': act_method, 'fp8_format': torch.float8_e4m3fn}),
            'fp8_wei': (quantize_fp8, {'method': "per-channel", 'fp8_format': torch.float8_e4m3fn}),
            'fp4': (quantize_fp4, {'num_bits': 4, 'group_size': mx_specs['block_size'], 'scheme': 'asym', 'quantile': 1.0, 'data_type': 'fp4_e2m1', 'return_int': False, 'full_range': False}),
            'fp4_': (quantize_fp4_, {'group_size': mx_specs['block_size'], 'first_method': "per-channel"}),
            'mxfp4': (quantize_mx_op, {'mx_specs': mx_specs, 'elem_format': 'fp4_e2m1', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False, 'clip': clip}),
            'mxfp6_e3m2': (quantize_mx_op, {'mx_specs': mx_specs, 'elem_format': 'fp6_e3m2', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False, 'clip': clip}),
            'mxfp6_e2m3': (quantize_mx_op, {'mx_specs': mx_specs, 'elem_format': 'fp6_e2m3', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False, 'clip': clip}),
            'mxfp4_e1m2': (quantize_mx_op, {'mx_specs': mx_specs, 'elem_format': 'fp4_e1m2', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False, 'clip': clip}),
            'int6': (quantize_mx_op, {'mx_specs': mx_specs, 'elem_format': 'int6', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False, 'clip': clip}),

            'mxfp8': (quantize_mx_op, {'mx_specs': mx_specs_fp8, 'elem_format': 'fp8_e4m3', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False}),
            'nvfp4_scale_mx': (quantize_nv4_op, {'mx_specs': mx_specs, 'elem_format': 'fp4_e2m1', 'axes': [-1], 'round': 'nearest', 'expand_and_reshape': False}),

            'hifp4': (quantize_hif4_kernel, {'N': 4}),
            'nvfp4': (quantize_nvf4_kernel, {'N': 4}),

            'nvfp4tohif4': (quantize_nvf4_to_hif4_kernel, {'N': 4}),
            'nvfp4tohif4s': (quantize_nvf4s_to_hif4_kernel, {'N': 4}),

            'nvfp4_scale': (quantize_nvf4_scaled_kernel, {'N': 4}),

            'fp4_v': (quantize_fp4_vanila, {'group_size': mx_specs['block_size']}),
            'fp4_m': (quantize_fp4_mocha, {'group_size': mx_specs['block_size']}),
            
            'mxfp4_': (quantize_mxfp4_, {'first_method': act_method, 'mx_specs': mx_specs}),
            'mxfp4_vanila': (quantize_mxfp4_vanila, {'mx_specs': mx_specs}),
            'mxfp4_mocha': (quantize_mxfp4_mocha, {'mx_specs': mx_specs}),
            'mxfp4_mocha_': (quantize_mxfp4_mocha_, {'mx_specs': mx_specs}),
            
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
        if perm is not None:
            # import pdb;pdb.set_trace()
            # print("Find Permutation")
            # print(perm)
            bf_in = bf_in[:, :, perm]


        # bf_in_ = bf_in.clone()
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
        # inp = bf_in.clone()

        #-------------------Caculate--------------------#
        qis_weight = (wei_func(bf_weight.to(torch.float32))).to(torch.float16)
        qis_input_res = (act_func(bf_in.to(torch.float32)))
        if isinstance(qis_input_res, tuple) and len(qis_input_res) == 2:
            # 如果是两个输出，说明有scale，则计算
            qis_input_, scale = qis_input_res
            qis_input = (qis_input_ * scale).to(torch.float16)
        else:
            # 如果是一个输出，则直接使用
            qis_input = qis_input_res.to(torch.float16)
            
        # mse_act = torch.nn.MSELoss()(inp, qis_input)
        mse_act = 0

        # mse = torch.nn.MSELoss()(qis_input, bf_in_)
        # print("mse:", mse)
        #-------------------Caculate--------------------#

        if bias is not None:
            # print("!!!BIAS")
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
        # qis_input = quantize_mx_op(
        #     bf_in.to(torch.float32),
        #     mx_specs,
        #     elem_format=mx_specs['a_elem_format'],
        #     axes=[-1],
        #     round=mx_specs["round_mx_output"],
        # ).to(torch.float16)

        # qis_weight = quantize_mx_op(
        #     bf_weight.to(torch.float32),
        #     mx_specs,
        #     elem_format=mx_specs['w_elem_format'],
        #     axes=[-1],
        #     round=mx_specs["round_mx_output"],
        # ).to(torch.float16)

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
        output = F.linear(qis_input, qis_weight.to(qis_input))
        # output = quantize_elemwise_op(
        #     output, mx_specs=mx_specs, round=mx_specs["round_output"]
        # )

        if bias is not None:
            # print("!!!BIAS!!!")
            output = output + bf_bias
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.name = name

        # print("MX OUTPUT RANGE: ", output.max().item(), output.min().item())
        # print("MX OUTPUT: \n", output)
        
        return output, mse_act
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
    perm=None,
    clip=1.0,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_linear(input, weight, bias=bias)

    mx_specs = apply_mx_specs(mx_specs)

    return LinearFunction.apply(input, weight, bias, mx_specs, act_method, act_func, wei_func, name, perm, clip)

class LinearMX(torch.nn.Linear):
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
        perm=None,
        clip=1.0,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)
        self.act_method = act_method
        self.act_func = act_func
        self.wei_func = wei_func
        self.perm = perm
        self.clip = clip        
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
        # print("inputs.shape, ", self.name, inputs.shape)
        output, mse_act =  linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            mx_specs=self.mx_specs,
            act_method = self.act_method,
            act_func = self.act_func,
            wei_func = self.wei_func,
            name=self.name,
            perm=self.perm,
            clip=self.clip,
        )

        # self.weight_shared_exp = weight_shared_exp
        # self.act_shared_exp = act_shared_exp
        # self.mse_w = mse_wei
        self.mse_a = mse_act
        # print("output.shape, ", self.name, output.shape)

        # return (output, mse_act)
        return output


# class FPLinear(torch.nn.Linear):
#     def __init__(
#         self,
#         in_features,
#         out_features,
#         bias=True,
#         name=None,
#         clip=1.0,
#     ):
#         self.name = name
#         self.clip = clip
#         super().__init__(in_features, out_features, bias)
#
#     def set_weight_bias(self, weight, bias):
#         print("Quantizing Weight MXFP4 FlashInferrrrrrrr, weight.dtype", weight.dtype)
#         # w_fp4, w_sf = mxfp4_quantize(weight)
#         # self.weight = torch.nn.Parameter(w_fp4, requires_grad=False)
#         # self.weight_scale = torch.nn.Parameter(w_sf, requires_grad=False)
#         # print("Linear: ", self.weight)
#         self.bias = bias
#
#     def append_name(self, postfix):
#         self.name += postfix
#
#     def forward(self, inputs):
#
#         xdevice = inputs.device
#         batch_size = inputs.shape[0]
#         inputs = inputs.reshape(-1, inputs.shape[-1])
#         # print("inputs.device", inputs.device)
#
#         # pdb.set_trace()
#
#         # a_fp4, a_sf = mxfp4_quantize(inputs) # act quant # qact: uint8 scale:e8m0 uint8
#
#         output = mm_fp4(  # matmul
#             a_fp4,
#             self.weight.T,
#             a_sf,
#             self.weight_scale.T,
#             alpha=None,
#             out_dtype=torch.float16,
#             out=None,
#             block_size=32,
#             use_8x4_sf_layout=False,
#             backend='cudnn',
#             use_nvfp4=False
#             ).to(xdevice)
#         output = output.reshape(batch_size, -1 ,output.shape[-1])
#         # print("output.device", output.device)
#
#         # output = mm_fp4(a_fp4,self.weight.T,a_sf,self.weight_scale.T,alpha=None,out_dtype=torch.bfloat16,out=None,block_size=32,use_8x4_sf_layout=False,backend='cudnn',use_nvfp4=False)
#         return output

