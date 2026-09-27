import torch
import torch.nn as nn
# from .ant_quant import generate_quant_grid
# from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

import logging
import time
import pdb
import numpy as np


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, input_x, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                        
                # # RTN quant
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                
                # # sec quant
                # q = self.secquant(weight=w.unsqueeze(0),input_x=input_x, group_size=-1, n_bit=4).squeeze(0)
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')
        

def split_last_dim_into_blocks(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    *dims, last_dim = x.shape  # 支持任意前面的维度，如 B, S, ...
    assert last_dim % block_size == 0, (
        f"最后一维大小 {last_dim} 必须能被 block_size={block_size} 整除"
    )
    num_blocks = last_dim // block_size
    return x.view(*dims, num_blocks, block_size)


def quant_with_scale(W, scales, quant_grid):
    labels = ((W / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    W_deq = quant_grid[labels] * scales
    return W_deq

def gptq_quant(
    weight, input_x, blocksize=32, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
):
    actorder = True
    
    quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    
    quant_grid = torch.tensor(quant_grid).to(dtype=torch.half).to(weight.device)
    
    W = weight.clone()
    W = W.float()
    
    columns = W.shape[1]

    tick = time.time()


    # compute Hessian
    
    H = torch.zeros((columns, columns), device=W.device)
    batch_size, seq_len, hidden_dim = input_x.shape
    
    inp = input_x.clone().detach().reshape((-1, input_x.shape[-1]))
    inp = inp.t()
    inp = math.sqrt(2 / batch_size) * inp.float()
    H += inp.matmul(inp.t())

    # H = self.H
    # del self.H
    
    
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # if static_groups:
    #     import copy
    #     groups = []
    #     for i in range(0, columns, groupsize):
            
    #         # find quantization parameters
    #         quantizer = copy.deepcopy(self.quantizer)
    #         quantizer.find_params(W[:, i:(i + groupsize)])
    #         groups.append(quantizer)

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        
        input_xx = input_x.clone().detach().reshape((-1, input_x.shape[-1]))[:, perm]
        
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=W.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        
        # find quantization param
        x = input_xx.reshape((-1, input_x.shape[-1]))
        _, _, _, scales = get_quant_weight_mxfp(W1.half(), input_x=x[:,i1*32:(i1+1)*32], quant_grid=quant_grid, zero_point=False, round_method="w_search")

        scales = scales.to(W.device)
        
        
        for i in range(count):
            w = W1[:, i]
            
            d = Hinv1[i, i]

            # if groupsize != -1:
            #     if not static_groups:
            #         if (i1 + i) % groupsize == 0:
            #             self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
            #     else:
            #         idx = i1 + i
            #         if actorder:
            #             idx = perm[idx]
            #         self.quantizer = groups[idx // groupsize]
                    
            # # RTN quant
            q = quant_with_scale(w.unsqueeze(1), scales, quant_grid)
            
            # print(q.shape, flush=True)
            # print(scales.shape, flush=True)
            # print(w.shape, flush=True)
            # exit()
            
            # # sec quant
            # q = self.secquant(weight=w.unsqueeze(0),input_x=input_x, group_size=-1, n_bit=4).squeeze(0)
            
            q = q.squeeze(1)
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    torch.cuda.synchronize()

    if actorder:
        Q = Q[:, invperm]

    weight = Q.reshape(weight.shape).to(weight.dtype)
    if torch.any(torch.isnan(weight)):
        logging.warning('NaN in weights')
        # import pprint
        # pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
        raise ValueError('NaN in weights')
    return weight, perm
    

def error_diffusion_quant(w, ed_quant_grid, calib_input):
    K = w.shape[1]
    N = w.shape[0]
    input = calib_input.reshape(-1, calib_input.shape[-1])
    
    M = input.shape[0]
    group_size = 32
    group_num = K / group_size
    
    w_quant = torch.zeros_like(w).to(w.device)
    
    # print((input == 0).to(torch.int32).sum(dim=0).tolist())
    # exit()
    
    
    # derive delta_O = delta_X * W
    
    q_x = torch.zeros_like(input).to(input.device)
    for i1 in range(0, K, group_size):
        i2 = min(i1 + group_size, K)
        q_x[:,i1:i2], _, _, _ = get_quant_act_mxfp(x=input[:,i1:i2], weight=None, quant_grid=ed_quant_grid, zero_point=False, round_method="up")
        
    delta_x = input - q_x
    delta_O_global = torch.mm(delta_x, w.T)
        
    U = torch.zeros(size=(M,N), dtype=w.dtype, device=w.device)
    
    for i1 in range(0, K, group_size):
        i2 = min(i1 + group_size, K)
        
        w_slice = w[:, i1:i2]
        x_slice = input[:, i1:i2]
        
        x_slice_q, _, _, _ = get_quant_act_mxfp(x=x_slice, weight=None, quant_grid=ed_quant_grid, zero_point=False, round_method="up")
        delta_x = x_slice - x_slice_q
        
        x_slice_q_pow = torch.pow(x_slice_q.to(torch.float32), 2)
        x_slice_q_squ_sum = x_slice_q_pow.sum(dim=0, keepdim=True)
        
        zero_num = (x_slice_q_squ_sum == 0).to(torch.int32).sum(dim=1).item()
        
        # delta_O = torch.mm(delta_x, w_slice.T)
        E = U + delta_O_global / group_num
        # E = delta_O_global / group_num + delta_O + U
        
        l_update = E / (group_size - zero_num)
        
        
        assert torch.isinf(x_slice_q_squ_sum).to(torch.int32).sum(0).sum(0) == 0, "overflow!"
        assert torch.isnan(x_slice_q_squ_sum).to(torch.int32).sum(0).sum(0) == 0, "x_square NAN!"
        
        
        # print(x_slice_q_squ_sum)

        
        x_slice_norm = x_slice.to(torch.float32) / (x_slice_q_squ_sum)

        # print(x_slice_q_squ_sum)
        # print(x_slice_norm[:,13].tolist())
        
        
        if torch.isinf(x_slice_norm).to(torch.int32).sum(0).sum(0) != 0:    
            x_slice_norm = torch.where(torch.isinf(x_slice_norm), torch.zeros_like(x_slice_norm).to(x_slice_norm.device), x_slice_norm)
        if torch.isnan(x_slice_norm).to(torch.int32).sum(0).sum(0) != 0:
            x_slice_norm = torch.where(torch.isnan(x_slice_norm), torch.zeros_like(x_slice_norm).to(x_slice_norm.device), x_slice_norm)
        
        
        # print(x_slice_norm[:,13].tolist())
        
        assert torch.isnan(x_slice_norm).to(torch.int32).sum(0).sum(0) == 0, "x_norm NAN"
        
        # print(torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32)))
        w_slice_new = w_slice + torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32)).to(torch.float16)
        # if i1==5024:
        #     # print(torch.isnan(torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32))).any())
        #     print(torch.isinf(l_update).any())
        #     print(torch.isinf(x_slice_norm).any())
        #     tmp = torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32))
        #     print(torch.isinf(tmp.to(torch.float16)).any())
        #     print(tmp.amax(dim=0).amax(dim=0))
        #     max_idx = torch.argmax(tmp)
        #     print(torch.argmax(tmp))
            
        #     print(tmp[max_idx//tmp.shape[1]][max_idx%tmp.shape[1]])
        #     print(w_slice[max_idx//tmp.shape[1]][max_idx%tmp.shape[1]])
        #     print(torch.isinf(w_slice_new).any())
        #     print("\n")
        assert torch.isnan(w_slice_new).any() == False, f"{i1} weight NAN"
        # print(w_slice_new)
        
        w_slice_quant, _, _, _ = get_quant_weight_mxfp(w_slice_new, quant_grid=ed_quant_grid, input_x=x_slice_q, zero_point=False, round_method="w_search")
        
        # U = U + torch.mm(x_slice_q, (w_slice - w_slice_quant).T) + delta_O_global / group_num
        U = U + torch.mm(x_slice_q, (w_slice - w_slice_quant).T) + delta_O_global / group_num
        
        if torch.isnan(w_slice_quant).any() or torch.isinf(w_slice_quant).any():
            print(torch.isnan(w_slice_quant).any())
            print(torch.isinf(w_slice_quant).any())
            print(i1)
        
        w_quant[:,i1:i2] = w_slice_quant
        
    origin_output = torch.mm(input, w.T)
    ed_output = torch.mm(q_x, w_quant.T)
    
    normal_deq_weight = torch.zeros_like(w).to(w.device)
    
    for i1 in range(0, K, group_size):
        i2 = min(i1 + group_size, K)
        # deq_weight[i*N : (i+1)*N, :], _, up_ratio = get_quant_weight_mxfp(weight[i*N : (i+1)*N, :], input_x=input[i*M : (i+1)*M, :], quant_grid=self.quant_grid, zero_point=False, round_method="up")
        normal_deq_weight[:,i1:i2], _, up_ratio, _ = get_quant_weight_mxfp(w[:,i1:i2], input_x=input[:,i1:i2], quant_grid=ed_quant_grid, zero_point=False, round_method="w_search")
        
    normal_output = torch.mm(q_x, normal_deq_weight.T)
    
    normal_mse = (normal_output - origin_output).pow_(2).mean(dim=0).mean(dim=0)
    ed_mse = (ed_output - origin_output).pow_(2).mean(dim=0).mean(dim=0)
    
    
    
    return w_quant, normal_mse, ed_mse
        
        
        
        
        
# [BS, H]
# [-1, 32]

# MXFP4 quant_grid [xxx, 6.0]

# MXFP8 quant_grid [xxx, 6.0]

@torch.no_grad()
def get_quant_weight_mxfp(w, fp, input_x=None, zero_point=True, q_group_size=-1, pos_value=None, round_method="rtn", error=None):
    '''
    return : dequantized weight, mse?
    '''
    
    if fp == 'fp4':
        quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    elif fp == 'fp8':
        quant_grid = [448.0]
    
    quant_grid = torch.tensor(quant_grid).to(dtype=torch.half)
 
    quant_grid = quant_grid.to(w.device)
    
    # round_method = "normal"
    

    max_val = w.abs().amax(dim=-1, keepdim=True)

    if pos_value is None or pos_value == True:
        max_quant_val = max(quant_grid)
    elif pos_value == False:
        max_quant_val = abs(min(quant_grid))
    else:
        raise NotImplementedError 
    
    # Compute the scaling factor
    
    # RTN rounding
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_rtn = torch.pow(2, exp)
    
    # All round down
    scales_down = torch.pow(2, (torch.floor(torch.log2(max_val/max_quant_val))))
    
    # # All round up
    scales_up = torch.pow(2, (torch.ceil(torch.log2(max_val/max_quant_val))))
    
    zeros = 0
        
    # Normal rounding
    scales_n = max_val / max_quant_val
        
    if round_method == "rtn":
        scales = scales_rtn
    elif round_method == "up": # recommend clip ratio 0.88
        scales = scales_up
    elif round_method == "down":
        scales = scales_down
    elif round_method == "nvfp":
        scales = scales_n.view(torch.short)
        hi = scales & 0xFF80
        r = scales & 0x0040
        
        scales = hi + r * 2
        
        scales = scales.view(torch.half)
    elif round_method == "w_search" and input_x != None:
        # labels_up = (((w + zeros) / scales_up).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        # labels_down = (((w + zeros) / scales_down).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        
        # w_deq_up = quant_grid[labels_up] * scales_up - zeros
        # w_deq_down = quant_grid[labels_down] * scales_down - zeros
        
        w_q_up = quant_mxfp4(w, scales_up)
        w_q_down = quant_mxfp4(w, scales_down)
        
        w_deq_up = w_q_up * scales_up
        w_deq_down = w_q_down * scales_down
        
        input_g = input_x.reshape(-1, 32)
        
        # deq_input_g, _, _, _ = get_quant_act_mxfp(x=input_g, weight=None, quant_grid=quant_grid, zero_point=False, round_method="up")
        
        out_origin = torch.mm(input_g, w.T)
        
        # out_up = torch.mm(input_g, w_deq_up.T)
        # out_down = torch.mm(input_g, w_deq_down.T)
        
        out_up = torch.mm(input_g, w_deq_up.T)
        out_down = torch.mm(input_g, w_deq_down.T)
        
        if error is None:
            up_mse = (out_origin - out_up).pow(2).mean(dim=0)
            down_mse = (out_origin - out_down).pow(2).mean(dim=0)
        else:
            up_mse = (out_up - out_origin + error).pow(2).mean(dim=0)
            down_mse = (out_down - out_origin + error).pow(2).mean(dim=0)
        
        # 1 for down, 0 for up
        mask = (((up_mse - down_mse) > 0).to(torch.int32)).unsqueeze(1)
        
        up_ratio = 1 - mask.sum() / mask.shape[0]
        
        scales = scales_down * mask + scales_up * (1 - mask)
    elif round_method == "normal":
        scales = scales_n

    # labels = (((w + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # # print(labels, labels.shape, quant_grid, quant_grid.shape)
    # w_deq = quant_grid[labels] * scales - zeros
    
    
    if fp == 'fp4':
        w_q = quant_mxfp4(w, scales)
    elif fp == 'fp8':
        w_q = quant_mxfp8(w, scales)

    w_deq = w_q * scales

    quant_mse = (w_deq-w).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # if get_labels:
    #     return w_deq, quant_mse_sum, labels, quant_grid * scales
    # else:
    
    # if round_method == "w_search":
    #     return w_deq, quant_mse_sum, up_ratio, scales
    # else:
    #     return w_deq, quant_mse_sum, 1, scales
    if round_method == "w_search":
        return w_q, quant_mse_sum, up_ratio, scales
    else:
        return w_q, quant_mse_sum, 1, scales
    





@torch.no_grad()
def get_quant_act_mxfp(x, fp, weight=None, zero_point=True, q_group_size=-1, pos_value=None, round_method="rtn", x_clip_r=0.88):
    '''
    return : dequantized weight, mse?
    '''
    ori_shape = x.shape
    x = split_last_dim_into_blocks(x)

    if fp == 'fp4':
        quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    elif fp == 'fp8':
        quant_grid = [448.0]
    
    quant_grid = torch.tensor(quant_grid).to(dtype=torch.half)
 
    quant_grid = quant_grid.to(x.device)

    # round_method = "normal"
    
    # print(x.shape, flush=True)
    
    max_val = x.abs().amax(dim=-1, keepdim=True)

    if pos_value is None or pos_value == True:
        max_quant_val = max(quant_grid)
    elif pos_value == False:
        max_quant_val = abs(min(quant_grid))
    else:
        raise NotImplementedError 
    
    # Compute the scaling factor
    
    # Normal rounding
    scales_n = max_val / max_quant_val
    
    # RTN rounding
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_rtn = torch.pow(2, exp)
    
    # All round down
    scales_down = torch.pow(2, (torch.floor(torch.log2(max_val/max_quant_val))))
    
    # # All round up
    scales_up = torch.pow(2, (torch.ceil(torch.log2(max_val * x_clip_r/max_quant_val))))
    
    
    zeros = 0
        
    if round_method == "rtn":
        scales = scales_rtn
    elif round_method == "up":
        scales = scales_up
    elif round_method == "down":
        scales = scales_down
    elif round_method == "x_search" and weight != None:
        labels_up = (((x + zeros) / scales_up).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        labels_down = (((x + zeros) / scales_down).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        
        x_deq_up = quant_grid[labels_up] * scales_up - zeros
        x_deq_down = quant_grid[labels_down] * scales_down - zeros
        
        
        out_origin = torch.mm(x, weight.T)
        
        out_up = torch.mm(x_deq_up, weight.T)
        out_down = torch.mm(x_deq_down, weight.T)
        
        up_mse = (out_origin - out_up).pow(2).mean(dim=1)
        down_mse = (out_origin - out_down).pow(2).mean(dim=1)
        
        # 1 for down, 0 for up
        mask = (((up_mse - down_mse) > 0).to(torch.int32)).unsqueeze(1)
        
        up_ratio = 1 - mask.sum() / mask.shape[0]
        # print(mask.shape, flush=True)
        # print(scales_down.shape, flush=True)
        
        scales = scales_down * mask + scales_up * (1 - mask)
        
    elif round_method == "bit_op_rtn":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        r_scales_bi = scale_s_e.view(torch.half)
        scales = r_scales_bi
        
    elif round_method == "bit_op_down":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        
        # scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        sign_bit = (max_val_bi & 0x0200) << 1
        r_scales_bi = (scale_s_e - (0x0400 - sign_bit)).view(torch.half)
        
        scales = r_scales_bi
        
    elif round_method == "bit_op_up":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        
        # scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        sign_bit = (max_val_bi & 0x0200) << 1
        r_scales_bi = (scale_s_e + sign_bit).view(torch.half)
        
        scales = r_scales_bi
        
        # print(f"Bit op scale up : {scales}", flush=True)
        # print(f"Normal scale up : {scales_up}", flush=True)
        # print(f"Scaling factor error : {(scales_up - scales).T} \n", flush=True)
        
    elif round_method == "normal":
        scales = scales_n
        
    # print(x.shape, flush=True)
    # labels = (((x + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # # print(labels, labels.shape, quant_grid, quant_grid.shape)
    # x_deq = quant_grid[labels] * scales - zeros
    
    
    # q_x = quant_mxfp4(x.clone().detach(), scales)

    if fp == 'fp4':
        q_x = quant_mxfp4(x.clone().detach(), scales)
    elif fp == 'fp8':
        q_x = quant_mxfp8(x.clone().detach(), scales)

    x_deq = q_x * scales
    
    # deal with zero channels
    
    # channel_sum = x_deq.sum(dim=0, keepdim=True)
    # zero_mask = (channel_sum == 0).to(torch.int32)
    
    # # assert (zero_mask == 1).any() == False, f"zero input channel, {zero_mask.tolist()}"
    # x_deq = x_deq + zero_mask * x 
    
    
    quant_mse = (x_deq - x).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)
    labels = None

    # if get_labels:
    #     return w_deq, quant_mse_sum, labels, quant_grid * scales
    # else:
    # if round_method == "x_search":
    #     return x_deq, quant_mse_sum, up_ratio, labels
    # else:
    #     return x_deq, quant_mse_sum, 1, labels
    
    if round_method == "x_search":
        return q_x, quant_mse_sum, up_ratio, scales
    else:
        dqinput = q_x*scales
        dqinput = dqinput.reshape(ori_shape)
        return dqinput
        # return q_x, quant_mse_sum, 1, scales


def find_nan_positions(tensor):
    nan_indices = torch.where(torch.isnan(tensor))
    num_nans = len(nan_indices[0]) if nan_indices else 0

    if num_nans == 0:
        print("✅ Tensor 中没有 NaN 值。")
        return

    print(f"⚠️ 发现 {num_nans} 个 NaN 值，位置如下：")
    for i in range(num_nans):
        # 获取每个维度上的索引
        idx = [dim[i].item() for dim in nan_indices]
        val = tensor[idx].item()
        print(f"  - 位置 {idx} → 值 = NaN")


# @torch.no_grad()
def quant_mxfp8(x, scaling_factor):
    xdtype = x.dtype
    
    finfo = torch.finfo(torch.float8_e4m3fn)
    
    q_x = (x / scaling_factor).clamp(min=finfo.min, max=finfo.max)

    q_x_fp8 = q_x.to(torch.float8_e4m3fn)

    return q_x_fp8.to(x.dtype)


@torch.no_grad()
def quant_mxfp4(x, scaling_factor):
    q_x = (x / scaling_factor).to(torch.half)
    x_bi = q_x.view(torch.short)
    
    x_sign = x_bi & 0x8000
    x_exp_m = x_bi & 0x7FFF
    # if x_exp_m < 0x3400:
    #     # x < 0.25
    #     q_x = 0x0000
    # elif x_exp_m < 0x3A00:
    #     # x < 0.75
    #     q_x = 0x3800
    # elif x_exp_m < 0x3C00:
    #     # x < 1.0
    #     q_x = 0x3C00
    # elif x_exp_m > 0x4600:
    #     # x > 6.0
    #     q_x = 0x4600
    # else:
    #     # x >= 1.0
    #     pass
    x_branch_0_mask = torch.where(x_exp_m >= 0x3400, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * x_branch_0_mask
    x_branch_0 = (0x0001 - x_branch_0_mask) * 0x0000
    
    x_branch_1_mask = torch.where(x_exp_m >= 0x3A00, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * x_branch_1_mask
    x_branch_1 = (x_branch_0_mask - x_branch_1_mask) * 0x3800
    
    x_branch_2_mask = torch.where(x_exp_m >= 0x3C00, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * x_branch_2_mask
    x_branch_2 = (x_branch_1_mask - x_branch_2_mask) * 0x3C00
    
    x_branch_3_mask = torch.where(x_exp_m >= 0x4600, 0x0001, 0x0000).to(torch.short)
    x_exp_m = x_exp_m * (x_branch_2_mask - x_branch_3_mask)
    x_branch_3 = x_branch_3_mask * 0x4600
    
    
    
    x_branch_4 = x_exp_m
    
    x_u = x_branch_4 & 0x7E00
    x_r = x_branch_4 & 0x0100
    
    x_branch_4 = x_u + 0x0002 * x_r
    
    x_exp_m = x_branch_0 + x_branch_1 + x_branch_2 + x_branch_3 + x_branch_4
    
    # print(x_exp_m.dtype)
    # print(x_branch_0.shape)
    # print(x_exp_m.shape, flush=True)
    
    q_x = (x_exp_m + x_sign).view(torch.half)

    return q_x



class MXFP_Linear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, ant_config, layer_id, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size 
        self.ant_config = ant_config

        self.layer_id = layer_id
        self.layer_name = layer_name
        
        # GPTQ option
        self.gptq = False
        self.gptq_perm = None
        
        # Error diffusion option
        self.erd = False
        
        # Permutation option
        self.permute = False
        self.idx = None
        
        
        # activation clipping parameter
        self.search_tw = True
        self.x_clip_tw = 1.0
        
        # chunk-wise activation clipping
        self.search_cw = False
        self.x_clip_cw = []
        
        
        self.w_rounding_method = None
        self.x_rounding_method = None

        # ANT param
        self.weight_quant_grid = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_alpha = -1
        
        self.quant_grid = None

        assert self.in_features % self.group_size == 0

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None
            
        self.skip_q_flag = False
        self.store_q_flag = True
        
        self.id = -1

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None, quant_mode=None, mxfp_config=None):

        mxfp_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading sd
            return mxfp_linear

        mxfp_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            mxfp_linear.bias = linear.bias.clone().half()
            
        quant_mode = mxfp_config["mx_type"]
        mxfp_linear.w_rounding_method = mxfp_config["w_round"]
        mxfp_linear.x_rounding_method = mxfp_config["x_round"]

        if w_bit > 6:
            mxfp_linear.ant_config['ant_mode'] = 'int'
            
        if quant_mode == "fp4_e2m1":
            quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        elif quant_mode == "fp6_e2m3":  
            quant_grid = [0.0,  0.125,  0.25,  0.375,  0.5,  0.625,  0.75,  0.875,
                                      -0.0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875,
                                      1.0,  1.125,  1.25,  1.375,  1.5,  1.625,  1.75,  1.875,
                                     -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
                                      2.0,  2.25,  2.5,  2.75,  3.0,  3.25,  3.5,  3.75,
                                     -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
                                      4.0,  4.5,  5.0,  5.5,  6.0,  6.5,  7.0,  7.5,
                                     -4.0, -4.5,  -5.0, -5.5, -6.0, -6.5, -7.0, -7.5]
        elif quant_mode == "fp6_e3m2":
            quant_grid = [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75,
                                      1.0, -1.0, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75,
                                      2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5,
                                      4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0,
                                      8.0, -8.0, 10.0, -10.0, 12.0, -12.0, 14.0, -14.0,
                                      16.0, -16.0, 20.0, -20.0, 24.0, -24.0, 28.0, -28.0,
                                      32.0, -32.0, 40.0, -40.0, 48.0, -48.0, 56.0, -56.0,
                                      64.0, -64.0, 80.0, -80.0, 96.0, -96.0, 112.0, -112.0]
            
        elif quant_mode == "int4":
            quant_grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                          -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
        mxfp_linear.quant_grid = torch.tensor(quant_grid).to(dtype=torch.half)
        
        
        
        return mxfp_linear
    
    @torch.no_grad()
    def forward(self, x):
        
        error_compensation = True
            
        out_shape = x.shape[:-1] + (self.out_features, )
        
        input = x.reshape(-1, x.shape[-1])
        
        # input.shape (seq_len, hidden_dim)
        if self.skip_q_flag:
            # if self.layer_name == "fc2":
            #     input = input.reshape(-1, input.shape[-1])
                
            #     input_sum = input.mean(dim=0, keepdim=True)
                
            #     zero_flag = (input_sum < 1e-4).to(torch.int32)
                
                
            #     # print(zero_flag.sum(dim=1) / input.shape[-1], flush=True)
                
            #     packed_flag = zero_flag.reshape(-1, 4).sum(dim=1, keepdim=True)
                
            #     all_zero = (packed_flag == 4).sum(dim=0) / packed_flag.shape[0]
            #     all_one = (packed_flag == 0).sum(dim=0) / packed_flag.shape[0]
            #     not_all_zero = 1.0 - all_zero - all_one
                
                
            #     if (self.id != None) and (self.id == self.layer_id):            
            #         print(f"Layer {self.layer_id} all zero = {all_zero}, not all zero = {not_all_zero}", flush=True)
            #         self.id = -2
            #     if self.id == -1:
            #         self.id = self.layer_id
                
                
            # print(self.layer_name, flush=True)
            
            
            out = F.linear(input, self.weight)

            out = out + self.bias if self.bias is not None else out
            return out.reshape(out_shape)

        
        # search and set data type and alpha in the first inference
        # calib weight activation quant weight
        if self.weight_quant_grid is None:
            
            g_size = 32

            org_w_shape = self.weight.shape
            
            N = org_w_shape[0]
            K = org_w_shape[1]
            M = input.shape[0]
            
            self.x_clip_cw = [0] * (K // 32)
            
            # weight = self.weight.reshape(-1, self.group_size)
            
            weight = self.weight
            
            deq_weight = weight.clone().detach()
            deq_input = input.clone().detach()
            
            error_list = [] # average error of each chunk
            
            if self.gptq:
                deq_weight, self.gptq_perm = gptq_quant(weight, input_x=x, blocksize=32)
                # print(self.gptq_perm.device, flush=True)
            elif self.erd and self.layer_id != 0:
                deq_weight, normal_mse, ed_mse = error_diffusion_quant(weight, ed_quant_grid=self.quant_grid, calib_input=x)
                print(f"Layer {self.layer_id}, name {self.layer_name}, normal mse: {normal_mse}, ed mse: {ed_mse}")
            elif self.permute:
                from .permmute import permutation_zigzag
                permute_x, idx = permutation_zigzag(input, group_size=32)
                self.idx = idx.to(weight.device)
                permute_weight = weight[:, idx]
                
                input = permute_x
                
                for i1 in range(0, K, g_size):
                    i2 = min(i1 + g_size, K)
                    # deq_weight[i*N : (i+1)*N, :], _, up_ratio = get_quant_weight_mxfp(weight[i*N : (i+1)*N, :], input_x=input[i*M : (i+1)*M, :], quant_grid=self.quant_grid, zero_point=False, round_method="up")
                    deq_weight[:,i1:i2], _, up_ratio, s = get_quant_weight_mxfp(permute_weight[:,i1:i2], input_x=input[:,i1:i2], quant_grid=self.quant_grid, zero_point=False, round_method=self.w_rounding_method)
                    deq_weight[:,i1:i2] = deq_weight[:,i1:i2] * s
            else:
                
                for i1 in range(0, K, g_size):
                    i2 = min(i1 + g_size, K)
                    # deq_weight[i*N : (i+1)*N, :], _, up_ratio = get_quant_weight_mxfp(weight[i*N : (i+1)*N, :], input_x=input[i*M : (i+1)*M, :], quant_grid=self.quant_grid, zero_point=False, round_method="up")
                    deq_weight[:,i1:i2], _, up_ratio, s = get_quant_weight_mxfp(weight[:,i1:i2], input_x=input[:,i1:i2], quant_grid=self.quant_grid, zero_point=False, round_method=self.w_rounding_method)
                    deq_weight[:,i1:i2] = deq_weight[:,i1:i2] * s
                    # ratio_list.append(up_ratio)
                # print(f'Layer {self.layer_id}, {self.layer_name}, rounding up ratio:{sum(ratio_list) / len(ratio_list)}', flush=True)
            
            assert torch.isnan(deq_weight).to(torch.int32).sum(0).sum(0) == 0, "weight NAN"
            # deq_weight, _ = get_quant_weight(weight, quant_grid=self.quant_grid, mode=None, zero_point=False, q_group_size=-1)
            
            
            if self.search_tw:
                # search clipping ratio of activation in tensor-wise
                min_clip_r = 0.8
                max_clip_r = 1.0
                
                
                opt_mse = 100000.0
                
                origin_output = torch.mm(input, weight.T)
                
                origin_input = input.reshape(-1, 32)
                
                origin_max = origin_input.abs().amax(dim=1, keepdim=True)
                
                while max_clip_r - min_clip_r > 0.01:
                    step = (max_clip_r - min_clip_r) / 5
                    for clip_r in np.arange(min_clip_r, max_clip_r+step, step):
                        
                        d_input = input.reshape(-1, g_size)
                        d_input, _, _, s = get_quant_act_mxfp(x=d_input, weight=None, quant_grid=self.quant_grid, zero_point=False, round_method="up", x_clip_r=clip_r)
                        d_input = d_input * s
                        d_input = d_input.reshape(input.shape)
                        
                        # minimize the quantization error of outliers
                        d_output = torch.mm(d_input, weight.T)
                        d_input = d_input.reshape(-1, 32)
                        d_max = d_input.abs().amax(dim=1, keepdim=True)
                        
                        # tmp_mse = ((d_max - origin_max).to(torch.float32) / origin_max.to(torch.float32)).abs().mean(dim=0)
                        
                        # kl divergence
                        # div = d_output.abs() / origin_output.abs()
                        # div = torch.where(torch.isinf(div), torch.ones_like(div).to(div.device), div)
                        # div = torch.where(torch.isnan(div), torch.ones_like(div).to(div.device), div)
                        
                        # kl_diver = ((div) * d_output.abs()).to(torch.float32).mean(dim=1).sum(dim=0)
                        
                        # tmp_mse = (d_output - origin_output).abs().to(torch.float32).sum(dim=1).mean(dim=0)
                        
                        
                        
                        # tmp_mse = (d_output - origin_output).abs().to(torch.float32).amax(dim=1).mean(dim=0)
                        
                        tmp_mse = (d_output - origin_output).abs().to(torch.float32).pow(2).sum(dim=1).mean(dim=0)
                        # print(kl_diver)
                        if tmp_mse < opt_mse:
                            opt_mse = tmp_mse
                            opt_clip_r = clip_r
                            
                            
                    max_clip_r = opt_clip_r + step
                    min_clip_r = opt_clip_r - step
                    
                    
                self.x_clip_tw = opt_clip_r
                d_input = input.reshape(-1, g_size)
                d_input, _, _, s = get_quant_act_mxfp(x=d_input, weight=None, quant_grid=self.quant_grid, zero_point=False, round_method="up", x_clip_r=opt_clip_r)
                d_input = d_input * s
                deq_input = d_input.reshape(input.shape)
                print(f"layer {self.layer_id}, name {self.layer_name}, opt clip ratio: {opt_clip_r}, mse: {opt_mse}", flush=True)
            
            # elif self.search_cw:
            #     if error_compensation:
            #         accu_error = torch.zeros((M, N), dtype=torch.half, device=input.device)
            #     for i1 in range(0, K, 32):
            #         i2 = min(i1 + 32, K)

            #         min_clip_r = 0.8
            #         max_clip_r = 1.0
                    
                    
            #         opt_mse = 100000.0
                    
            #         origin_output = torch.mm(input[:, i1:i2], deq_weight[:, i1:i2].T)
                    
            #         # origin_input = input.reshape(-1, 32)
                    
            #         # origin_max = origin_input.abs().amax(dim=1, keepdim=True)
                    
                    
                    
            #         while max_clip_r - min_clip_r > 0.01:
            #             step = (max_clip_r - min_clip_r) / 5
            #             for clip_r in np.arange(min_clip_r, max_clip_r+step, step):
                            
            #                 # d_input = input[:,i1,i2].clone().detach()
            #                 d_input, _, _, _ = get_quant_act_mxfp(x=input[:,i1:i2], weight=None, quant_grid=self.quant_grid, zero_point=False, round_method="up", x_clip_r=clip_r)
            #                 # d_input = d_input.reshape(input.shape)
                            
            #                 # minimize the quantization error of outliers
            #                 d_output = torch.mm(d_input, deq_weight[:,i1:i2].T)
                            
            #                 # print(accu_error.shape)
            #                 # print(d_output.shape)
            #                 # print(origin_output.shape, flush=True)
            #                 # d_input = d_input.reshape(-1, 32)
            