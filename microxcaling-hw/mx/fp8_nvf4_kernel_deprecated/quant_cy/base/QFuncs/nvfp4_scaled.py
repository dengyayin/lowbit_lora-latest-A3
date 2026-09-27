import torch 
from ..QType import QType 
from torch import Tensor 

def linear_rescale(x, a, b, c, d):
    x_norm = (x - a) / (b - a)
    y = c + x_norm * (d - c)
    return y

def inverse_linear_rescale(y, a, b, c, d):

    y_norm = (y - c) / (d - c)
    x = a + y_norm * (b - a)
    return x

@torch.no_grad()
def quant_nvf4_scaled(x: Tensor, Q: QType, qdim: int): 
    x = x.unflatten(qdim, (-1, 16))
    print(f"original value: \n{x}")
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    grp_max = torch.amax(x_unsigned, dim=qdim, keepdim=True)
    sf = grp_max / 6

    print(f"fp16 scale:\n{sf}")
    #sf = torch.clip_(sf, 2**-9, 448)
    #scale_double = sf.flatten(-1).min()[0] / (2 ** -9)
    sf_min, sf_max = sf.flatten(-1).min(), sf.flatten(-1).max()
    sf = linear_rescale(sf, sf_min, sf_max, 2**-9, 448)

    print(f"normed sf:\n{sf}")
    if x.dtype==torch.float16:
        sf_exp = torch.floor(torch.log2(sf + 2**-14))
    else:
        sf_exp = torch.floor(torch.log2(sf + 2**-45))
    sf_exp.clamp_(-6)

    E4M3 = torch.round(sf * 2 ** (-sf_exp + 3)) * 2 ** (-3 + sf_exp)

    print(f"E4M3 value: \n{E4M3}")
    print(f"E4M3 shape: \n{E4M3.shape}")
    #E4M3.clamp_(2E-9, 448)
    E4M3 = linear_rescale(E4M3, 2**-9, 448, sf_min, sf_max)
    print(f"E4M3 value after clamp: \n{E4M3}")
    igv = x_unsigned / E4M3
    print(f"before processed igv: \n{igv}")
    #igv[~torch.isfinite(igv)] = 6
    #print(f"after processed igv: \n{igv}")
    if x.dtype==torch.float16:
        E2 = torch.floor(torch.log2(igv + 2**(-14)))
    else:
        E2 = torch.floor(torch.log2(igv + 2**(-45)))
    E2.clamp_(0)
    M1 = torch.round(igv*2**(-E2 + 1))*2**(-1)
    E2M1 = 2**E2*M1
    E2M1.clamp_(0, 6)

    print(f"E2M1 value: \n{E2M1}")

    res = sign * E4M3 * E2M1
    res = res.flatten(qdim-1, qdim)
    
    print(f"thquant result: \n{res}")
    return res 
