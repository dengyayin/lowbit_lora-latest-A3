import torch
from mx.fp4_ops import quantize_hif4_kernel
from transformers import set_seed
import pdb

data = torch.load("/home/z00663592/trans/test_hif4_data.pt", map_location="cpu")

x = data['x1']
print(x)
y = quantize_hif4_kernel(x, N=4)
y1 = data['y1']
print(f"y: {y}")
print(f"y1: {y1}")
if not torch.allclose(y, y1, rtol=1e-3, atol=1e-4):
    raise ValueError
rel_err = torch.norm(x - y) / torch.norm(x)
print(f"rel err: {rel_err}")

x = data['x2']
print(x)
y = quantize_hif4_kernel(x, N=4)
y1 = data['y2']
if not torch.allclose(y, y1, rtol=1e-4, atol=1e-5):
    raise ValueError
rel_err = torch.norm(x - y) / torch.norm(x)
print(f"rel err: {rel_err}")

x = data['x3']
print(x)
y = quantize_hif4_kernel(x, N=4)
y1 = data['y3']
if not torch.allclose(y, y1, rtol=1e-4, atol=1e-5):
    raise ValueError
rel_err = torch.norm(x - y) / torch.norm(x)
print(f"rel err: {rel_err}")