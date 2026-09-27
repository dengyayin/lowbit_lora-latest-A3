import torch 
from quant_cy import QType, quant_dequant_float
import numpy as np 
import HW_MXINT8_MXFP8

np.random.seed(42)

N = 1
M = 1024*1024

x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32)
# x = torch.load('a.pth')[:,None]
x_torch = torch.from_numpy(x)#.cuda()

qtype = QType('e0m7k8b32').dim(0)

y1 = HW_MXINT8_MXFP8.To_MXINT8(x)
y2 = quant_dequant_float(x_torch, qtype, force_py=True).cpu().numpy()
# print(y1)
# print(y2)

diff = np.abs(y1 - y2)
print('DIFF MAX: ', diff.max())
# arg = diff.flatten().argmax()
# print(x.flatten()[arg//32*32:arg//32*32+32])
# print(y1.flatten()[arg//32*32:arg//32*32+32])
# print(y2.flatten()[arg//32*32:arg//32*32+32])
# torch.save(x.flatten()[arg//32*32:arg//32*32+32], 'a.pth')
