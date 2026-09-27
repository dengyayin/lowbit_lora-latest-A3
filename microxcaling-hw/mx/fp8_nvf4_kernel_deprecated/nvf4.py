import torch 
from quant_cy import QType, quant_dequant_float
import numpy as np 
import HiF4_NVFP4

np.random.seed(42)

N = 1024
M = 1024

# x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32)
x = torch.load('tensor.pt')[:,None]#.reshape([512*24*2, 96])
print(x.shape)
# x_torch = torch.from_numpy(x).cuda()
x_torch = x.cuda()

qtype = QType('nvf4').dim(0)

# y1 = HiF4_NVFP4.To_NVFP4(x)
# y1 = quant_dequant_float(x_torch, qtype, force_py=True).cpu().numpy()
y2 = quant_dequant_float(x_torch, qtype, force_py=False).cpu().numpy()
print(y2.max())

# diff = np.abs(y1 - y2)
# print('DIFF MAX: ', diff.max())

# arg = diff.flatten().argmax()
# start = arg // 16 * 16
# print(x.flatten()[arg], y1.flatten()[arg], y2.flatten()[arg])
# print(x[start:start+16,0])
# print(y1[start:start+16,0])
# print(y2[start:start+16,0])

# torch.save(x[start:start+16,0], 'a.pth')
