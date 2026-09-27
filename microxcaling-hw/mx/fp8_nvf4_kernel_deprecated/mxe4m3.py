import torch 
from quant_cy import QType, quant_dequant_float
import numpy as np 
import To_F8
import HW_MXINT8_MXFP8

np.random.seed(42)

N = 1
M = 1024*1024

x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32)
# x = torch.load('a.pth')[:,None]
x_torch = torch.from_numpy(x).cuda()

qtype = QType('mxfp8e4m3').dim(0)

# y1 = To_F8.To_MXFP8(x)[0]
y1 = HW_MXINT8_MXFP8.To_MXFP8(x)[0]
# y1 = quant_dequant_float(x_torch, qtype, force_py=True).cpu().numpy()
y2 = quant_dequant_float(x_torch, qtype, force_py=False).cpu().numpy()


diff = np.abs(y1 - y2)
print('DIFF MAX: ', diff.max())
arg = diff.flatten().argmax()
start = arg // 32 * 32
# print(x.flatten()[arg], y1.flatten()[arg], y2.flatten()[arg])
# print(x[start:start+32,0])
# print(y1[start:start+32,0])
# print(y2[start:start+32,0])

# torch.save(x[start:start+32,0], 'a.pth')
