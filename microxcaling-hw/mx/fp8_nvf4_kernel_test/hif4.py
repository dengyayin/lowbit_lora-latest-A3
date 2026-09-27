import torch 
from quant_cy import QType, quant_dequant_float_test
import numpy as np 
from HiFX import to_HiFX_V12


np.random.seed(42)
torch.manual_seed(42)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}, linewidth=65)


if __name__=='__main__':
    ##  生成随机变量
    M = 1024
    N = 1024
    x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N)))
    # x = np.zeros([M, N])
    x = x.astype(np.float32)

    x_torch = torch.from_numpy(x).cuda()


    for force in [True, False]:
        for N in [2,3,4,5]:
            y1 = to_HiFX_V12(x, N=N)

            # pytorch 实现
            qtype_str = 'hifx%d_v12'%(N)
            print('Testing N=%d  Qtype string: %s  force_py: %s'%(N, qtype_str, force))
            quant_type = QType(qtype_str).dim(0)

            y2 = quant_dequant_float_test(x_torch, quant_type, force_py=force, force_fp32=True).cpu().numpy()
            
            diff = np.abs(y1 - y2)
            # for i in range(y1.shape[0]):
            #     print(y1[i,0], y2[i,0], diff[i,0])
            print('ABS diff max (HIFXG):', np.max(diff))