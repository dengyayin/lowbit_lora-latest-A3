import torch 
from quant_cy import QType, quant_dequant_float
import numpy as np 
import HiF4_NVFP4_v14f16


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

    x_torch = torch.from_numpy(x).bfloat16().float().cuda()
    x = x_torch.cpu().numpy()


    for force in [False]:
        for N in [2,3,4,5]:
            y1 = HiF4_NVFP4_v14f16.To_HiFX(x, N=N)

            # pytorch 实现
            qtype_str = 'hifx%d'%(N)
            print('Testing N=%d  Qtype string: %s  force_py: %s'%(N, qtype_str, force))
            quant_type = QType(qtype_str).dim(0)

            y2 = quant_dequant_float(x_torch.bfloat16(), quant_type, force_py=force).float().cpu().numpy()
            
            diff = np.abs(y1 - y2)
            print('ABS diff max (HIFX):', np.max(diff))
        
    print('Testing zero values')
    y1 = HiF4_NVFP4_v14f16.To_HiFX(x*0, N=N)
    y2 = quant_dequant_float((x_torch*0), QType('hifx4').dim(0), force_py=False).float().cpu().numpy()
    diff = np.abs(y1 - y2)
    print('ABS diff max (zero values):', np.max(diff))
