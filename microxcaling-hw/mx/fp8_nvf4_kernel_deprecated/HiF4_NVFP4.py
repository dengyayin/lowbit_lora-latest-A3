import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def To_MXFP4(x, G = 32): 
    # E8 + E2M1, column-wise convert
    # Round Half tie to even
    # if overflow, clip to upper-bound of E2M1

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmp = np.abs(ori)   # abs of grp values

            E = np.floor(np.log2(tmp + 2**(-1000)))          
            Emax = np.max(E) 
            E8 = Emax - 2       # Sub Max_Exp of E2M1

            if E8 < -127:      
                E8 = -127

            igv= tmp*2**(-E8)    # original in_grp values

            E2 = np.floor(np.log2(igv + 2**(-1000)))    
            E2[E2 < 0] = 0      # subnormal     
            M1 = np.round(igv*2**(-E2 + 1))*2**(-1)     # Round Half to Even
            E2M1 = 2**E2*M1
            E2M1[E2M1 > 6] = 6  # Clip to Upper-bound

            grp = S * E2M1 * 2**E8

            res[i*G:i*G+G,j] = grp

    return res


def To_NVFP4(x, G = 16): 
    # E4M3 + E2M1, column-wise convert
    # Round Half tie to even
    # if overflow, clip to upper-bound of E2M1

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmp = np.abs(ori)   # abs of grp values

            # 1 DIV-1 ---> replace by (x * 1/6)
            SF = np.max(tmp) / 6  # Scale Factor = Max_Value / 6 (6 is the max number represented by E2M1)   
            SF = 448 if (SF > 448) else SF    # overflow handling
            E_SF = np.floor(np.log2(SF + 2**(-1000)))    # Exp of E4M3
            E_SF = -6 if (E_SF < -6) else E_SF    # subnormal handling
            E4M3 = np.round(SF * 2 ** (-E_SF + 3)) * 2 ** (-3 + E_SF)
            if E4M3==0:
                res[i*G:i*G+G,j] = 0
                continue

            # 16 DIV-2
            igv= tmp / E4M3    # original in_grp values

            E2 = np.floor(np.log2(igv + 2**(-1000)))    
            E2[E2 < 0] = 0      # subnormal     
            M1 = np.round(igv*2**(-E2 + 1))*2**(-1)     # Round Half to Even
            E2M1 = 2**E2*M1
            E2M1[E2M1 > 6] = 6  # Clip to Upper-bound

            grp = S * E2M1 * E4M3

            res[i*G:i*G+G,j] = grp

    return res


def To_HiFX(x, N = 4, G = 64): 
    # {E8 + 8*E1 + 16*E1} + 1PN: Three-level scaling
    # EXP = [-127, 128], column-wise convert

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 2

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values

            EG = np.floor(np.log2(tmpG + 2**(-1000))) 

            E16 = np.zeros(16)
            for k in range(16):
                E16[k] = np.max(EG[k*4:k*4+4])

            E8 = np.zeros(8)
            for k in range(8):
                E8[k] = np.max(E16[k*2:k*2+2])
            
            Emax = np.max(E8)

            # level-1 exp
            E8_1 = Emax - 2     # [-127, 126]

            # level-2 exp
            E1_8 = E8 - E8_1 - 1    # <= 1
            E1_8[E1_8 < 0] = 0      # [0, 1]

            # level-3 exp
            E1_8x2 = np.zeros(16)
            for k in range(8):
                E1_8x2[k*2:k*2+2] = E1_8[k]
            
            E1_16 = E16 - E1_8x2 - E8_1  # <= 1
            E1_16[E1_16 < 0] = 0    # [0, 1]

            # Restore grp vals
            E16G = E1_16 + E1_8x2 + E8_1    # Fused 16 exp vals
            EG = np.zeros(G)                # Fused 64 exp vals
            for k in range(16):
                EG[k*4:k*4+4] = E16G[k]

            in_grp = np.floor(tmpG*2**(-EG + Ng)+0.5)*2.0**(-Ng)    # 1PNg
            in_grp[in_grp >= 2] = 2- 2**(-Ng)

            grp = S * in_grp * 2.0**EG

            res[i*G:i*G+G,j] = grp

    return res


# M = 1024
# N = 1024

# mu = -0.01
# sigma = 9
# bd = sigma*0.05
# x = np.random.normal(mu, sigma, (M, N)) + np.random.uniform(-bd, bd, (M, N))

# MXFP4 = To_MXFP4(x)
# MSE_MXFP4 = np.sum((MXFP4 - x)**2)/(M*N)
# print('MXFP4: MSE = ', MSE_MXFP4)

# NVFP4 = To_NVFP4(x)
# MSE_NVFP4 = np.sum((NVFP4 - x)**2)/(M*N)
# print('NVFP4: MSE = ', MSE_NVFP4)

# HiF4 = To_HiFX(x)
# MSE_HiF4 = np.sum((HiF4 - x)**2)/(M*N)
# print('HiF4: MSE = ', MSE_HiF4)