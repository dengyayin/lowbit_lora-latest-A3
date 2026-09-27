import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def To_BF16(x):
    x = np.float64(x)
    tmp = np.abs(x)
    E = np.floor(np.log2(tmp))
    # E[E < -126] = -126
    E = -126 if E < -126 else E
    res = np.round(x*2**(-E + 7))*2**(E - 7)
    return res

def To_MXINT8(x, G = 32): 
    # convert matrix [M,N] to MX-INT8 with group-size of G (column-wise)
    # in-grp format: S1P6
    # Scale Foator: E8

    # x = np.array(x).astype(np.float64)

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

            E8 = np.floor(np.log2(tmp + 2**(-1000)))
            Emax = np.max(E8)
            # print('EMAX', Emax)
            # print('MANT', tmp * 2 ** (-Emax + 6))
            igv = np.round(tmp * 2 ** (-Emax + 6))  # tie to even
            igv[igv > 127] = 127
            igv = igv * 2 ** (- 6)

            grp = S * igv *2 ** Emax
            res[i*G:i*G+G,j] = grp
    return res

def To_FP8(x, fmt = 'E4M3'):
    # Convert data to FP8
    # fmt: 'E4M3' for default, others for 'E5M2'
    # Round Mode: Round Half tie to Even; if overflow, clip to upper bound
    x = np.float64(x)
    S = np.ones_like(x)
    S[x < 0] = -1
    tmp = np.abs(x)
    E = np.floor(np.log2(tmp + 2**(-1000)))

    if fmt == 'E4M3':
        E[E < -6] = -6
        res_abs = np.round(tmp*2**(-E + 3))*2**(E - 3)
        res_abs[np.abs(res_abs) > 448] = 448    # Clip to upper-bound
        res = S * res_abs
    else:   # for 'E5M2'
        E[E < -14] = -14
        res_abs = np.round(tmp*2**(-E + 2))*2**(E - 2)
        res_abs[np.abs(res_abs) > 57344] = 57344    # Clip to upper-bound
        res = S * res_abs

    return res

def To_MXFP8(x, fmt = 'E4M3'):
    # Convert data to MXFP8, Column-wise
    # fmt: 'E4M3' for default, others for 'E5M2'
    # Round Mode: Round Half tie to Even; if overflow, clip to upper bound
    x = np.float64(x)
    G = 32
    M = x.shape[0]
    N = x.shape[1]
    Mcnt = np.ceil(M/G).astype(int)
    res = np.zeros((M,N))
    E8 = np.zeros((Mcnt,N))
    grp = np.zeros((M,N))
    if fmt == 'E4M3':
        Emax = 8    # Max Exp of E4M3
    else:
        Emax = 15   # Max Exp of E5M2
    for i in range(Mcnt):
        for j in range(N):
            ori = x[i*G:i*G+G,j]
            tmp = np.abs(ori)   # abs of grp values

            E = np.floor(np.log2(tmp + 2**(-1000)))          
            E8g = np.max(E) - Emax
            E8g = -127 if (E8g < -127) else E8g

            xgi = ori * 2 ** (-E8g)        # in_grp convert input
            xgo = To_FP8(xgi,fmt = fmt)    # in_grp convert output
            xgr = xgo * 2 ** (E8g)         # restored grp values
            E8[i,j] = E8g
            grp[i*G:i*G+G,j] = xgo
            res[i*G:i*G+G,j] = xgr 
    return res, E8, grp



# M = 1024
# N = 512

# Tcnt = 18   # Test Number

# MXFP8_MSE       = np.zeros(Tcnt)
# MXINT8_MSE      = np.zeros(Tcnt)

# MSE_NORM_MXFP8 = np.zeros(Tcnt)

# for NT in range(Tcnt):      
#     sigma = 1e-2 * 2 ** NT
#     mu = -sigma*0.05
#     bd = mu
#     x = np.random.normal(mu, sigma, (M, N)) + np.random.uniform(-bd, bd, (M, N))

#     MXFP8, E8, grp = To_MXFP8(x)
#     MSE_MXFP8 = np.sum((MXFP8 - x)**2)/(M*N)

#     MXINT8 = To_MXINT8(x)
#     MSE_MXINT8 = np.sum((MXINT8 - x)**2)/(M*N)

#     MSE_MXFP8_NORM = MSE_MXFP8 / MSE_MXINT8

#     MXFP8_MSE       [NT] = MSE_MXFP8
#     MXINT8_MSE      [NT] = MSE_MXINT8

#     MSE_NORM_MXFP8 [NT] = MSE_MXFP8_NORM


# print('MXFP8_MSE = \n', MXFP8_MSE)
# print('MXINT8_MSE = \n', MXINT8_MSE)
# print('MSE_NORM_MXFP8 = \n', MSE_NORM_MXFP8)


# MSE_NORM_MXINT8 = np.ones(18)
# MSE_NORM = np.vstack([MSE_NORM_MXFP8, MSE_NORM_MXINT8]).T
# columns = ['MSE_MXFP8', 'MSE_MXINT8']
# idx = range(18)

# df = pd.DataFrame(MSE_NORM, idx, columns)
# df.plot(
#     linestyle = '-',
#     linewidth = 2,
#     marker = 'o',
#     markersize = 6,
#     markeredgecolor = 'black',
#     markerfacecolor = 'red',
#     title = 'Mean Squared Error Normalized to MXINT8')
# plt.xlabel('Gaussian Dist: sigma = 0.01 * 2 ** X, mu = -sigma * 0.05')
# plt.ylabel('Normalized MSE')
# plt.show()