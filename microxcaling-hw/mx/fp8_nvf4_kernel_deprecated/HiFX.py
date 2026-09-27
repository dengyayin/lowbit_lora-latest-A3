import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt

def to_MXN(x, N = 5, G = 32, ExpUD = 1): 
    # E8 + 1PN
    # ExpUD = 1, Update Exp; ExpUD = 0, Clip In-Grp Data

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
            tmp = np.abs(ori)   # abs of grp values

            E = np.floor(np.log2(tmp + 2**(-140)))          
            Emax = np.max(E) 
            E8 = Emax

            if ExpUD == 1:
                if (tmp*2**(-Emax) >= 2 - 2.0**(-(N-1))).any():     # 1p111xx：Round Carry-in updates Exp
                    E8 = Emax + 1

            if E8 < -127:        # Saturate, No indeterminacy, -16 for NaN
                E8 = -127
            
            if E8 > 127:
                E8 = 127

            in_grp= np.floor(tmp*2**(-E8+Ng)+0.5)*2**(-Ng)    # 1PNg
            if ExpUD == 0:
                in_grp[in_grp == 2] = 2 - 2**(-Ng)

            grp = S * in_grp * 2**E8

            res[i*G:i*G+G,j] = grp

    return res

def to_TMXN(x, N = 5, G = 32, ExpUD = 0): 
    # E8 + 8*E1 + 1PN: Two-level scaling MX Format
    # ExpUD = 1, Update Exp; ExpUD = 0, Clip In-Grp Data
    # MX6: G = 16, ExpUD = 0

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 2
    NR = N - 1
    G2 = np.int32(G/8)

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmp = np.abs(ori)   # abs of grp values

            E = np.floor(np.log2(tmp + 2**(-140)))          
            Emax = np.max(E) 
            if ExpUD == 1:          # Exp UpDate
                if (tmp*2**(-Emax) >= 2 - 2.0**(-NR)).any():     # 1p1111xx：Round Carry-in updates Exp
                    Emax = Emax + 1

            E8 = Emax - 1       # level-1 Scaling

            if E8 > 127:        # 128 for NaN
                E8 = 127

            if E8 < -127:
                E8 = -127

            E1x8 = np.zeros(8)  # level-2 Scaling
            for k in range(8):
                Ek_max = np.max(E[k*G2:k*G2+G2])
                if ExpUD == 1:          # Exp UpDate
                    if (tmp[k*G2:k*G2+G2]*2**(-Ek_max) >= 2 - 2.0**(-NR)).any():
                        Ek_max = Ek_max + 1
                E1x8[k] = Ek_max
            
            E1x8 = E1x8 - E8
            E1x8[E1x8 < 0] = 0

            E8G = E1x8 + E8     # Fused 8 scaling exp
            EG = np.zeros(G)    # Fused G scaling exp
            for k in range(8):
                EG[k*G2:k*G2+G2] = E8G[k]

            in_grp = np.floor(tmp*2**(-EG + Ng)+0.5)*2.0**(-Ng)    # 1PNg
            if ExpUD == 0:
                in_grp[in_grp == 2] = 2 - 2**(-Ng)

            grp = S * in_grp * 2.0**EG

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFXG_V3(x, N = 5, G = 32): 
    # This is HiFXG_V3
    # convert matrix x to HiFX with group_size = G (column-wise)
    # N: in_group format width, can be configured as 5, 4, 3 and 2 for 
    #    HiF5 (S1P3), HiF4 (S1P2), HiF3 (S1P1), and HiF2 (S1P0)
    # G: group_size, can be configured as 32 and 16 for HiFX _G32 & HiFX_G16

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    DEout = np.zeros((Mcnt, Ni))       # Store delta E

    NM = N - 2      # fractional width of in_group format: S_1_P_NM, NM = 3, 2, 1, & 0
    NR = N - 1      # Round Width for S_1_P_NM 

    for i in range(Mcnt):
        for j in range(Ni):

            Gh = np.int32(G/2)

            ori = x[i*G:i*G+G,j]
            ori_A = ori[0:Gh]
            ori_B = ori[Gh:G]

            S_A = np.ones(Gh)  # sign of grp_A values
            S_A[ori_A < 0] = -1
            S_A = S_A.T
            tmpA = np.abs(ori_A)   # abs of grp_A values

            S_B = np.ones(Gh)  # sign of grp_B values
            S_B[ori_B < 0] = -1
            S_B = S_B.T
            tmpB = np.abs(ori_B)   # abs of grp_B values


            EA = np.floor(np.log2(tmpA + 2**(-50)))          
            EmaxA = np.max(EA)                  
            if (tmpA*2**(-EmaxA) >= 2 - 2.0**(-NR)).any():     # 1p1111xx：Round Carry-in updates Exp
                EmaxA = EmaxA + 1

            if EmaxA < -42:        # Saturate, No indeterminacy, -43 for NaN
                EmaxA = -42
            
            if EmaxA > 20:
                EmaxA = 20

            EB = np.floor(np.log2(tmpB + 2**(-50)))
            EmaxB = np.max(EB)
            if (tmpB*2**(-EmaxB) >= 2 - 2.0**(-NR)).any():     # 1p1111xxx: Round Carry-in updates Exp
                EmaxB = EmaxB + 1

            if EmaxB < -42:        # Saturate, No indeterminacy, -43 for NaN
                EmaxB = -42
            
            if EmaxB > 20:
                EmaxB = 20

            DE2 = EmaxB - EmaxA     # 2-bit = [-2, 1], 2'sC
            # E6B = E6A + DE2
            if DE2 > 1:
                DE2 = 1
                E6A = EmaxB - 1
                E6B = EmaxB
            elif DE2 < -2:
                DE2 = -2
                E6A = EmaxA
                E6B = EmaxA - 2
            else: 
                E6A = EmaxA
                E6B = EmaxB

            tmpA2 = tmpA*2.0**(-E6A) # absolute value of grp-A
            tmpB2 = tmpB*2.0**(-E6B) # absolute value of grp-B

            fp4A_abs = np.floor(tmpA2*2.0**NM + 0.5)*2.0**(-NM)    # Round Half for mantissa
            fp4B_abs = np.floor(tmpB2*2.0**NM + 0.5)*2.0**(-NM)    # Round Half for mantissa    

            grpA = S_A*fp4A_abs*2**E6A
            grpB = S_B*fp4B_abs*2**E6B

            grp = np.zeros((1,G))             # unified upper bound: 2**20*1.5 = 1572864 for all HiFXG
            if (ori == np.nan).any() or (ori == np.inf).any() or (ori == -np.inf).any() or (np.max(abs(ori)) >= 1572864):
                grp = np.nan
            else: 
                grp = np.append(grpA, grpB)

            res[i*G:i*G+G,j] = grp
            DEout[i,j] = DE2

    return res, DEout

def to_HiFXG_V4(x, N = 5, G = 32): 
    # N = 5，4，3，2; Ng = N - 1 = 4,3,2,1
    # E8 = E6 + M2 = E6M2
    # M2 --> 1.M2 = [1, 2)
    # In_grp format: N-bit,S0P-Ng

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 1      # in-grp format：S-P-Ng

    # Bounds：Unified 1P7 Format, MSB = 1'b1
    if N != 2:
        # Constants for Boundaries (Also Update EXP), for HiF5/4/3
        bd1 = (1 - 2**(-N)) * 1.25          # >= bd1 RH --> 1 * 1.25
        bd2 = (1 - 2**(-N)) * 1.50          # >= bd2 RH --> 1 * 1.50
        bd3 = (1 - 2**(-N)) * 1.75          # >= bd3 RH --> 1 * 1.75
        bd4 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd4 RH --> 2 * 1.00 
    else: 
        # Constants for Boundaries (Also Update EXP), for HiF2 only 
        bd1 = (1 - 2**(-N)) * 1.50          # >= bd1 RH --> 1 * 1.50
        bd2 = (1 - 2**(-N)) * 1.75          # >= bd2 RH --> 1 * 1.75
        bd3 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd3 RH --> 2 * 1.00
        bd4 = (1 - 2**(-N)) * 1.25 * 2.00   # >= bd4 RH --> 2 * 1.25

    # Constants for 1 / M2P1:8-bit
    rec_1p00 = 1.0
    rec_1p01 = np.floor((1.0/1.25)*2.0**8)*2.0**(-8)
    rec_1p10 = np.floor((1.0/1.50)*2.0**8)*2.0**(-8)
    rec_1p11 = np.floor((1.0/1.75)*2.0**8)*2.0**(-8)

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmp = np.abs(ori)   # abs of grp values

            max_tmp = np.max(tmp)   # max abs value of group
            Emax = np.floor(np.log2(max_tmp + 2**(-50))) # EXP of max abs value
            max_igrp = max_tmp * 2.0 ** (-Emax)      # [1, 2)

            M2P1 = 1.0
            rec = 1.0

            # original ulp = 2^(-(N-1)), RH-->original Error Bound (EUB) = 0.5 ulp
            if N != 2:  # HiF5/4/3
                if (max_igrp < bd1):
                    M2P1 = 1.25
                    rec = rec_1p01          # EUB: 1.25 ulp * 0.5 * 0.5 = 0.3125 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd4):     
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
            else:   # HiF2
                if (max_igrp < bd1):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
                elif (max_igrp >= bd4):   
                    M2P1 = 1.50
                    rec = rec_1p10
                    Emax = Emax + 1         # EUB: 1.5 ulp * 0.5 = 0.75 ulp (RH)

            E6 = Emax
            if E6 < -42:        # offset = 42: [-42, 20], 21 for NaN
                E6 = -42
            
            if E6 > 20:         
                E6 = 20

            tmp2 = tmp * 2.0 ** (-E6) # Values of significands

            in_grp = np.floor((tmp2 * rec)*2.0**Ng + 0.5)*2.0**(-Ng) # 0PNg: Round Half to Away

            # in_grp[in_grp >= 1] = 1 - 2.0**(-Ng)   # 1PNg --> PNg: Make sure to be less than 1

            grp = S * M2P1 * in_grp * 2.0 ** E6

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFXG_V5(x, N = 5, G = 32): 
    # N = 5，4，3，2; Ng = N - 1 = 4,3,2,1
    # E8 = E4A-M2-DE2, E4A for grpA
    # DE2 = [-2, 1]
    # E4B = E4A + DE2, for grpB
    # M2 --> 1.M2 = [1, 2)
    # In_grp format: N-bit,S0P-Ng

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    DEout = np.zeros((Mcnt, Ni))       # Store delta E

    Ng = N - 1      # in-grp format：S-P-Ng

    # Bounds：Unified 1P7 Format, MSB = 1'b1
    if N != 2:
        # Constants for Boundaries (Also Update EXP), for HiF5/4/3
        bd1 = (1 - 2**(-N)) * 1.25          # >= bd1 RH --> 1 * 1.25
        bd2 = (1 - 2**(-N)) * 1.50          # >= bd2 RH --> 1 * 1.50
        bd3 = (1 - 2**(-N)) * 1.75          # >= bd3 RH --> 1 * 1.75
        bd4 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd4 RH --> 2 * 1.00 
    else: 
        # Constants for Boundaries (Also Update EXP), for HiF2 only 
        bd1 = (1 - 2**(-N)) * 1.50          # >= bd1 RH --> 1 * 1.50
        bd2 = (1 - 2**(-N)) * 1.75          # >= bd2 RH --> 1 * 1.75
        bd3 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd3 RH --> 2 * 1.00
        bd4 = (1 - 2**(-N)) * 1.25 * 2.00   # >= bd4 RH --> 2 * 1.25

    # Constants for 1 / M2P1:8-bit
    rec_1p00 = 1.0
    rec_1p01 = np.floor((1.0/1.25)*2.0**8)*2.0**(-8)
    rec_1p10 = np.floor((1.0/1.50)*2.0**8)*2.0**(-8)
    rec_1p11 = np.floor((1.0/1.75)*2.0**8)*2.0**(-8)

    for i in range(Mcnt):
        for j in range(Ni):

            Gh = np.int32(G/2)

            ori = x[i*G:i*G+G,j]
            ori_A = ori[0:Gh]
            ori_B = ori[Gh:G]

            S_A = np.ones(Gh)  # sign of grp_A values
            S_A[ori_A < 0] = -1
            S_A = S_A.T
            tmpA = np.abs(ori_A)   # abs of grp_A values
            max_tmpA = np.max(tmpA)
            EmaxA = np.floor(np.log2(max_tmpA + 2**(-50))) # EXP of max abs value of grpA
            max_igrpA = max_tmpA * 2.0 ** (-EmaxA)      # [1, 2)

            S_B = np.ones(Gh)  # sign of grp_B values
            S_B[ori_B < 0] = -1
            S_B = S_B.T
            tmpB = np.abs(ori_B)   # abs of grp_B values
            max_tmpB = np.max(tmpB)
            EmaxB = np.floor(np.log2(max_tmpB + 2**(-50))) # EXP of max abs value of grpB
            max_igrpB = max_tmpB * 2.0 ** (-EmaxB)      # [1, 2)


            if max_tmpA >= max_tmpB:
                max_igrp = max_igrpA
                Emax = EmaxA
            else:
                max_igrp = max_igrpB
                Emax = EmaxB

            M2P1 = 1.0
            rec = 1.0

            # original ulp = 2^(-(N-1)), RH-->original Error Bound (EUB) = 0.5 ulp
            if N != 2:  # HiF5/4/3
                if (max_igrp < bd1):
                    M2P1 = 1.25
                    rec = rec_1p01          # EUB: 1.25 ulp * 0.5 * 0.5 = 0.3125 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1           # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd4):     
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1           # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
            else:   # HiF2
                if (max_igrp < bd1):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1           # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1           # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd4):   
                    M2P1 = 1.50
                    rec = rec_1p10
                    Emax = Emax + 1           # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)

            if max_tmpA >= max_tmpB: 
                EA = Emax
                DE = 1 + np.floor(np.log2(max_tmpB * 2.0 ** (-Emax) * rec + 2**(-50)))  # <=0
                EB = Emax + DE
                DE2 = DE
            else:
                EB = Emax
                DE = 1 + np.floor(np.log2(max_tmpA * 2.0 ** (-Emax) * rec + 2**(-50)))  # <=0
                EA = Emax + DE
                DE2 = - DE
            
            

            # DE2 = EmaxB - EmaxA     # 2-bit = [-2, 1], 2'sC
            # E4B = E4A + DE2
            if DE2 > 1:
                DE2 = 1
                E4A = EB - 1
                E4B = EB
            elif DE2 < -2:
                DE2 = -2
                E4A = EA
                E4B = EA - 2
            else: 
                E4A = EA
                E4B = EB
       
            if E4A < -7:        # offset = -7: [-7, 7], 8 for NaN
                E4A = -7
            
            if E4A > 7:
                E4A = 7

            if E4B < -7:        
                E4B = -7
            
            if E4B > 7:
                E4B = 7

            tmpA2 = tmpA*2.0**(-E4A) # absolute value of grp-A
            in_grpA = np.floor((tmpA2 * rec)*2.0**Ng + 0.5)*2.0**(-Ng) # 0PNg: Round Half to Away
            in_grpA[in_grpA >= 1] = 1 - 2**(-Ng)    # clip for small value grp
            grpA = S_A * M2P1 * in_grpA * 2.0 ** E4A

            tmpB2 = tmpB*2.0**(-E4B) # absolute value of grp-B
            in_grpB = np.floor((tmpB2 * rec)*2.0**Ng + 0.5)*2.0**(-Ng) # 0PNg: Round Half to Away
            in_grpB[in_grpB >= 1] = 1 - 2**(-Ng)    # clip for small value grp
            grpB = S_B * M2P1 * in_grpB * 2.0 ** E4B

            res[i*G:i*G+G,j] = np.append(grpA, grpB)
            DEout[i,j] = DE2

    return res, DEout

def to_HiFX_V6(x, N = 5, G = 32, ExpUD = 0): 
    # {E6 + 2*E1 + 8*E1} + 1PN: Three-level scaling
    # EXP = [-46, 18]
    # ExpUD = 1, Update Exp; ExpUD = 0, Clip In-Grp Data

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 2
    NR = N - 1

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values

            E8 = np.zeros(8)
            for k in range(8):
                tmp_max = np.max(tmpG[k*4:k*4 + 4])
                Ek = np.floor(np.log2(tmp_max + 2**(-140))) 
                if ExpUD == 1:  
                    if tmp_max*2**(-Ek) >= 2 - 2**(-NR):        # RH updates EXP
                        Ek += 1
                E8[k] = Ek
            
            E2 = np.zeros(2)
            for k in range(2):
                E2[k] = np.max(E8[k*4:k*4+4])
            
            Emax = np.max(E2)

            E6_1 = Emax - 2             # level-1 Scaling, 1 6-bit exp vals: -46 ~ 16, 17 for NaN

            if E6_1 < -46:              # 17 for NaN
                E6_1 = -46

            if E6_1 > 16:
                E6_1 = 16

            E1_2 = E2 - E6_1 - 1     # level-2 Scaling, 2 1-bit exp vals: 0 ~ 1
            E1_2[E1_2 < 0] = 0

            E1P2 = E6_1 + E1_2       # sum of level-1 & 2 scaling, 2 exp vals
            E1_8 = np.append(E8[0:4] - E1P2[0], E8[4:8] - E1P2[1])          # level-3 Scaling, 8 1-bit exp vals: 0 ~ 1
            E1_8[E1_8 < 0] = 0

            E8G = np.append(E1_8[0:4] + E1P2[0], E1_8[4:8] + E1P2[1] )   # Fused 8 scaling exp vals: level-1 + level-2 + level + 3

            EG = np.zeros(G)    # Fused G scaling exp vals
            for k in range(8):
                EG[k*4:k*4+4] = E8G[k]

            in_grp = np.floor(tmpG*2**(-EG + Ng)+0.5)*2.0**(-Ng)    # 1PNg
            if ExpUD == 0:
                in_grp[in_grp == 2] = 2- 2**(-Ng)

            grp = S * in_grp * 2.0**EG

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFX_V7(x, N = 5, G = 32, ExpUD = 0): 
    # {E4 + 4*E1 + 8*E1} + 1PN: Three-level scaling
    # EXP = [-8, 8]
    # ExpUD = 1, Update Exp; ExpUD = 0, Clip In-Grp Data

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 2
    NR = N - 1

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values

            E8 = np.zeros(8)
            for k in range(8):
                tmp_max = np.max(tmpG[k*4:k*4 + 4])
                Ek = np.floor(np.log2(tmp_max + 2**(-140)))
                if ExpUD == 1:  
                    if tmp_max*2**(-Ek) >= 2 - 2**(-NR):        # RH updates EXP
                        Ek += 1
                E8[k] = Ek
            
            E4 = np.zeros(4)
            for k in range(4):
                E4[k] = np.max(E8[k*2:k*2+2])
            
            Emax = np.max(E4) 

            E4_1 = Emax - 2            # level-1 Scaling, 1 4-bit exp vals: -8 ~ 6, 7 for NaN

            if E4_1 < -8:              # 7 for NaN
                E4_1 = -8

            if E4_1 > 6:
                E4_1 = 6

            E1_4 = E4 - E4_1 - 1     # level-2 Scaling, 4 1-bit exp vals: 0 ~ 1
            E1_4[E1_4 < 0] = 0

            E1P2 = E4_1 + E1_4       # sum of level-1 & 2 scaling, 4 exp vals
            E1_8 = np.zeros(8)       # level-3 Scaling, 8 1-bit exp vals: 0 ~ 1
            for k in range(4): 
                E1_8[k*2:k*2+2] = E8[k*2:k*2+2] - E1P2[k]      
            E1_8[E1_8 < 0] = 0

            E8G = np.zeros(8)        # Fused 8 scaling exp vals: level-1 + level-2 + level + 3
            for k in range(4):
                E8G[k*2:k*2+2] = E1_8[k*2:k*2+2] + E1P2[k]

            EG = np.zeros(G)    # Fused G scaling exp vals
            for k in range(8):
                EG[k*4:k*4+4] = E8G[k]

            in_grp = np.floor(tmpG*2**(-EG + Ng)+0.5)*2.0**(-Ng)    # 1PNg
            if ExpUD == 0:
                in_grp[in_grp == 2] = 2 - 2**(-Ng)

            grp = S * in_grp * 2.0**EG

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFX_V8(x, N = 5, G = 32): 
    # E6M2 + 8*E1 + S.MN: Two-level scaling 
    # E = [-46, 17]

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 1

    # Bounds：Unified 1P7 Format, MSB = 1'b1
    if N != 2:
        # Constants for Boundaries (Also Update EXP), for HiF5/4/3
        bd1 = (1 - 2**(-N)) * 1.25          # >= bd1 RH --> 1 * 1.25
        bd2 = (1 - 2**(-N)) * 1.50          # >= bd2 RH --> 1 * 1.50
        bd3 = (1 - 2**(-N)) * 1.75          # >= bd3 RH --> 1 * 1.75
        bd4 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd4 RH --> 2 * 1.00 
    else: 
        # Constants for Boundaries (Also Update EXP), for HiF2 only 
        bd1 = (1 - 2**(-N)) * 1.50          # >= bd1 RH --> 1 * 1.50
        bd2 = (1 - 2**(-N)) * 1.75          # >= bd2 RH --> 1 * 1.75
        bd3 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd3 RH --> 2 * 1.00
        bd4 = (1 - 2**(-N)) * 1.25 * 2.00   # >= bd4 RH --> 2 * 1.25

    # Constants for 1 / M2P1:8-bit
    rec_1p00 = 1.0
    rec_1p01 = np.floor((1.0/1.25)*2.0**8)*2.0**(-8)
    rec_1p10 = np.floor((1.0/1.50)*2.0**8)*2.0**(-8)
    rec_1p11 = np.floor((1.0/1.75)*2.0**8)*2.0**(-8)

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values

            tmp8 = np.zeros(8)
            for k in range(8):
                tmp8[k] = np.max(tmpG[k*4:k*4+4])
            tmp_max = np.max(tmp8)

            Emax = np.floor(np.log2(tmp_max + 2**(-140)))
            max_igrp = tmp_max*2**(-Emax)     # [1, 2)

            M2P1 = 1.0
            rec = 1.0

            # original ulp = 2^(-(N-1)), RH-->original Error Bound (EUB) = 0.5 ulp
            if N != 2:  # HiF5/4/3
                if (max_igrp < bd1):
                    M2P1 = 1.25
                    rec = rec_1p01          # EUB: 1.25 ulp * 0.5 * 0.5 = 0.3125 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd4):     
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
            else:   # HiF2
                if (max_igrp < bd1):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
                elif (max_igrp >= bd4):   
                    M2P1 = 1.50
                    rec = rec_1p10
                    Emax = Emax + 1         # EUB: 1.5 ulp * 0.5 = 0.75 ulp (RH)

            E6 = Emax - 1      

            if E6 > 16:        # 17 for NaN
                E6 = 16

            if E6 < -46:
                E6 = -46

            DE8 = np.floor(np.log2(tmp8*2**(-E6)*rec + 2**(-140))) + 1    # DE8 = E8 - E6 <= 1
            DE8[DE8 < 0] = 0
            E1_8 = DE8

            E8G = E6 + E1_8     # Fused 8 scaling exp
            EG = np.zeros(G)    # Fused G scaling exp
            for k in range(8):
                EG[k*4:k*4+4] = E8G[k]

            in_grp = np.floor((tmpG*2.0**(-EG) * rec)*2.0**Ng + 0.5)*2.0**(-Ng) # 0PNg: Round Half to Away      
            in_grp[in_grp == 1] = 1 - 2**(-Ng)    # 0.xyyy --> 0.0yyy---> 0.yyy --> RH --> 1
            grp = S * M2P1 * in_grp * 2.0 ** EG

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFX_V9(x, N = 5, G = 32): 
    # E4M2 + 2*E1 + 8*E1 + S.MN: Three-level scaling 
    # E = [-8, 8]

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 1

    # Bounds：Unified 1P7 Format, MSB = 1'b1
    if N != 2:
        # Constants for Boundaries (Also Update EXP), for HiF5/4/3
        bd1 = (1 - 2**(-N)) * 1.25          # >= bd1 RH --> 1 * 1.25
        bd2 = (1 - 2**(-N)) * 1.50          # >= bd2 RH --> 1 * 1.50
        bd3 = (1 - 2**(-N)) * 1.75          # >= bd3 RH --> 1 * 1.75
        bd4 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd4 RH --> 2 * 1.00 
    else: 
        # Constants for Boundaries (Also Update EXP), for HiF2 only 
        bd1 = (1 - 2**(-N)) * 1.50          # >= bd1 RH --> 1 * 1.50
        bd2 = (1 - 2**(-N)) * 1.75          # >= bd2 RH --> 1 * 1.75
        bd3 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd3 RH --> 2 * 1.00
        bd4 = (1 - 2**(-N)) * 1.25 * 2.00   # >= bd4 RH --> 2 * 1.25

    # Constants for 1 / M2P1:8-bit
    rec_1p00 = 1.0
    rec_1p01 = np.floor((1.0/1.25)*2.0**8)*2.0**(-8)
    rec_1p10 = np.floor((1.0/1.50)*2.0**8)*2.0**(-8)
    rec_1p11 = np.floor((1.0/1.75)*2.0**8)*2.0**(-8)

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values

            tmp8 = np.zeros(8)
            for k in range(8):
                tmp8[k] = np.max(tmpG[k*4:k*4+4])
           
            tmp_max = np.max(tmp8)

            Emax = np.floor(np.log2(tmp_max + 2**(-140)))
            max_igrp = tmp_max*2**(-Emax)     # [1, 2)

            M2P1 = 1.0
            rec = 1.0

            # original ulp = 2^(-(N-1)), RH-->original Error Bound (EUB) = 0.5 ulp
            if N != 2:  # HiF5/4/3
                if (max_igrp < bd1):
                    M2P1 = 1.25
                    rec = rec_1p01          # EUB: 1.25 ulp * 0.5 * 0.5 = 0.3125 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd4):     
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
            else:   # HiF2
                if (max_igrp < bd1):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
                elif (max_igrp >= bd4):   
                    M2P1 = 1.50
                    rec = rec_1p10
                    Emax = Emax + 1         # EUB: 1.5 ulp * 0.5 = 0.75 ulp (RH)

            E4_1 = Emax - 2     # 1 level-1 exp val

            if E4_1 > 6:        # 7 for NaN
                E4_1 = 6

            if E4_1 < -8:
                E4_1 = -8
            
            DE8 = np.floor(np.log2(tmp8*2**(-E4_1)*rec + 2**(-140))) + 1   # DE8 = E8 - E4_1 <= 2
            DE8[DE8 < 0] = 0

            E1_2 = np.zeros(2)  # 2 level-2 exp vals
            E1_2[0] = 1 if np.max(DE8[0:4]) == 2 else 0
            E1_2[1] = 1 if np.max(DE8[4:8]) == 2 else 0

            E1_8 = np.append(DE8[0:4] - E1_2[0], DE8[4:8] - E1_2[1])       # 8 level-3 exp vals
            E1_8[E1_8 < 0] = 0

            E1P2 = E4_1 + E1_2  # 2 vals
            E8G = np.append(E1P2[0] + E1_8[0:4], E1P2[1] + E1_8[4:8])     # Fused 8 scaling exp

            EG = np.zeros(G)    # Fused G scaling exp
            for k in range(8): 
                EG[k*4:k*4+4] = E8G[k]

            in_grp = np.floor((tmpG*2.0**(-EG) * rec)*2.0**Ng + 0.5)*2.0**(-Ng) # 0PNg: Round Half to Away
            in_grp[in_grp == 1] = 1 - 2**(-Ng)    # clip for small-exp sub-grps
            grp = S * M2P1 * in_grp * 2.0 ** EG

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFX_V10(x, N = 5, G = 32): 
    # E4 + M2A + M2B + 8*E1 + S.MN: Two-level scaling 
    # E = [-8, 7]

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 1

    # Bounds：Unified 1P7 Format, MSB = 1'b1
    if N != 2:
        # Constants for Boundaries (Also Update EXP), for HiF5/4/3
        bd1 = (1 - 2**(-N)) * 1.25          # >= bd1 RH --> 1 * 1.25
        bd2 = (1 - 2**(-N)) * 1.50          # >= bd2 RH --> 1 * 1.50
        bd3 = (1 - 2**(-N)) * 1.75          # >= bd3 RH --> 1 * 1.75
        bd4 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd4 RH --> 2 * 1.00 
    else: 
        # Constants for Boundaries (Also Update EXP), for HiF2 only 
        bd1 = (1 - 2**(-N)) * 1.50          # >= bd1 RH --> 1 * 1.50
        bd2 = (1 - 2**(-N)) * 1.75          # >= bd2 RH --> 1 * 1.75
        bd3 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd3 RH --> 2 * 1.00
        bd4 = (1 - 2**(-N)) * 1.25 * 2.00   # >= bd4 RH --> 2 * 1.25

    # Constants for 1 / M2P1:8-bit
    rec_1p00 = 1.0
    rec_1p01 = np.floor((1.0/1.25)*2.0**8)*2.0**(-8)
    rec_1p10 = np.floor((1.0/1.50)*2.0**8)*2.0**(-8)
    rec_1p11 = np.floor((1.0/1.75)*2.0**8)*2.0**(-8)

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values

            tmp8 = np.zeros(8)
            for k in range(8):
                tmp8[k] = np.max(tmpG[k*4:k*4+4])
            tmp_maxA = np.max(tmp8[0:4])
            tmp_maxB = np.max(tmp8[4:8])

            EmaxA = np.floor(np.log2(tmp_maxA + 2**(-140)))
            max_igrpA = tmp_maxA*2**(-EmaxA)     # [1, 2)

            EmaxB = np.floor(np.log2(tmp_maxB + 2**(-140)))
            max_igrpB = tmp_maxB*2**(-EmaxB)     # [1, 2)

            M2P1A = 1.0
            recA = 1.0

            # original ulp = 2^(-(N-1)), RH-->original Error Bound (EUB) = 0.5 ulp
            if N != 2:  # HiF5/4/3
                if (max_igrpA < bd1):
                    M2P1A = 1.25
                    recA = rec_1p01          # EUB: 1.25 ulp * 0.5 * 0.5 = 0.3125 ulp (RH)
                elif (max_igrpA >= bd1) and (max_igrpA < bd2):
                    M2P1A = 1.50
                    recA = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrpA >= bd2) and (max_igrpA < bd3):
                    M2P1A = 1.75
                    recA = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrpA >= bd3) and (max_igrpA < bd4):
                    M2P1A = 1.00
                    recA = rec_1p00
                    EmaxA = EmaxA + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrpA >= bd4):     
                    M2P1A = 1.25
                    recA = rec_1p01
                    EmaxA = EmaxA + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
            else:   # HiF2
                if (max_igrpA < bd1):
                    M2P1A = 1.50
                    recA = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrpA >= bd1) and (max_igrpA < bd2):
                    M2P1A = 1.75
                    recA = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrpA >= bd2) and (max_igrpA < bd3):
                    M2P1A = 1.00
                    recA = rec_1p00
                    EmaxA = EmaxA + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrpA >= bd3) and (max_igrpA < bd4):
                    M2P1A = 1.25
                    recA = rec_1p01
                    EmaxA = EmaxA + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
                elif (max_igrpA >= bd4):   
                    M2P1A = 1.50
                    recA = rec_1p10
                    EmaxA = EmaxA + 1         # EUB: 1.5 ulp * 0.5 = 0.75 ulp (RH)

            M2P1B = 1.0
            recB = 1.0

            # original ulp = 2^(-(N-1)), RH-->original Error Bound (EUB) = 0.5 ulp
            if N != 2:  # HiF5/4/3
                if (max_igrpB < bd1):
                    M2P1B = 1.25
                    recB = rec_1p01          # EUB: 1.25 ulp * 0.5 * 0.5 = 0.3125 ulp (RH)
                elif (max_igrpB >= bd1) and (max_igrpB < bd2):
                    M2P1B = 1.50
                    recB = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrpB >= bd2) and (max_igrpB < bd3):
                    M2P1B = 1.75
                    recB = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrpB >= bd3) and (max_igrpB < bd4):
                    M2P1B = 1.00
                    recB = rec_1p00
                    EmaxB = EmaxB + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrpB >= bd4):     
                    M2P1B = 1.25
                    recB = rec_1p01
                    EmaxB = EmaxB + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
            else:   # HiF2
                if (max_igrpB < bd1):
                    M2P1B = 1.50
                    recB = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrpB >= bd1) and (max_igrpB < bd2):
                    M2P1B = 1.75
                    recB = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrpB >= bd2) and (max_igrpB < bd3):
                    M2P1B = 1.00
                    recB = rec_1p00
                    EmaxB = EmaxB + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrpB >= bd3) and (max_igrpB < bd4):
                    M2P1B = 1.25
                    recB = rec_1p01
                    EmaxB = EmaxB + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
                elif (max_igrpB >= bd4):   
                    M2P1B = 1.50
                    recB = rec_1p10
                    EmaxB = EmaxB + 1         # EUB: 1.5 ulp * 0.5 = 0.75 ulp (RH)

            Emax = np.max([EmaxA, EmaxB])     

            E4 = Emax - 1       # level-1 exp
            if E4 > 6:        # 7 for NaN
                E4 = 6

            if E4 < -8:
                E4 = -8



            DE8 = np.floor(np.log2(np.append(tmp8[0:4]*2**(-Emax)*recA, tmp8[4:8]*2**(-Emax)*recB) + 2**(-140)))    # <= -1
            E1_8 = np.zeros(8)      # level-2 exp
            E1_8[DE8 == -1] = 1

            E8G = E4 + E1_8     # Fused 8 scaling exp
            EG = np.zeros(G)    # Fused G scaling exp
            for k in range(8):
                EG[k*4:k*4+4] = E8G[k]

            igrp = tmpG*2.0**(-EG) # absolute value of grp
            in_grp = np.floor(np.append(igrp[0:16] * recA, igrp[16:32] * recB)*2.0**Ng + 0.5)*2.0**(-Ng) # 0PNg: Round Half to Away
            in_grp[in_grp == 1] = 1 - 2**(-Ng)    # clip for small-exp sub-grps
            grp = S * np.append(M2P1A * in_grp[0:16], M2P1B * in_grp[16:32]) * 2.0 ** EG

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFX_V11(x, N = 5, G = 64): 
    # E6M2 + 8*E1 + 16*E1 + S.MN: Four-level scaling 
    # E = [-46, 18]

    x = np.array(x)

    Mi = x.shape[0]
    Ni = x.shape[1]
    
    Mcnt = np.ceil(Mi/G).astype(int)

    res = np.zeros((Mi, Ni))
    Ng = N - 1

    # Bounds：Unified 1P7 Format, MSB = 1'b1
    if N != 2:
        # Constants for Boundaries (Also Update EXP), for HiF5/4/3
        bd1 = (1 - 2**(-N)) * 1.25          # >= bd1 RH --> 1 * 1.25
        bd2 = (1 - 2**(-N)) * 1.50          # >= bd2 RH --> 1 * 1.50
        bd3 = (1 - 2**(-N)) * 1.75          # >= bd3 RH --> 1 * 1.75
        bd4 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd4 RH --> 2 * 1.00 
    else: 
        # Constants for Boundaries (Also Update EXP), for HiF2 only 
        bd1 = (1 - 2**(-N)) * 1.50          # >= bd1 RH --> 1 * 1.50
        bd2 = (1 - 2**(-N)) * 1.75          # >= bd2 RH --> 1 * 1.75
        bd3 = (1 - 2**(-N)) * 1.00 * 2.00   # >= bd3 RH --> 2 * 1.00
        bd4 = (1 - 2**(-N)) * 1.25 * 2.00   # >= bd4 RH --> 2 * 1.25

    # Constants for 1 / M2P1:8-bit
    rec_1p00 = 1.0
    rec_1p01 = np.floor((1.0/1.25)*2.0**8)*2.0**(-8)
    rec_1p10 = np.floor((1.0/1.50)*2.0**8)*2.0**(-8)
    rec_1p11 = np.floor((1.0/1.75)*2.0**8)*2.0**(-8)

    for i in range(Mcnt):
        for j in range(Ni):

            ori = x[i*G:i*G+G,j]

            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)   # abs of grp values

            tmp16 = np.zeros(16)
            for k in range(16):
                tmp16[k] = np.max(tmpG[k*4:k*4+4])
            tmp_max = np.max(tmp16)

            Emax = np.floor(np.log2(tmp_max + 2**(-140)))
            max_igrp = tmp_max*2**(-Emax)     # [1, 2)

            M2P1 = 1.0
            rec = 1.0

            # original ulp = 2^(-(N-1)), RH-->original Error Bound (EUB) = 0.5 ulp
            if N != 2:  # HiF5/4/3
                if (max_igrp < bd1):
                    M2P1 = 1.25
                    rec = rec_1p01          # EUB: 1.25 ulp * 0.5 * 0.5 = 0.3125 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd4):     
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
            else:   # HiF2
                if (max_igrp < bd1):
                    M2P1 = 1.50
                    rec = rec_1p10          # EUB: 1.50 ulp * 0.5 * 0.5 = 0.375 ulp (RH)
                elif (max_igrp >= bd1) and (max_igrp < bd2):
                    M2P1 = 1.75
                    rec = rec_1p11          # EUB: 1.75 ulp * 0.5 * 0.5 = 0.4375 ulp (RH)
                elif (max_igrp >= bd2) and (max_igrp < bd3):
                    M2P1 = 1.00
                    rec = rec_1p00
                    Emax = Emax + 1         # EUB: 1.00 ulp * 0.5 = 0.5 ulp (RH)
                elif (max_igrp >= bd3) and (max_igrp < bd4):
                    M2P1 = 1.25
                    rec = rec_1p01
                    Emax = Emax + 1         # EUB: 1.25 ulp * 0.5 = 0.625 ulp (RH)
                elif (max_igrp >= bd4):   
                    M2P1 = 1.50
                    rec = rec_1p10
                    Emax = Emax + 1         # EUB: 1.5 ulp * 0.5 = 0.75 ulp (RH)

            E6_1 = Emax - 2      # 1 level-1 exp val

            if E6_1 > 16:        # [-46, 16], 17 for NaN
                E6_1 = 16

            if E6_1 < -46:
                E6_1 = -46
            
            DE16 = np.floor(np.log2(tmp16*2**(-E6_1)*rec + 2**(-140))) + 1   # DE16 = E16 - E6_1 <= 2
            DE16[DE16 < 0] = 0      # {0, 1, 2}

            DE8x2 = DE16 - 1        
            DE8x2[DE8x2 < 0] = 0    # {0, 1}

            E1_8x2 = np.zeros(16)   # {8 level-2 exp vals} * 2
            for k in range(8):
                E1_8x2[k*2:k*2+2] = np.max(DE8x2[k*2:k*2+2])

            E1_16 = DE16 - E1_8x2   # 16 level-3 exp vals 
            E1_16[E1_16 < 0] = 0    # {0, 1}

            E16G = E1_16 + E1_8x2 + E6_1    # Fused 16 Saling exp vals
            EG = np.zeros(G)                # Fused G scaling exp vals
            for k in range(16): 
                EG[k*4:k*4+4] = E16G[k]

            in_grp = np.floor((tmpG*2.0**(-EG) * rec)*2.0**Ng + 0.5)*2.0**(-Ng) # 0PNg: Round Half to Away
            in_grp[in_grp == 1] = 1 - 2**(-Ng)    # clip for small-exp sub-grps: 0.xyyy --> 0.0yyy---> 0.yyy --> RH --> 1
            grp = S * M2P1 * in_grp * 2.0 ** EG   

            res[i*G:i*G+G,j] = grp

    return res

def to_HiFX_V12(x, N = 5, G = 64): 
    # {E8 + 8*E1 + 16*E1} + 1PN: Three-level scaling
    # EXP = [-127, 127]

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

            EG = np.floor(np.log2(tmpG + 2**(-140))) 
            # print('EG', EG.flatten())

            E16 = np.zeros(16)
            for k in range(16):
                E16[k] = np.max(EG[k*4:k*4+4])

            E8 = np.zeros(8)
            for k in range(8):
                E8[k] = np.max(E16[k*2:k*2+2])
            
            Emax = np.max(E8)

            # level-1 exp
            E8_1 = Emax - 2     # [-127, 127]
            # print('E81', E8_1)

            # level-2 exp
            E1_8 = E8 - E8_1 - 1    # <= 1
            E1_8[E1_8 < 0] = 0      # [0, 1]
            # print('E1_8', E1_8.flatten())

            # level-3 exp
            E1_8x2 = np.zeros(16)
            for k in range(8):
                E1_8x2[k*2:k*2+2] = E1_8[k]
            
            E1_16 = E16 - E1_8x2 - E8_1  # <= 1
            E1_16[E1_16 < 0] = 0    # [0, 1]
            # print('E1_16', E1_16.flatten())

            # Restore grp vals
            E16G = E1_16 + E1_8x2 + E8_1    # Fused 16 exp vals
            # print('E16G', E16G.flatten())
            EG = np.zeros(G)                # Fused 64 exp vals
            for k in range(16):
                EG[k*4:k*4+4] = E16G[k]

            in_grp = np.floor(tmpG*2**(-EG + Ng)+0.5)*2.0**(-Ng)    # 1PNg
            in_grp[in_grp == 2] = 2- 2**(-Ng)

            grp = S * in_grp * 2.0**EG

            res[i*G:i*G+G,j] = grp

    return res

# M = 1024
# N = 1024

# # Xall = 2
# # G = 32
# # print("Test for HiF -",Xall," and G -",G)

# mu = 1.71
# delta = mu*0.05
# x = mu*np.random.randn(M,N) + np.random.uniform(-delta, delta,(M,N)) 
# print("mu for gaussian dist = ", mu)
# print("bound for normal dist = ", delta)

# E = np.floor(np.log2(np.abs(x) + 2**(-140)))
# print("Emax = ", np.max(E))
# print("Emin = ", np.min(E))


# Xmax = np.max(abs(x))
# Xmin = np.min(abs(x))
# XErange = np.log2(Xmax) - np.log2(Xmin)

# print("Xmax = ",Xmax)
# print("Xmin = ",Xmin)
# print("XErange = ", XErange)
# print("\n")

# SN = 3
# MX_MSE_all = np.zeros(SN)
# MX_RE_all = np.zeros(SN)
# MX_PNZ_all = np.zeros(SN)
# MX_Mbit_all = np.zeros(SN)

# TMX_G32_MSE_all = np.zeros(SN)
# TMX_G32_RE_all = np.zeros(SN)
# TMX_G32_PNZ_all = np.zeros(SN)
# TMX_G32_Mbit_all = np.zeros(SN)

# # TMX_G16_MSE_all = np.zeros(SN)
# # TMX_G16_RE_all = np.zeros(SN)
# # TMX_G16_PNZ_all = np.zeros(SN)
# # TMX_G16_Mbit_all = np.zeros(SN)

# V3_MSE_all = np.zeros(SN)
# V3_RE_all = np.zeros(SN)
# V3_PNZ_all = np.zeros(SN)
# V3_Mbit_all = np.zeros(SN)


# V4_MSE_all = np.zeros(SN)
# V4_RE_all = np.zeros(SN)
# V4_PNZ_all = np.zeros(SN)
# V4_Mbit_all = np.zeros(SN)

# # V5_MSE_all = np.zeros(SN)
# # V5_RE_all = np.zeros(SN)
# # V5_PNZ_all = np.zeros(SN)

# V6_MSE_all = np.zeros(SN)
# V6_RE_all = np.zeros(SN)
# V6_PNZ_all = np.zeros(SN)
# V6_Mbit_all = np.zeros(SN)

# # V7_MSE_all = np.zeros(SN)
# # V7_RE_all = np.zeros(SN)
# # V7_PNZ_all = np.zeros(SN)
# # V7_Mbit_all = np.zeros(SN)

# V8_MSE_all = np.zeros(SN)
# V8_RE_all = np.zeros(SN)
# V8_PNZ_all = np.zeros(SN)
# V8_Mbit_all = np.zeros(SN)

# # V9_MSE_all = np.zeros(SN)
# # V9_RE_all = np.zeros(SN)
# # V9_PNZ_all = np.zeros(SN)

# # V10_MSE_all = np.zeros(SN)
# # V10_RE_all = np.zeros(SN)
# # V10_PNZ_all = np.zeros(SN)

# # V11_MSE_all = np.zeros(SN)
# # V11_RE_all = np.zeros(SN)
# # V11_PNZ_all = np.zeros(SN)
# # V11_Mbit_all = np.zeros(SN)

# V12_MSE_all = np.zeros(SN)
# V12_RE_all = np.zeros(SN)
# V12_PNZ_all = np.zeros(SN)
# V12_Mbit_all = np.zeros(SN)

# X_all = [5,4,3,2]
# G16 = 16
# G32 = 32
# G64 = 64
# for i in range(SN): 
#     Xall = X_all[i]

#     print("Test for MX -", Xall, ", HiF - ", Xall)

#     ############################################################
#     print("****** Test for MX-FP ******")
#     print("Group_Size = ", G16)
#     f = to_MXN(x,Xall,G16,ExpUD=1) 
#     MSE = np.sum((f - x)**2)/(M*N)
#     RE = np.mean(np.abs((f-x)/x))
#     PNZ = np.mean(np.mean(f != 0.0))
#     M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     print("Mean Significand Bit = ", M_bit)
#     print("None-Zero Proportion: ", PNZ)
#     fmax = np.max(abs(f))
#     f[f==0] = np.inf
#     fmin = np.min(abs(f))
#     Erange = np.log2(fmax) - np.log2(fmin)
#     print("fmax = ",fmax)
#     print("fmin = ",fmin)
#     print("fErange = ", Erange)
#     print("MSE = ", MSE)
#     print("RE = ", RE)

#     MX_MSE_all[i] = MSE
#     MX_RE_all[i] = RE
#     MX_PNZ_all[i] = PNZ
#     MX_Mbit_all[i] = M_bit

#     ############################################################
#     print("****** Test for DaVinci HiFXG_V3 ******")
#     print("Group_Size = ", G16)
#     f, dE = to_HiFXG_V3(x,Xall,G16) 
#     MSE = np.sum((f - x)**2)/(M*N)
#     RE = np.mean(np.abs((f-x)/x))
#     PNZ = np.mean(np.mean(f != 0.0))
#     M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     print("Mean Significand Bit = ", M_bit)
#     print("None-Zero Proportion: ", PNZ)
#     fmax = np.max(abs(f))
#     f[f==0] = np.inf
#     fmin = np.min(abs(f))
#     Erange = np.log2(fmax) - np.log2(fmin)
#     print("fmax = ",fmax)
#     print("fmin = ",fmin)
#     print("fErange = ", Erange)
#     print("MSE = ", MSE)
#     print("RE = ", RE)

#     V3_MSE_all[i] = MSE
#     V3_RE_all[i] = RE
#     V3_PNZ_all[i] = PNZ
#     V3_Mbit_all[i] = M_bit

#     ############################################################
#     print("****** Test for DaVinci HiFXG_V4 ******")
#     print("Group_Size = ", G16)
#     f = to_HiFXG_V4(x,Xall,G16) 
#     MSE = np.sum((f - x)**2)/(M*N)
#     RE = np.mean(np.abs((f-x)/x))
#     PNZ = np.mean(np.mean(f != 0.0))
#     M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     print("Mean Significand Bit = ", M_bit)
#     print("None-Zero Proportion: ", PNZ)
#     fmax = np.max(abs(f))
#     f[f==0] = np.inf
#     fmin = np.min(abs(f))
#     Erange = np.log2(fmax) - np.log2(fmin)
#     print("fmax = ",fmax)
#     print("fmin = ",fmin)
#     print("fErange = ", Erange)
#     print("MSE = ", MSE)
#     print("RE = ", RE)

#     V4_MSE_all[i] = MSE
#     V4_RE_all[i] = RE
#     V4_PNZ_all[i] = PNZ
#     V4_Mbit_all[i] = M_bit

#     ############################################################
#     print("****** Test for TMX-G32 ******")
#     print("Group_Size = ", G32)
#     f = to_TMXN(x,Xall,G32,ExpUD=0) 
#     MSE = np.sum((f - x)**2)/(M*N)
#     RE = np.mean(np.abs((f-x)/x))
#     PNZ = np.mean(np.mean(f != 0.0))
#     M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     print("Mean Significand Bit = ", M_bit)
#     print("None-Zero Proportion: ", PNZ)
#     fmax = np.max(abs(f))
#     f[f==0] = np.inf
#     fmin = np.min(abs(f))
#     Erange = np.log2(fmax) - np.log2(fmin)
#     print("fmax = ",fmax)
#     print("fmin = ",fmin)
#     print("fErange = ", Erange)
#     print("MSE = ", MSE)
#     print("RE = ", RE)

#     TMX_G32_MSE_all[i] = MSE
#     TMX_G32_RE_all[i] = RE
#     TMX_G32_PNZ_all[i] = PNZ
#     TMX_G32_Mbit_all[i] = M_bit

#     # ############################################################
#     # print("****** Test for DaVinci HiFXG_V5 ******")
#     # print("Group_Size = ", G16)
#     # Emax = np.floor(np.log2(Xmax + 2.0**(-50)))
#     # glb_scale = 2.0 ** (6 - Emax)
#     # xin = x * glb_scale

#     # f, dE = to_HiFXG_V5(xin,Xall,G16) 
#     # f = f / glb_scale

#     # MSE = np.sum((f - x)**2)/(M*N)
#     # RE = np.mean(np.abs((f-x)/x))
#     # PNZ = np.mean(np.mean(f != 0.0))
#     # print("None-Zero Proportion: ", PNZ)
#     # PDE = np.mean(np.mean(dE != 0))
#     # print("None-zero DE2 Proporton: ", PDE)
#     # fmax = np.max(abs(f))
#     # f[f==0] = np.inf
#     # fmin = np.min(abs(f))
#     # Erange = np.log2(fmax) - np.log2(fmin)
#     # print("fmax = ",fmax)
#     # print("fmin = ",fmin)
#     # print("fErange = ", Erange)
#     # print("MSE = ", MSE)
#     # print("RE = ", RE)

#     # V5_MSE_all[i] = MSE
#     # V5_RE_all[i] = RE
#     # V5_PNZ_all[i] = PNZ

#     ############################################################
#     print("****** Test for DaVinci HiFX_V6 ******")
#     print("Group_Size = ", G32)
#     f = to_HiFX_V6(x,Xall,G32,ExpUD=0) 
#     MSE = np.sum((f - x)**2)/(M*N)
#     RE = np.mean(np.abs((f-x)/x))
#     PNZ = np.mean(np.mean(f != 0.0))
#     M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     print("Mean Significand Bit = ", M_bit)
#     print("None-Zero Proportion: ", PNZ)
#     fmax = np.max(abs(f))
#     f[f==0] = np.inf
#     fmin = np.min(abs(f))
#     Erange = np.log2(fmax) - np.log2(fmin)
#     print("fmax = ",fmax)
#     print("fmin = ",fmin)
#     print("fErange = ", Erange)
#     print("MSE = ", MSE)
#     print("RE = ", RE)

#     V6_MSE_all[i] = MSE
#     V6_RE_all[i] = RE
#     V6_PNZ_all[i] = PNZ
#     V6_Mbit_all[i] = M_bit

#     # ############################################################
#     # print("****** Test for DaVinci HiFX_V7 ******")
#     # print("Group_Size = ", G32)

#     # Emax = np.floor(np.log2(Xmax + 2.0**(-140)))
#     # glb_scale = 2.0 ** (7 - Emax)
#     # xin = x * glb_scale

#     # f = to_HiFX_V7(xin,Xall,G32,ExpUD=0)
#     # f = f / glb_scale 
#     # MSE = np.sum((f - x)**2)/(M*N)
#     # RE = np.mean(np.abs((f-x)/x))
#     # PNZ = np.mean(np.mean(f != 0.0))
#     # M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     # print("Mean Significand Bit = ", M_bit)
#     # print("None-Zero Proportion: ", PNZ)
#     # fmax = np.max(abs(f))
#     # f[f==0] = np.inf
#     # fmin = np.min(abs(f))
#     # Erange = np.log2(fmax) - np.log2(fmin)
#     # print("fmax = ",fmax)
#     # print("fmin = ",fmin)
#     # print("fErange = ", Erange)
#     # print("MSE = ", MSE)
#     # print("RE = ", RE)

#     # V7_MSE_all[i] = MSE
#     # V7_RE_all[i] = RE
#     # V7_PNZ_all[i] = PNZ
#     # V7_Mbit_all[i] = M_bit

#     ############################################################
#     print("****** Test for DaVinci HiFX_V8 ******")
#     print("Group_Size = ", G32)
#     f = to_HiFX_V8(x,Xall,G32) 
#     MSE = np.sum((f - x)**2)/(M*N)
#     RE = np.mean(np.abs((f-x)/x))
#     PNZ = np.mean(np.mean(f != 0.0))
#     M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     print("Mean Significand Bit = ", M_bit)
#     print("None-Zero Proportion: ", PNZ)
#     fmax = np.max(abs(f))
#     f[f==0] = np.inf
#     fmin = np.min(abs(f))
#     Erange = np.log2(fmax) - np.log2(fmin)
#     print("fmax = ",fmax)
#     print("fmin = ",fmin)
#     print("fErange = ", Erange)
#     print("MSE = ", MSE)
#     print("RE = ", RE)

#     V8_MSE_all[i] = MSE
#     V8_RE_all[i] = RE
#     V8_PNZ_all[i] = PNZ
#     V8_Mbit_all[i] = M_bit

#     # ############################################################
#     # print("****** Test for TMX-G16 ******")
#     # print("Group_Size = ", G16)
#     # f = to_TMXN(x,Xall,G16,ExpUD) 
#     # MSE = np.sum((f - x)**2)/(M*N)
#     # RE = np.mean(np.abs((f-x)/x))
#     # PNZ = np.mean(np.mean(f != 0.0))
#     # M_bit = 1-np.log2(np.mean(np.abs((f-x)*2**(-E))))
#     # print("Mean Significand Bit = ", M_bit)
#     # print("None-Zero Proportion: ", PNZ)
#     # fmax = np.max(abs(f))
#     # f[f==0] = np.inf
#     # fmin = np.min(abs(f))
#     # Erange = np.log2(fmax) - np.log2(fmin)
#     # print("fmax = ",fmax)
#     # print("fmin = ",fmin)
#     # print("fErange = ", Erange)
#     # print("MSE = ", MSE)
#     # print("RE = ", RE)

#     # TMX_G16_MSE_all[i] = MSE
#     # TMX_G16_RE_all[i] = RE
#     # TMX_G16_PNZ_all[i] = PNZ
#     # TMX_G16_Mbit_all[i] = M_bit

#     # ############################################################
#     # print("****** Test for DaVinci HiFX_V9 ******")
#     # print("Group_Size = ", G32)

#     # Emax = np.floor(np.log2(Xmax + 2.0**(-140)))
#     # glb_scale = 2.0 ** (7 - Emax)
#     # xin = x * glb_scale

#     # f = to_HiFX_V9(xin,Xall,G32)
#     # f = f / glb_scale 
#     # MSE = np.sum((f - x)**2)/(M*N)
#     # RE = np.mean(np.abs((f-x)/x))
#     # PNZ = np.mean(np.mean(f != 0.0))
#     # print("None-Zero Proportion: ", PNZ)
#     # fmax = np.max(abs(f))
#     # f[f==0] = np.inf
#     # fmin = np.min(abs(f))
#     # Erange = np.log2(fmax) - np.log2(fmin)
#     # print("fmax = ",fmax)
#     # print("fmin = ",fmin)
#     # print("fErange = ", Erange)
#     # print("MSE = ", MSE)
#     # print("RE = ", RE)
#     # print('\n')

#     # V9_MSE_all[i] = MSE
#     # V9_RE_all[i] = RE
#     # V9_PNZ_all[i] = PNZ

#     # ############################################################
#     # print("****** Test for DaVinci HiFX_V10 ******")
#     # print("Group_Size = ", G32)

#     # Emax = np.floor(np.log2(Xmax + 2.0**(-140)))
#     # glb_scale = 2.0 ** (6 - Emax)
#     # xin = x * glb_scale

#     # f = to_HiFX_V10(xin,Xall,G32)
#     # f = f / glb_scale 
#     # MSE = np.sum((f - x)**2)/(M*N)
#     # RE = np.mean(np.abs((f-x)/x))
#     # PNZ = np.mean(np.mean(f != 0.0))
#     # print("None-Zero Proportion: ", PNZ)
#     # fmax = np.max(abs(f))
#     # f[f==0] = np.inf
#     # fmin = np.min(abs(f))
#     # Erange = np.log2(fmax) - np.log2(fmin)
#     # print("fmax = ",fmax)
#     # print("fmin = ",fmin)
#     # print("fErange = ", Erange)
#     # print("MSE = ", MSE)
#     # print("RE = ", RE)
#     # print("\n")

#     # V10_MSE_all[i] = MSE
#     # V10_RE_all[i] = RE
#     # V10_PNZ_all[i] = PNZ

#     # ############################################################
#     # print("****** Test for DaVinci HiFX_V11 ******")
#     # print("Group_Size = ", G64)
#     # f = to_HiFX_V11(x,Xall,G64) 
#     # MSE = np.sum((f - x)**2)/(M*N)
#     # R