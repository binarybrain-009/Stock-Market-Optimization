# D-- Data list
# PD-- Price data
def EMA(D, N): #ith day included
    P_EMA = D.copy()
    k = 2/(N+1)
    E_0 = sum(D[:N])/N
    for i in range(N):
        P_EMA[i] = 0
    P_EMA[N-1] = E_0
    for i in range(N, len(D)):
         P_EMA[i] = D[i]*k+P_EMA[i-1]*(1-k)
    return P_EMA
def SMA(D,N): #ith  day  included SMA at day i= pi
    P_SMA = D.copy()
    for i in range(N):
        P_SMA[i] = 0
    for i in range(N-1, len(D)):
        P_SMA[i] = sum(D[i+1-N:i+1])/N
    return P_SMA
def ROC(PD,N): # x_t= {P_t-P_(t-N)}/P_(t-N)
     X= PD.copy()
     for i in range(N):
        X[i] = 0
     for t in range(N,len(PD)):
         X[t]=(PD[t]-PD[t-N])/PD[t-N]  #t_th day
     return X
def MACD(PD): # Ema12-Ema 26
    if(len(PD)<=26):
        print("ERROR: MACD can't be calculated")
        return PD
    else:
        E12=EMA(PD,12)
        E26=EMA(PD,26)
        M=PD.copy()
        for i in range(26-1): # -1 bcz indx start from 0
            M[i]=0
        for t in range(26-1,len(PD)):
            M[t]=E12[t]-E26[t]
        return M
def RSI(PD,N):
    Gain=[] # 0 bc 1st day NAN
    Loss=[]
    for i in range(1,len(PD)):
        change=PD[i]-PD[i-1]
        if(change>=0):
            Gain.append(change)
            Loss.append(0)
        else:
            Gain.append(0)
            Loss.append(-change)

    AvgG=SMA(Gain,N)
    AvgL=SMA(Loss,N)
    AvgL.insert(0,0)  # 0 bc 1st day NAN
    AvgG.insert(0,0)
    # print("G",Gain,"\n","L",Loss,"\n","AL",AvgL,"\n","AG",AvgG)
    RSI=PD.copy()
    for i in range(N):
        RSI[i] = 0
    for t in range(N,len(PD)):
        RSI[t]=100-(100/(1+(AvgG[t]/AvgL[t])))
    return RSI
def Will_R(PD,HP,LP,N): #PD-- closing price
    HHP=[0 for i in range(len(PD))]
    LLP=[0 for i in range(len(PD))]
    Will=[0 for i in range(len(PD))]
    for i in range(N-1,len(PD)):
        HHP[i]=max(HP[i+1-N:i+1])
        LLP[i]=min(LP[i+1-N:i+1])
        Will[i]=((PD[i]-HHP[i])/(HHP[i]-LLP[i]))*100
    print(HHP,"\n",LLP)
    return Will
def SO(PD,HP,LP,N,N1):
    HHP=[0 for i in range(len(PD))]
    LLP=[0 for i in range(len(PD))]
    SO=[0 for i in range(len(PD))]
    for i in range(N-1,len(PD)):
        HHP[i]=max(HP[i+1-N:i+1])
        LLP[i]=min(LP[i+1-N:i+1])
        SO[i]=((PD[i]-LLP[i])/(HHP[i]-LLP[i]))*100
    # print(HHP,"\n",LLP,"\n",SO)
    SO_SMA=SMA(SO[N-1:],N1)
    for i in range(N-1):
        SO_SMA.insert(i,0)
    return SO_SMA
def AD(PD,HP,LP,V,N):
    HHP=[0 for i in range(len(PD))]
    LLP=[0 for i in range(len(PD))]
    AD=[0 for i in range(len(PD))]
    for i in range(N-1,len(PD)):
        HHP[i]=max(HP[i+1-N:i+1])
        LLP[i]=min(LP[i+1-N:i+1])
        AD[i]=((2*PD[i]-LLP[i]-HHP[i])/(HHP[i]-LLP[i]))*V[i]
    return AD
def ADX(n):
    pass
def MFR(CD,LP,HP,V,N):
    RMF=[]
    TP=[]
    PRMF=[]
    NRMF=[]
    MFR=[0 for i in range(len(CD))]
    for i in range(len(CD)):
        TP.append((CD[i]+LP[i]+HP[i])/3)
        RMF.append(TP[i]*V[i])
        change=TP[i]-TP[i-1]
        if(i==0):
            PRMF.append(0)
            NRMF.append(0)
        elif(change>=0):
            PRMF.append(RMF[i])
            NRMF.append(0)
        else:
            NRMF.append(RMF[i])
            PRMF.append(0)
    for t in range(N,len(CD)):
        MFR[t]=(sum(PRMF[t+1-N:t+1])/sum(NRMF[t+1-N:t+1]))
    return MFR


def MFI(CD,LP,HP,V,N):
    MFR=MFR(CD,LP,HP,V,N)
    MFI=[0 for i in range(len(CD))]
    for t in range(N,len(CD)):
        MFI[t]=100-(100/(1+MFR))
    return MFI



if __name__=="__main__":
    '''h=[127.01,
128.01,
129.01,
130.01,
131.01,
132.01,
133.01,
134.01,
135.01,
136.01,
137.01,
138.01,
139.01,
140.01,
141.01,
142.01,
143.01,
144.01,
145.01,
146.01,
147.01,
148.01,
149.01,
150.01,
151.01,
152.01,
153.01,
154.01,
155.01,
156.01,]
    l=[125.36,
126.16,
124.93,
126.09,
126.82,
126.48,
126.03,
124.83,
126.39,
125.72,
124.56,
124.57,
125.07,
126.86,
126.63,
126.80,
126.71,
126.80,
126.13,
125.92,
126.99,
127.81,
128.47,
128.06,
127.61,
127.60,
127.00,
126.90,
127.49,
127.40,
]
    DL=[
    0,0,0,0,0,0,0,0,0,0,0,0,0,
    127.29,
127.18,
128.01,
127.11,
127.73,
127.06,
127.33,
128.71,
127.87,
128.58,
128.60,
127.93,
128.11,
127.60,
127.60,
128.69,
128.27]'''
    import yfinance as yf
    df = yf.download(['RELIANCE.NS'],start="2019-12-01",end="2020-01-01")
    print(len(df),df,df['Close'])
    # print(SO(DL,h,l,14,3))
