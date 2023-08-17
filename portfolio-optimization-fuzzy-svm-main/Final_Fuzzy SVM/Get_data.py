import yfinance as yf
import numpy as np
from Features import *
from SVM_Fuzzy import *
# Taking data of aaple and CAT
df = yf.download(['RELIANCE.NS'],start="2019-12-01",end="2020-01-01")
print(len(df),df,df['Close'])
# daily percentage Return
R=df['Close'].pct_change()
R=R.iloc[1:]
PD=list(PD)
#closing price
PD=df['Close']
PD=list(PD)
#high
H=df['High']
#LOW
L=df['Low']
#volume
V=df['Volume']


x1=list(df)
x3=EMA(x1,5)
N=5
M=1
x2=SMA(x1,5)
print(x2)

R=x1[N-1:len(x1)-M] # M is the no. days


X=[]
for i in range(len(R)):
    X.append([R[i],x2[i],x3[i]])

y=[]
for i in range(N,len(x1)):
    if x1[i] >= 0.001:
        y.append(1)
    else:
        y.append(-1)
print("X=",X)
print("y=",y)
dat = np.array([[1,8],[4,5],[1,1],[8,3],[4,4]])
labels = np.array([1,1,1,-1,1])
C=200
p=100
F_sigma=0.2
kernel='L'
display_fuzzySVM_result(dat,labels,C,p,F_sigma,kernel,M='S')
# s=np.array([1 for i in range(len(dat[:4]))])
# s=get_siMember(dat[:4],F_sigma)
# alpha=optimize_alpha(dat[:4],labels[:4],kernel,None,C,s)
# print(alpha)
# b=get_w0(alpha,labels[:4],dat[:4],C,kernel,None)
# print(b)
# P=classify_points(dat[4:],dat[:4],labels[:4],alpha,b,kernel,None)
# print(P)

