import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from cvxopt.modeling import variable,op
import cvxopt.solvers
# df=pd.read_excel(r"C:\Users\lenovo\Dropbox\PC\Documents\firstproject\nifty50_daily_2015-present.xlsx").iloc[1305:,1:17]
# df1=pd.read_excel(r"C:\Users\lenovo\Dropbox\PC\Documents\firstproject\Nifty_index2015-present.xlsx").iloc[1305:,1:17]
df=pd.read_excel(r"C:\Users\Ruchika\Desktop\Simrandeep\dowjones30daily.xlsx").iloc[852:952,2:]
df1=pd.read_excel(r"C:\Users\Ruchika\Desktop\Simrandeep\dowjones30daily.xlsx").iloc[852:952,1:2]
print(df)
print(df1)
b=np.array(df1.iloc[:,0:1])
# print("b",b)
p=([1/len(df) for i in range(len(df))])
print("value of p ",p)
r=np.array(df)
#print("r",r)
#A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
#b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
#c = matrix([ 2.0, 1.0 ])
#sol=solvers.lp(c,A,b)
#print(sol['x'])
def v_k(b,k):
   result1 = 0
   for t in range(len(df)):
        # result1=result1+max(0,(b[k][0]-b[t][0])*p[t])
        result1=result1+max(0,(b[k]-b[t])*p[t])
   return result1
C1=np.zeros((1,len(df)**2+len(df.columns)+1))
def sumrtpt(r,p,n):
    sum=0
    for t in range(len(df)):
        sum+=r[t][n]*p[t]
    return sum
for s in range(1):
    for k in range(len(df)**2+len(df.columns)+1):
        if(s==0 and k==0):
            C1[s][k]=1
        elif(s==0 and k<=len(df.columns)):
            C1[s][k]= -sumrtpt(r,p,k-1)
        elif(s==0 and k>len(df.columns)):
            C1[s][k]=0
print("C1",C1)
C2=np.zeros((len(df)**2,len(df.columns)+1))

def sumrt(r,n):
    sum=0
    for t in range(len(df)):
        sum+=r[t][n]
        return sum
n=len(df.columns)
T=len(df)
divT = -1
for s in range(T**2):
    if s%T==0:
        divT+=1
    for k in range(len(df.columns)+1):
        if(0<k<=n):
            C2[s][k]=-r[divT][k-1]
print(C2)
C2_Identity = -np.identity(len(df)**2)
C2 = np.hstack((C2,C2_Identity))
print("C2",C2)
#
C3=np.zeros((T,n+1))
for t in range(len(df)):
    C3 = np.hstack((C3,p[t]*(np.identity(len(df)))))
print("C3",C3)
C4=np.hstack((np.zeros((T**2,n+1)),-np.identity(T**2)))
print("C4",C4)
C5=np.hstack((-np.identity((n)),np.zeros((n,T**2))))
C5=np.hstack((np.zeros((n,1)),C5))
print("C5",C5)
A=np.vstack((C1,C2))
A=np.vstack((A,C3))
A=np.vstack((A,C4))
A=np.vstack((A,C5))
print("A",A)
#lets make b
#1
b1=[0.0]
#2
#b2=np.zeros((T**2,1))
b2=cvxopt.matrix(b)
b21=cvxopt.matrix(b)
for i in range(T-1):
    b2=-np.vstack((b2,b21))
print("b2",b2)
# divT=-1
# for i in range(T**2):
#     if(i%T==0):
#         divT+=1
#     b2[i]=-b[divT]
# print("b2", b2)
#3
b3=np.zeros((T,1))
for i in range(T):
    b3[i]=v_k(b,i)
print("b3",b3)
#4
b4=np.zeros((T**2,1))
#5
b5=np.zeros((n,1))
#B
B=np.vstack((b1,b2))
B=np.vstack((B,b3))
B=np.vstack((B,b4))
B=np.vstack((B,b5))
print("B",B)
A=cvxopt.matrix(A)
B=cvxopt.matrix(B)
# define C
c=np.zeros((T**2+n+1,1))
c[0][0]=-1.0
c=cvxopt.matrix(c)
print(c)
# G define
G=np.zeros((1,T**2+n+1))
for i in range(n):
    G[0][i+1]=1.0
print("G",G)
h=[1.0]
G=cvxopt.matrix(G)
h=cvxopt.matrix(h)
sol=solvers.lp(c,A,B,G,h)
print(sol['x'])
# X=np.array(sol['x'])
# l=[]
# for x in range(1,len(df.columns)+1):
#     l.append(X[x][0])
# print(l)
# print(sum(l))
# l1=[]
# for x in range(len(df.columns)+1,(T**2)+n+1):
#     l1.append(X[x][0])
# print(l1)
# theta=print(X[0][0])
# df2 = pd.DataFrame(l)
# print(df2)
# data = pd.DataFrame({'xi': l, 'd_tk': l1,'theta': theta})
# print(data)
# C6=-np.identity(len(df)**2+len(df.columns)+1)
# print(C6)
#b=[]
#c=[]
#sol=solvers.lp(c,A,b)
#print(sol['x'])
# for s in range(T**2):
#     for k in range((T**2)+1):
#         if(s!=0 and k!=0):
#             C3[s][k]=C3_p(k-1)
#         else:
#             C3[s][k]=0
# h1=cvxopt.matrix(np.hstack(C3_p,C3))
# def sumdtk(n):
#     sum=0
#     for t in range(len(df)):
#         sum+=d_t_k[t][n]
#         return sum
# for s in range(1):
#     for k in range(len(df)**2+len(df.columns)+1):
#         if(s==0 and k==0):
#             C3[s][k]=0
#         elif(s!=0 and k>=0):
#             C3[s][k]=s1
#         elif(s==0 and k>len(df.columns)):
#             C3[s][k]=0
# C4=np.zeros(1,len(df)**2+len(df.columns)+1)
# for s in range(1):
#     for k in range(len(df)**2+len(df.columns)+1):
#         if(s==0 and k==0):
#             C4[s][k]=0
#         elif(s!=0 and k>=0):
#             C4[s][k]=-d_t_k
#         elif(s==0 and k>len(df.columns)):
#             C4[s][k]=0
# C5=np.zeros(1,len(df)**2+len(df.columns)+1)
# def sumx(n):
#     sum=0
#     for t in range(len(df)):
#         sum+=x[n]
#         return sum
# for s in range(1):
#     for k in range(len(df)**2+len(df.columns)+1):
#         if(s==0 and k==0):
#             C5[s][k]=0
#         if(s==0 and k<len(df.columns)):
#             C5[s][k]=sumx(k-1)
#         elif(s==0 and k>len(df.columns)):
#             C5[s][k]=0
# #A=matrix([1,C1[s][k]],[])
# C=np.zeros(1,len(df)**2+len(df.columns)+1)
# for s in range(1):
#     for k in range(len(df)**2+len(df.columns)+1):
#         if(s==0 and k==0):
#             C[s][k]=1
#         else:
#             C[s][k]=0
# B=np.zeros(1,len(df)**2+len(df.columns)+1)
