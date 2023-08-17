import yfinance as yf
import numpy as np
import pandas as pd
import gurobipy as gp
# Taking data of aaple and CAT
#df = yf.download(['LT.NS','INFY.NS','TATAMOTORS.NS','WIPRO.NS','RELIANCE.NS','BAJFINANCE.NS','M&M.NS','BRITANNIA.NS','NESTLEIND.NS','ASIANPAINT.NS'],start="2022-25-02",end="2022-28-02")
#df1 = yf.download(['AMZN'],start="2019-12-01",end="2019-12-10")
#df=pd.read_excel("10companies_Nfty.xlsx").iloc[:30,1:]
#df1=pd.read_excel("NFTY50_150days.xlsx").iloc[:30,1:2]
# df=pd.read_excel(r"C:\Users\lenovo\Dropbox\PC\Documents\firstproject\10companies_nfty.xlsx").iloc[:3,1:]
# df1=pd.read_excel(r"C:\Users\lenovo\Dropbox\PC\Documents\firstproject\NFTY50_150days.xlsx").iloc[:3,1:2]
df=pd.read_excel(r"C:\Users\Ruchika\Downloads\1.xlsx").iloc[1302:1305,3:15]
df1=pd.read_excel(r"C:\Users\Ruchika\Downloads\1.xlsx").iloc[1302:1305,1:2]
# daily percentage Return
# df=df['Close'].pct_change()
# df=df.iloc[1:,:]
print(df)
print("le",len(df))
# df1=df1['Close'].pct_change()
# df1=df1.iloc[1:]
print(df1)
#b=np.array(df1.iloc[:,0:1])
b=np.array(df1.iloc[:,0:1])
print("bench",b)
# r=np.array(df)
# print(b)
p=([1/len(df) for i in range(len(df))])
print(p)
def v_k(b,k):
   result1 = 0
   for t in range(len(df)):
        result1=result1+max(0,(b[k][0]-b[t][0])*p[t])
   return result1
print("vk",v_k(b,0))
def constraint_1_doublesum(x):
    result=0
    x=x.to_numpy()
    for t in range(len(df)):
            result=result+np.dot(df.iloc[t],x) *p[t]
    return result
#print(v_k(b))
# x=pd.Series([0.3,0.7,0,0,0,0,0,0,0])
# x=x.to_numpy()
# print("return",np.dot(r[0],x))
#implementing LPSSD
m = gp.Model("lp")
x = pd.Series(m.addVars(len(df.columns),lb=0))
d_m=[]
for i in range(len(df)):
    d=pd.Series(m.addVars(len(df),lb=0))
    d_m.append(d)
r=np.matrix(df)
print(r)
theta = m.addVar(lb= -float('inf'),name= "theta")

# x=pd.Series([1,2])
# p=1/2
# print((np.dot(r,x)[0,0])*p)
# Set the objective function
m.setObjective(theta, gp.GRB.MAXIMIZE)
#
# #Add Constraints
m.addConstr(sum(x)==1)
m.addConstr(theta <= constraint_1_doublesum(x))

for t in range(len(df)):
    for k in range(len(df)): #t=0,k=0
         m.addConstr( b[k][0]-np.dot(df.iloc[t],x)<= d_m[t][k])
# x=np.array([1,2])
# print(df.iloc[0])
# print(np.dot(df.iloc[0],x))

for k in range(len(df)):
     m.addConstr(np.dot(d_m[k],p)<=v_k(b,k))

# Solve the model

m.optimize()
print("Optimal Solution:",theta)
print("x:",x)
#print("d_m:",d_m)

