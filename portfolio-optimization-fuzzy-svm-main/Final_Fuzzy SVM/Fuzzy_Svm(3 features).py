import numpy as np
# For optimization
from scipy import linalg
from scipy.optimize import Bounds, BFGS
from scipy.optimize import LinearConstraint, minimize

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def RBF_ker(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) * 2 / (2 * (sigma * 2)))

# For plotting
import matplotlib.pyplot as plt
# import seaborn as sns
# For generating dataset
# import sklearn.datasets as dt
ZERO = 1e-5
# Objective function
def lagrange_dual(alpha, x, t):
    result = 0
    for i in range(len(x)):
        for k in range(len(x)):
            result = result + alpha[i] * alpha[k] * t[i] * t[k] * RBF_ker(x[i,:],x[k,:],0.5)
    result = sum(alpha) - 0.5 * result
    return result


def optimize_alpha(x, t,C,s):
    m, n = x.shape
    np.random.seed(1)
    # Initialize alphas to random values
    alpha_0 = np.random.rand(m) * C
    # Define the constraint
    linear_constraint = LinearConstraint(t, [0], [0])
    # Define the bounds
    bounds_alpha = Bounds(np.zeros(m), s*C)
    # Find the optimal value of alpha
    result = minimize(lagrange_dual, alpha_0, args=(x, t), method='trust-constr',
                      hess=BFGS(), constraints=[linear_constraint],
                      bounds=bounds_alpha)
    # The optimized value of alpha lies in result.x
    alpha = result.x
    return alpha
def get_w(alpha, t, x):
    m = len(x)
    # Get all support vectors
    w = np.zeros(x.shape[1])
    for i in range(m):
        w = w + alpha[i] * t[i] * x[i, :]
    return w
def get_w0(alpha, t, x, w, C):
    C_numeric = C - ZERO
    # Indices of support vectors with alpha<C
    ind_sv = np.where((alpha > ZERO) & (alpha < C_numeric))[0]
    w0 = 0.0
    for s in ind_sv:
        w0 = w0 + (-t[s] + np.dot(x[s, :], w))
    # Take the average
    w0 = w0 / len(ind_sv)
    return w0

def classify_points(x_test, w, w0):
    # get y(x_test)
    predicted_labels = np.sum(x_test * w, axis=1) +w0
    predicted_labels = np.sign(predicted_labels)
    # Assign a label arbitrarily a +1 if it is zero
    predicted_labels[predicted_labels == ZERO] = 1
    return predicted_labels


def misclassification_rate(labels, predictions):
    total = len(labels)
    errors = sum(labels != predictions)
    return errors / total * 100

def display_fuzzySVM_result(x, t, C):
    x_test,t_test=test(x,t)
    x_train,t_train=train(x,t)
    # Get the alphas
    s=get_siMember(x_train)
    alpha = optimize_alpha(x_train, t_train, C,s)
    print("alpha",alpha)
    # Get the weights
    w = get_w(alpha, t_train, x_train)
    print("wieghts",w)
    w0 = get_w0(alpha, t_train, x_train, w, C)
    print("b",w0)
    # Get the misclassification error and display it as title
    predictions = classify_points(x_test, w, w0)
    print("prediction",predictions)
    err = misclassification_rate(t_test, predictions)
    print( 'C = ' + str(C) + ',  Errors: ' + '{:.1f}'.format(err) + '%')
    print( ',  total SV = ' + str(len(alpha[alpha > ZERO])))
p=90
def train(x,t):
    x1=x[:int((len(x)/100)*p)]
    t=t[:int((len(x)/100)*p)]
    return x1,t
def test(x,t):
    x1=x[int((len(x)/100)*p):]
    t=t[int((len(x)/100)*p):]
    return x1,t
def get_siMember(x,sigma=100):
    l=len(x)
    t=[i for i in range(len(x)+1) ]
    s=[]
    for i in range(1,len(x)+1):
        si=((1-sigma)/(t[l]-t[1]))*t[i]+ (t[l]*sigma-t[1])/(t[l]-t[1])
        s.append(si)
    s=np.array(s)
    return s

import yfinance as yf

# Taking data of aaple and CAT
df = yf.download(['RELIANCE.NS'],start="2019-12-01",end="2020-05-01")
print(len(df))
# daily percentage Return
df=np.log(1+df['Adj Close'].pct_change())
df=df.iloc[1:]
x1=list(df)
print(x1)
def EMA(PO,N):
    P_EMA=PO.copy()
    k=2/(N+1)
    E_0=sum(PO[:N])/N

    for i in range(N):
        P_EMA[i]=0
    P_EMA[N-1]=E_0
    for i in range(N,len(PO)):
         P_EMA[i]=PO[i]*k+P_EMA[i-1]*(1-k)
    return P_EMA
def SMA(PO,N):
    P_SMA=PO.copy()
    for i in range(N):
        P_SMA[i]=0
    for i in range(N,len(PO)):
        P_SMA[i]=sum(PO[i-N:i])/N
    return P_SMA

x3=EMA(x1,5)
print(x3)

N=5
M=1
x2=SMA(x1,5)[5:]

print(x2)
R=x1[N-1:len(x1)-M] # M is the no. days

print(x1)
X=[]
for i in range(len(R)):
    X.append([R[i],x2[i],x3[i]])
print("data",X)
y=[]
for i in range(N,len(x1)):
    if x1[i] >= 0.005:
        y.append(1)
    else:
        y.append(-1)
print(len(y),len(X))
print("The y is :", y)
dat = np.array(X)
labels = np.array(y)
print("train=",(len(dat)/100)*p,"test=",(len(dat)/100)*(100-p))
C=0.125
 #print(len(get_siMember(dat)*C))
#for C in range(1,500,20):

display_fuzzySVM_result(dat,labels,C,s)
