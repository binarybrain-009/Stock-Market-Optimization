import numpy as np
# For optimization
from scipy import linalg
from scipy.optimize import Bounds, BFGS
from scipy.optimize import LinearConstraint, minimize
from Kernels import *
ZERO = 1e-5
def lagrange_dual(alpha, x, t,Kernel,K_Var):
    result = 0
    for i in range(len(x)):
        for k in range(len(x)):
            result = result + alpha[i] * alpha[k] * t[i] * t[k] * get_Ker(x[i],x[k],Kernel,K_Var)
    result = -sum(alpha) +0.5 * result
    return result
#get Kernel

def optimize_alpha(x, t,Kernel,K_Var,C,s):
    m, n = x.shape
    np.random.seed(1)
    # Initialize alphas to random values
    alpha_0 = np.random.rand(m)
    # Define the constraint
    linear_constraint = LinearConstraint(t, [0], [0])
    # Define the bounds
    bounds_alpha = Bounds(np.zeros(m), s*C)
    # Find the optimal value of alpha
    result = minimize(lagrange_dual, alpha_0, args=(x, t,Kernel,K_Var), method='trust-constr',
                      hess=BFGS(), constraints=[linear_constraint],
                      bounds=bounds_alpha)
    # The optimized value of alpha lies in result.x
    alpha = result.x
    return alpha
# def get_w(alpha, t, x):
#     m = len(x)
#     # Get all support vectors
#     w = np.zeros(x.shape[1])
#     for i in range(m):
#         w = w + alpha[i] * t[i] * x[i, :]
#     return w
def phi_x(x_test_i,x,t,alpha,kernel,K_Var): # x,y is the training data set
    ind_sv = np.where((alpha > ZERO))[0]
    s=0
    for i in ind_sv:
        s=s+alpha[i]*t[i]*get_Ker(x[i],x_test_i,kernel,K_Var)
    return s

def get_b(alpha, t, x, C,kernel,K_Var):
    C_numeric = C - ZERO
    # Indices of support vectors with alpha<C
    ind_sv = np.where((alpha > ZERO) & (alpha < C_numeric))[0]
    b = 0.0
    for s in ind_sv:
        b = b + (-t[s] + phi_x(x[s],x,t,alpha,kernel,K_Var))
    # Take the average
    b = b / len(ind_sv)
    return b

def classify_points(x_test,x,t,alpha,w0,kernel,K_Var):
    # get y(x_test)
    predicted_labels=[]
    for x in x_test:
        for i in self.SVind:
        predicted_labels.append(phi_x(x_test[i],x,t,alpha,kernel,K_Var)-w0)
    predicted_labels = np.array(predicted_labels)
    predicted_labels = np.sign(predicted_labels)
    # Assign a label arbitrarily a +1 if it is zero
    predicted_labels[predicted_labels == ZERO] = 1
    return predicted_labels


def misclassification_rate(labels, predictions):
    total = len(labels)
    errors = sum(labels != predictions)
    return errors / total * 100
def SplitTrainTest(x,t,p):
    x1=x[:int((len(x)/100)*p)]
    t=t[:int((len(x)/100)*p)]
    return x1,t
def test(x,t,p):
    x1=x[int((len(x)/100)*p):]
    t=t[int((len(x)/100)*p):]
    return x1,t
def get_siMember(x,sigma):
    l=len(x)
    t=[i for i in range(len(x)+1) ]
    s=[]
    for i in range(1,len(x)+1):
        si=((1-sigma)/(t[l]-t[1]))*t[i]+ (t[l]*sigma-t[1])/(t[l]-t[1])
        s.append(si)
    s=np.array(s)
    return s
def display_fuzzySVM_result(x, t, C,p,F_sigma,kernel,K_Var=None,M='FS'): #trainTestPercentage
    x_test,t_test=test(x,t,p)
    x_train,t_train=train(x,t,p)
    # Get the alphas
    if(M=='S'):
        s=np.array([1 for i in range(len(x_train))])
    elif(M=='FS'):
        s=get_siMember(x_train,F_sigma)
    alpha = optimize_alpha(x_train, t_train,kernel,K_Var, C,s)
    print("alpha",alpha)
    # Get the weights
    # w = get_w(alpha, t_train, x_train)
    # print("wieghts",w)
    w0 = get_w0(alpha, t_train, x_train, C,kernel,K_Var)
    print("b",w0)
    # Get the misclassification error and display it as title
    predictions = classify_points(x_test,x_train,t_train,alpha,w0,kernel,K_Var)
    print("prediction",predictions)
    err = misclassification_rate(t_test, predictions)
    print( 'C = ' + str(C) + ',  Errors: ' + '{:.1f}'.format(err) + '%')
    print( ',  total SV = ' + str(len(alpha[alpha > ZERO])))
