import numpy as np
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def RBF_ker(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) **2 / (2 * (sigma **2)))
def get_Ker(x,y,kernel,R_Var):
    if(kernel=='L'):
        return linear_kernel(x,y)
    elif(kernel=='P'):
        return polynomial_kernel(x,y,R_Var)
    elif(kernel=='R'):
        return RBF_ker(x,y,R_var)
