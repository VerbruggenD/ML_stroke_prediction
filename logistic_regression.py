# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np
np.set_printoptions(suppress=True)

import csv
import sys

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# we use pandas to import a comma-seperated values dataset
import pandas as pd

def sigmoid(z):
    
    # convert input to a numpy array
    z = np.array(z)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = 1/(1 + np.exp(-z))

    # =============================================================
    return g


def lrcostFunctionReg(theta, X, y, lambda_):
    #Initialize some useful values
    m = y.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    
    # ====================== YOUR CODE HERE ======================

    H = sigmoid(np.matmul(X, theta))
    J = (sum(-y*np.log(H)-(1-y)*np.log(1-H)))/(m)+((lambda_/(2*m))*sum(theta[1:]**2))
    
    diff = np.subtract(H, y)
    
    grad = ((np.matmul(np.transpose(X), diff))/m) + ((lambda_ * theta)/m)
    grad[0]= ((np.matmul(np.transpose(X), diff))/m)[0]
        
    # =============================================================
    return J, grad

def lrOptimization(lrcostFunctionReg, initial_theta, X, y, _lambda):
    # set options for optimize.minimize
    options= {'maxiter': 500}

    # see documention for scipy's optimize.minimize  for description about
    # the different parameters
    # The function returns an object `OptimizeResult`
    # We use truncated Newton algorithm for optimization which is 
    # equivalent to MATLAB's fminunc
    # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy

    res = optimize.minimize(lrcostFunctionReg,
                            initial_theta,
                            (X, y, _lambda),
                            jac=True,
                            method='TNC',
                            options=options)

    #result = optimize.fmin_tnc(func=lrCostFunction, x0=initial_theta, args=(X, y, _lambda))

    # the fun property of `OptimizeResult` object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property
    theta = res.x

    #theta = result[0]
    (cost, grad) = lrcostFunctionReg(theta, X, y, _lambda)

    # Print theta to screen
    #print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))

    #print('theta:')
    #print('\t[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(*theta))

    return theta, cost, grad

def predict(theta, X, percentile):
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================

    p = sigmoid(np.matmul(X, theta))
    
    #np.set_printoptions(threshold=sys.maxsize)
    #print(p)
    
    p [p >= percentile] = 1
    p [p < percentile]  = 0
    
    
    # ============================================================
    return p