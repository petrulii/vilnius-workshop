
""" Logistic Regression Problem """

import numpy as np
import csv
from sklearn import preprocessing

file = open('ionosphere.data')
# Dimension of the feature vector
d = 34
# Variable size + intercept
n = d+1
# Number of examples
m = 351
# Regularization hyperparameter (best is 0.001)
lam2 = 0.001
# All feature vectors
A = np.zeros((m,d))
# All labels
b = np.zeros(m)
# Read the feature vectors and the labels from the data file
reader = csv.reader(file, delimiter=',')
i = 0
for row in reader:
    A[i] = np.array(row[:d]) 
    if row[d] == 'b':
        b[i] = -1.0
    else:
        b[i] =  1.0 
    i+=1

scaler = preprocessing.StandardScaler().fit(A)
A = scaler.transform(A)  

# Adding an intercept
A_inter = np.ones((m,n))
A_inter[:,:-1] = A
A = A_inter

# L-smoothness constant
L = 0.25*max(np.linalg.norm(A,2,axis=1))**2 + lam2


def f(x):
    """ Gives the value of f wrt x """
    l = 0.0
    for i in range(A.shape[0]):
        if b[i] > 0 :
            l += np.log( 1 + np.exp(-np.dot( A[i] , x ) ) ) 
        else:
            l += np.log( 1 + np.exp(np.dot( A[i] , x ) ) ) 
    return l/m + lam2/2.0*np.dot(x,x)


def f_grad(x):
    """ Gives the gradient of f wrt x """
    g = np.zeros(n)
    for i in range(A.shape[0]):
        if b[i] > 0:
            g += -A[i]/( 1 + np.exp(np.dot( A[i] , x ) ) ) 
        else:
            g += A[i]/( 1 + np.exp(-np.dot( A[i] , x ) ) ) 
    return g/m + lam2*x


def f_grad_hessian(x):
    """ Gives the Hessian of f wrt x """
    g = np.zeros(n)
    H = np.zeros((n,n))
    for i in range(A.shape[0]):
        if b[i] > 0:
            g += -A[i]/( 1 + np.exp(np.dot( A[i] , x ) ) ) 
            H +=  (np.exp(np.dot( A[i] , x ))/( 1 + np.exp(np.dot( A[i] , x ) ) )**2)*np.outer(A[i],A[i])
        else:
            g += A[i]/( 1 + np.exp(-np.dot( A[i] , x ) ) ) 
            H +=  (np.exp(-np.dot( A[i] , x ))/( 1 + np.exp(-np.dot( A[i] , x ) ) )**2)*np.outer(A[i],A[i])
    g =  g/m + lam2*x
    H = H/m + lam2*np.eye(n)
    return g,H


def prediction(w, PRINT=False):
    """ A linear prediction function parametrized by the weights w """
    pred = np.zeros(A.shape[0])
    perf = 0
    for i in range(A.shape[0]):
        p = 1.0/(1 + np.exp(-np.dot(A[i], w)))
        if p>0.5:
            pred[i] = 1.0
            if b[i]>0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),1,(p-0.5)*200,correct))
        else:
            pred[i] = -1.0
            if b[i]<0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),-1,100-(0.5-p)*200,correct))
    return pred,float(perf)/A.shape[0]