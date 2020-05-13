import numpy as np
import pandas as pd
import matplotlib as mql
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from numpy.random import standard_normal as stn
from datetime import datetime
from sklearn import linear_model
import statsmodels.api as sm
import scipy.stats as scs
from scipy.stats import norm
import math
from cvxopt import matrix, solvers
import scipy.optimize as sco
from sympy import *
import warnings
warnings.filterwarnings("ignore")




## Problem 1
T=1/4
K=12
vol=0.20
S0=10
r=0.04

# black-scholes
d1 = (np.log(S0 / K) + (r+vol ** 2 / 2)*T) / (vol* np.sqrt(T))
d2 = (np.log(S0 / K) + (r-vol ** 2 / 2)*T) / (vol * np.sqrt(T))
c=S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

print('The price is {}'.format(c))

# left
def BL(x):


    return (x*S0-K)/(x*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))

def Riemamn_left(f,N,a,b):

    x=np.linspace(a,b,N+1)
    d=(b-a)/N
    w=0
    for i in x:
        if i!=b:
            w=w+f(i)*d

    return w

mu=(r - 0.5 * vol ** 2) * T
sigma=vol*np.sqrt(T)
a=K/S0
b=1+mu+7*sigma


N=[5,10,50,100]
diff_list=[]
for i in N:
    diff=abs(np.exp(-r * T)*Riemamn_left(BL,i,a,b)-c)
    diff_list.append(diff)

print(diff_list)

# paint
NN1=np.arange(5,100,5)
diff_list_left1=[]
benchmark1=[]
for i in NN1:
    diff=abs(np.exp(-r * T)*Riemamn_left(BL,i,a,b)-c)
    diff_list_left1.append(diff)
    benchmark1.append(1/i**(2)/5)


plt.title('Convergence for left Riemann rule')
plt.plot(NN1, diff_list_left1,color='b', label='Quadrature error')
plt.plot(NN1,benchmark1,color='r',label='O(N^-2)')
plt.legend()
plt.show()

# midpoint
def Riemamn_mid(f,N,a,b):

    x=np.linspace(a,b,N+1)
    d=(b-a)/N
    w=0
    for i in x:
        if i!=b:
            w=w+f(i+d/2)*d

    return w

mu=(r - 0.5 * vol ** 2) * T
sigma=vol*np.sqrt(T)
a=K/S0
b=1+mu+7*sigma


N=[5,10,50,100]
diff_list=[]
for i in N:
    diff=abs(np.exp(-r * T)*Riemamn_mid(BL,i,a,b)-c)
    diff_list.append(diff)

print(diff_list)

# paint
NN2=np.arange(5,100,5)
diff_list_left2=[]
benchmark2=[]
for i in NN2:
    diff=abs(np.exp(-r * T)*Riemamn_mid(BL,i,a,b)-c)
    diff_list_left2.append(diff)
    benchmark2.append(1 / i ** (2)/10 )

plt.figure()
plt.title('Convergence for mid Riemann rule')
plt.plot(NN2, diff_list_left2,color='b', label='Quadrature error')
plt.plot(NN1,benchmark2,color='r',label='O(N^-2)')
plt.legend()
plt.show()


# Gauss
# points, weights = np.polynomial.legendre.leggauss(N)

def Gauss_legendre(f,N,a,b):

    x,weight=np.polynomial.legendre.leggauss(N)
    x=x*(b - a) / 2 + (b + a) / 2

    w=0
    for i in range(N):
        w=w+weight[i]*f(x[i])*(b-a)/2

    return w

N=[5,10,50,100]
diff_list=[]
benchmark3=[]
for i in N:
    diff=abs(np.exp(-r * T)*Gauss_legendre(BL,i,a,b)-c)
    diff_list.append(diff)

print(diff_list)

# paint
NN3=np.arange(5,100,5)
diff_list_left3=[]
for i in NN3:
    diff=abs(np.exp(-r * T)*Gauss_legendre(BL,i,a,b)-c)
    diff_list_left3.append(diff)
    benchmark3.append(1 / i ** (i)/7)

plt.figure()
plt.title('Convergence for Gauss_legendre rule')
plt.plot(NN3, diff_list_left3,color='b', label='Quadrature error')
plt.plot(NN3,benchmark3,color='r',label='O(N^-N)')
plt.legend()
plt.show()


# paint
plt.figure()
plt.title("Comparing")
plt.plot(NN1, diff_list_left1,color='b', label='left Riemann rule error')
plt.plot(NN2, diff_list_left2,color='red', label='Mid Riemann rule error')
plt.plot(NN3, diff_list_left3,color='green', label='Gauss error')
plt.legend()
plt.show()

























