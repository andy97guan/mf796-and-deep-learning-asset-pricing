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


def normal_fun(x):


    return (x-K1)/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu1)**2/(2*sigma**2))


def Gauss_legendre(f,N,a,b):

    x,weight=np.polynomial.legendre.leggauss(N)
    x=x*(b - a) / 2 + (b + a) / 2

    w=0
    for i in range(N):
        w=w+weight[i]*f(x[i])*(b-a)/2

    return w


# 1
S0=325
sigma1=20
sigma2=15
p=0.95
r=0
T1=1
T2=0.5


K1=370
sigma=sigma1
mu1=S0
mu2=S0
a=K1
b=mu1 + 7 * sigma1

vanilla_option=Gauss_legendre(normal_fun,100,a,b)


print(vanilla_option)

# 2
# p=0.95

K2=365

def binormal_fun(x,y):

    return (x-K1)/(2*np.pi*sigma1*sigma2*np.sqrt(1-p**2))*np.exp(-1/(2*(1-p**2))*((x-mu1)**2/sigma1**2-(2*p*(x-mu1)*(y-mu2))/(sigma1*sigma2)+(y-mu2)**2/sigma2**2))


def Gauss_legendre_2(f,N,ax,bx,ay,by):

    x,weight_x=np.polynomial.legendre.leggauss(N)
    y, weight_y = np.polynomial.legendre.leggauss(N)
    x=x*(bx - ax) / 2 + (bx + ax) / 2
    y=y*(by - ay) / 2 + (by + ay) / 2

    w=0
    ww=0
    for i in range(N):
        w=0
        for j in range(N):
            w=w+weight_y[j]*f(x[i],y[j])*(by-ay)/2

        ww=ww+weight_x[i]*w*(bx-ax)/2

    return ww

ax=K1
bx=mu1 + 7 * sigma1
ay=mu2 - 7 * sigma2
by=K2

print(Gauss_legendre_2(binormal_fun,100,ax,bx,ay,by))

# 3
P=[0.8,0.5,0.2]
value=[]
for i in P:
    p=i

    value.append(Gauss_legendre_2(binormal_fun, 100, ax, bx, ay, by))

print(value)

# 5
p=0.95
KK=[360,350,340]
value2=[]

for i in KK:
    K2=i
    ax = K1
    bx = mu1 + 7 * sigma1
    ay = mu2 - 7 * sigma2
    by = K2

    value2.append(Gauss_legendre_2(binormal_fun, 100, ax, bx, ay, by))

print(value2)


# 7
# p=0.95 , k2 move
p=0.95
diff=[]
for i in np.arange(300,500,1):

    K2=i
    ax = K1
    bx = mu1 + 7 * sigma1
    ay = mu2 - 7 * sigma2
    by = K2
    diff.append(Gauss_legendre_2(binormal_fun, 100, ax, bx, ay, by)-vanilla_option)

plt.figure()
plt.title("when p=0.95 where is K")
plt.plot(np.arange(300,500,1),diff,label='diff')
plt.plot([300, 500], [0, 0], color='red', label='0')
plt.legend()
plt.show()

# p=-0.95, k2 move
p=-0.95
diff=[]
for i in np.arange(300,500,1):

    K2=i
    ax = K1
    bx = mu1 + 7 * sigma1
    ay = mu2 - 7 * sigma2
    by = K2
    diff.append(Gauss_legendre_2(binormal_fun, 100, ax, bx, ay, by)-vanilla_option)

plt.figure()
plt.title("when p=-0.95 where is K")
plt.plot(np.arange(300,500,1),diff,label='diff')
plt.plot([300, 500], [0, 0], color='red', label='0')
plt.legend()
plt.show()

# p=0, k2 move
p=0
diff=[]
for i in np.arange(300,500,1):

    K2=i
    ax = K1
    bx = mu1 + 7 * sigma1
    ay = mu2 - 7 * sigma2
    by = K2
    diff.append(Gauss_legendre_2(binormal_fun, 100, ax, bx, ay, by)-vanilla_option)

plt.figure()
plt.title("when p=0 where is K")
plt.plot(np.arange(300,500,1),diff,label='diff')
plt.plot([300, 500], [0, 0], color='red', label='0')
plt.legend()
plt.show()



















