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
from scipy.fftpack import fft,ifft
from sympy import *

# function

def Riemamn_left(f,N,a,b):

    x=np.linspace(a,b,N+1)
    d=(b-a)/N
    w=0
    for i in x:
        if i!=b:
            w=w+f(i)*d

    return w

def Riemamn_mid(f,N,a,b):

    x=np.linspace(a,b,N+1)
    d=(b-a)/N
    w=0
    for i in x:
        if i!=b:
            w=w+f(i+d/2)*d

    return w


def Gauss_legendre(f,N,a,b):

    x,weight=np.polynomial.legendre.leggauss(N)
    x=x*(b - a) / 2 + (b + a) / 2

    w=0
    for i in range(N):
        w=w+weight[i]*f(x[i])*(b-a)/2

    return w



r=0.02
q=0
S0=250


# (a)

K=250
lk=np.log(K)
sigma=0.2
v0=0.08
k=0.7
rho=-0.4
theta=0.1

# i

j=complex(0,1)
t=1/2
alpha=1


def w(u):

    lambda1=np.sqrt(sigma**2*(u**2+j*u)+(k-j*rho*sigma*u)**2)

    A=np.exp(j*u*np.log(S0)+j*u*(r-q)*t+(k*theta*t*(k-j*rho*sigma*u))/(sigma**2))
    B=(np.cosh(lambda1*t/2)+(k-j*rho*sigma*u)/(lambda1)*np.sinh((lambda1*t/2)))**((2*k*theta)/(sigma**2))

    return A/B


def phi(u):

    lambda1 = np.sqrt(sigma ** 2 * (u ** 2 + j * u) + (k - j * rho * sigma * u) ** 2)
    C=np.exp(-((u**2+j*u)*v0)/(lambda1*np.cosh(lambda1*t/2)/(np.sinh(lambda1*t/2))+k-j*rho*sigma*u))

    return w(u)*C

def kesai(v):

    D=(np.exp(-r*t))/((alpha+j*v)*(alpha+j*v+1))

    return D*phi(v-(alpha+1)*j)

def f(v):

    return np.exp(-j*v*lk)*kesai(v)


# original
b = S0+S0*7*sigma
N=2**10

R1=Riemamn_left(f,N,0,b)
R2=Riemamn_mid(f,N,0,b)
R3=Gauss_legendre(f,N,0,b)
CT1=np.exp(-alpha*lk)/np.pi*R1
CT2=np.exp(-alpha*lk)/np.pi*R2
CT3=np.exp(-alpha*lk)/np.pi*R3
print(CT1)
print(CT2)
print(CT3)












# FFT



def pre_fft(N,K_1,alpha):

    b = S0+S0*7*sigma
    N=2**10

    dv = b / N
    dk = 2 * np.pi / b

    beta = np.log(S0) -  dk * N / 2
    vv = np.linspace(0, b, N+1)[0:-1]
    kk = np.linspace(0,N,N+1)[0:-1]*dk+beta

    x=[]
    for i in range(1,N+1):

        if i==1:
            delta=1
        else:
            delta=0

        a=((2-delta)*dv*np.exp(-r*t))/(2*(alpha+j*vv[i-1])*(alpha+j*vv[i-1]+1))
        b=np.exp(-j*(np.log(S0)-(dk*N/2))*vv[i-1])*phi(vv[i-1]-(alpha+1)*j)

        x.append(a*b)

    y = np.fft.fft(x)
    y_real = y.real
    CT = []
    for i in range(1, N + 1):
        aa = np.exp(-alpha * (np.log(S0) - dk * (N / 2 - (i - 1)))) / np.pi * y_real[i - 1]
        CT.append(aa)

    cc = np.interp(K_1, kk, CT)


    return [cc,kk]

N=2**10
[y,kk]=pre_fft(N,lk,1)
print(y)


aa=np.arange(0.1,3,0.05)
bb=[]
factor=1
for i in aa:
    N=2**10
    alpha=i
    [y,kk]=pre_fft(N,lk,alpha)
    bb.append(y)


    if np.size(bb)>=2:
        if np.abs(bb[-1]-bb[-2])<=10**(-3) and factor==1:

            alpha_stable=i
            factor=0

print(alpha_stable)








plt.title('Convergence for alpha')
plt.plot(aa,bb)
plt.savefig('1_1.jpg')
plt.show()






NN=[]
YY=[]
for i in range(15):
    N=2**i
    NN.append(N)

    [y,kk]=pre_fft(N,lk,1)
    YY.append(y)

plt.title('Convergence for alpha')
plt.plot(range(15),YY)
plt.savefig('1_2.jpg')
plt.show()




lk=np.log(260)
NN=[]
YY=[]
for i in range(15):
    N=2**i
    NN.append(N)

    [y,kk]=pre_fft(N,lk,1)
    YY.append(y)

plt.title('Convergence for N')
plt.plot(range(15),YY)
plt.savefig('1_3.jpg')
plt.show()






















