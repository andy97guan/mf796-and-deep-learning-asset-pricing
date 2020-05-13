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
from scipy.optimize import fsolve


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









r=0.025
q=0
S0=150


# (b)


sigma=0.4
v0=0.09
k=0.5
rho=0.25
theta=0.12

# i

j=complex(0,1)
t=1/4
alpha=1


N=2**10


def pre_fft_2(N,K_1):
    b = S0*(1+7*sigma)


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

    return cc


lk=np.log(150)
cc=pre_fft_2(N,lk)
K_2=np.arange(100,400,0.1)
llk=np.log(K_2)
K=150
ccc=pre_fft_2(N,K)

c_t=[]
tt=[]
it=range(4,100)
iv0=np.arange(0.01,2,0.01)
ikappa=np.arange(4.925,4.950,0.001)
irho=np.arange(-1,1,0.01)
itheta=np.arange(0.01,1,0.01)

for i in itheta:
    theta=i
    ctc = pre_fft_2(N, K)
    c_t.append(ctc)




T=1/4
def BL(vol):

    d1 = (np.log(S0 / K) + (r + vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    c = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return c
#
#
#


def dichotomy(T_T,y_y):

    cc=[]
    for i in range(np.size(T_T)):
        c=y_y[i]
        c_est=0
        floor=0
        top=10
        vol=(floor+top)/2


        while abs(c-c_est)>10**(-8):
            # print(vol)
            c_est=BL(vol)
            # print(c_est)
            if c-c_est>0:
                floor=vol
                vol=(vol+top)/2
            else:
                top=vol
                vol=(vol+floor)/2

        cc.append(vol)


    return cc
#
vol1=dichotomy(itheta,c_t)
plt.title('theta to implied vol')
plt.plot(itheta,vol1)
plt.show()

