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
from sklearn.decomposition import PCA
from scipy import stats
from scipy.optimize import fsolve
from scipy.optimize import minimize
from pyfinance import ols



def w(u,sigma,v0,k,rho,theta,t,K,S0):

    lambda1=np.sqrt(sigma**2*(u**2+j*u)+(k-j*rho*sigma*u)**2)

    A=np.exp(j*u*np.log(S0)+j*u*(r-q)*t+(k*theta*t*(k-j*rho*sigma*u))/(sigma**2))
    B=(np.cosh(lambda1*t/2)+(k-j*rho*sigma*u)/(lambda1)*np.sinh((lambda1*t/2)))**((2*k*theta)/(sigma**2))

    return A/B


def phi(u,sigma,v0,k,rho,theta,t,K,S0):

    lambda1 = np.sqrt(sigma ** 2 * (u ** 2 + j * u) + (k - j * rho * sigma * u) ** 2)
    C=np.exp(-((u**2+j*u)*v0)/(lambda1*np.cosh(lambda1*t/2)/(np.sinh(lambda1*t/2))+k-j*rho*sigma*u))

    return w(u,sigma,v0,k,rho,theta,t,K,S0)*C

def kesai(v,sigma,v0,k,rho,theta,t,K,S0):

    D=(np.exp(-r*t))/((alpha+j*v)*(alpha+j*v+1))

    return D*phi(v-(alpha+1)*j,sigma,v0,k,rho,theta,t,K,S0)

def f(v,sigma,v0,k,rho,theta,t,K,S0):

    return np.exp(-j*v*np.log(K))*kesai(v,sigma,v0,k,rho,theta,t,K,S0)

def increasing(list):

    return all(x >= y for x, y in zip(list, list[1:])) or all(x <= y for x, y in zip(list, list[1:]))
#(A)



# i

j=complex(0,1)


def pre_fft(N,K_1,sigma,v0,k,rho,theta,alpha,t,r,q,S0):
    b = 1000

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
        b=np.exp(-j*(np.log(S0)-(dk*N/2))*vv[i-1])*phi(vv[i-1]-(alpha+1)*j,sigma,v0,k,rho,theta,t,K,S0)

        x.append(a*b)

    y = np.fft.fft(x)
    y_real = y.real
    CT = []
    for i in range(1, N + 1):
        aa = np.exp(-alpha * (np.log(S0) - dk * (N / 2 - (i - 1)))) / np.pi * y_real[i - 1]
        CT.append(aa)

    cc = np.interp(K_1, kk, CT)


    return [cc,kk]


k = 3.51
theta = 0.052
sigma = 1.17
rho = -0.77
v0 = 0.034
S0 = 267.15
r = 0.015
q = 0.0177
T = 0.25
K = 275
alpha = 1
N = 2**15


[y1,kk]=pre_fft(N,np.log(K),sigma,v0,k,rho,theta,alpha,T,r,q,S0)
print(y1)


[cplus,kk] = pre_fft(N,np.log(K),sigma,v0,k,rho,theta,alpha,T,r,q,S0+2)
print(cplus)

[cminus,kk] = pre_fft(N,np.log(K),sigma,v0,k,rho,theta,alpha,T,r,q,S0-2)
print(cminus)


delta = (cplus-cminus)/4
print(delta)

def bsm_call_value(s0, k, t, r, sigma):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    value = (s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))

    return value

def call_delta(K, sigma, T, S0, r):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    c = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return norm.cdf(d1)

def put_delta(K, sigma, T, S0, r):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    p = K * np.exp(-r * T) * norm.cdf(-d2)-S0 * norm.cdf(-d1)

    return 1-norm.cdf(d1)

implied_vol = fsolve(lambda x: bsm_call_value(S0,K , T, r, x)-y1,x0=0.5)[0]

bsm_delta =call_delta(K, implied_vol, T, S0, r)

print(bsm_delta)









#(B)


[cplus1,kk] = pre_fft(N,np.log(K),sigma,v0+0.01,k,rho,theta+0.01,alpha,T,r,q,S0)
[cminus1,kk] = pre_fft(N,np.log(K),sigma,v0-0.01,k,rho,theta-0.01,alpha,T,r,q,S0)

print(cplus1,cminus1)

Vega = (cplus1-cminus1)/0.02
print(Vega)


def call_vega(K,sigma,T,S0=100,r=0):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) \
             / (sigma * np.sqrt(T))
    return S0*norm.pdf(d1)*np.sqrt(T)



bsm_vega = call_vega(K,implied_vol,T,S0,r)
print(bsm_vega)

















