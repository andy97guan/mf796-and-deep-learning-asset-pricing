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

data = pd.read_excel('mf796-hw5-opt-data.xlsx')

S0=267.15
K=250
lk=np.log(K)
sigma=0.2
v0=0.08
k=0.7
rho=-0.4
theta=0.1
r=0.015
q=0.0177


# i

j=complex(0,1)
t=1/2
alpha=1


def w(u,sigma,v0,k,rho,theta):

    lambda1=np.sqrt(sigma**2*(u**2+j*u)+(k-j*rho*sigma*u)**2)

    A=np.exp(j*u*np.log(S0)+j*u*(r-q)*t+(k*theta*t*(k-j*rho*sigma*u))/(sigma**2))
    B=(np.cosh(lambda1*t/2)+(k-j*rho*sigma*u)/(lambda1)*np.sinh((lambda1*t/2)))**((2*k*theta)/(sigma**2))

    return A/B


def phi(u,sigma,v0,k,rho,theta):

    lambda1 = np.sqrt(sigma ** 2 * (u ** 2 + j * u) + (k - j * rho * sigma * u) ** 2)
    C=np.exp(-((u**2+j*u)*v0)/(lambda1*np.cosh(lambda1*t/2)/(np.sinh(lambda1*t/2))+k-j*rho*sigma*u))

    return w(u,sigma,v0,k,rho,theta)*C

def kesai(v,sigma,v0,k,rho,theta):

    D=(np.exp(-r*t))/((alpha+j*v)*(alpha+j*v+1))

    return D*phi(v-(alpha+1)*j,sigma,v0,k,rho,theta)

def f(v,sigma,v0,k,rho,theta):

    return np.exp(-j*v*lk)*kesai(v,sigma,v0,k,rho,theta)

def increasing(list):

    return all(x >= y for x, y in zip(list, list[1:])) or all(x <= y for x, y in zip(list, list[1:]))



#(a)

print(data.groupby(['expT']).call_bid.apply(increasing))
print(data.groupby(['expT']).call_ask.apply(increasing))
print(data.groupby(['expT']).put_bid.apply(increasing))
print(data.groupby(['expT']).put_ask.apply(increasing))


#(b)

S0=267.15
K=250
lk=np.log(K)
sigma=0.2
v0=0.08
k=0.7
rho=-0.4
theta=0.1
r=0.015
q=0.0177


# i

j=complex(0,1)
t=1/2
alpha=1

def pre_fft(N,K_1,sigma,v0,k,rho,theta,alpha,t):
    b = 600

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
        b=np.exp(-j*(np.log(S0)-(dk*N/2))*vv[i-1])*phi(vv[i-1]-(alpha+1)*j,sigma,v0,k,rho,theta)

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
[y,kk]=pre_fft(N,lk,sigma,v0,k,rho,theta,alpha,t)
print(y)




def optimizer(sigma,v0,k,rho,theta,dataset):

    i = 0
    sse = 0
    while i< dataset.shape[0]:
        T = dataset.expT[i]
        strike = dataset.K[i]
        lk=np.log(strike)
        [y, kk] = pre_fft(N, lk, sigma,v0,k,rho,theta, alpha, T)
        c1=y
        c = dataset.call_mid[i]
        sse = (c1 - c)**2
        i = i + 1
    return sse


data['call_mid']  = (data['call_bid'] + data['call_ask'])/2

x0 = [2,0.2,0.5,-1,0.1]
bnds = ((0.01, 5), (0, 2),(0,1),(-1,1),(0,1))
args1 = minimize(lambda p: optimizer(p[0],p[1],p[2],p[3],p[4],data),x0,method='SLSQP',bounds=bnds)
args1



x0 = [2,0.2,0.5,-1,0.1]
bnds = ((0.01, 2.5), (0, 1),(0,1),(-1,0.5),(0,0.5))
args2 = minimize(lambda p: optimizer(p[0],p[1],p[2],p[3],p[4],data),x0,method='SLSQP',bounds=bnds)
args2

def weighted_optimizer(sigma,v0,k,rho,theta,dataset):
    i = 0
    sse = 0
    while i< dataset.shape[0]:
        T = dataset.expT[i]
        strike = dataset.K[i]
        w = 1/(dataset.call_ask[i] - dataset.call_bid[i])
        [y, kk] = pre_fft(N, lk, sigma, v0, k, rho, theta, alpha, T)
        c1 = y
        c = dataset.call_mid[i]
        sse = w*(c1 - c)**2
        i = i + 1
    return sse

x0 = [2,0.2,0.5,-1,0.1]
bnds = ((0.01, 2.5), (0, 1),(0,1),(-1,0.5),(0,0.5))
args3 = minimize(lambda p: weighted_optimizer(p[0],p[1],p[2],p[3],p[4],data),x0,method='SLSQP',bounds=bnds)
args3











