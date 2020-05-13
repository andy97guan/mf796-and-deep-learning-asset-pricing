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
alpha=1.5


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


    return [cc,[kk,CT]]


lk=np.log(250)
[cc,[kk,CT]]=pre_fft_2(N,lk)
K_2=np.arange(60,400,0.1)
llk=np.log(K_2)
[ccc,[kk,CT]]=pre_fft_2(N,llk)

lllk=np.log(150)

[cccc,[kk,CT]]=pre_fft_2(N,lllk)















# KK=np.exp(kk)
#
# K_1=np.arange(0.1,np.log(250),0.01)
#
# cc = np.interp(K_1, kk, y)
#
#
#
#
#
#
#
# black-scholes
T=t

def BL(S0, K, T, r, vol):

    d1 = (np.log(S0 / K) + (r + vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    c = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return c
#
#
#
def dichotomy(K_K,y_y):

    cc=[]
    for i in range(np.size(K_K)):
        c=y_y[i]
        c_est=0
        floor=0
        top=5
        vol=(floor+top)/2


        while abs(c-c_est)>10**(-8):
            print(vol)
            c_est=BL(150,np.exp(K_K[i]),T,r,vol)
            # print(c_est)
            if c-c_est>0:
                floor=vol
                vol=(vol+top)/2
            else:
                top=vol
                vol=(vol+floor)/2

        cc.append(vol)


    return cc

def impliedvol_to_K(n=10,alpha=1,power=7,t=0.25):
    K_list = np.linspace(80,200,100)
    c0_list = []

    [cccc, [kk, CT]] = pre_fft_2(N, lllk)
    # print(cccc)
    strike_list, price_list = np.exp(kk),CT
    for k in K_list:
        c0 = np.interp(k, strike_list, price_list)
        c0_list.append(c0)
    #plt.plot(K_list, c0_list)
    vol_list = []
    for i in np.arange(0,len(K_list),1):
        vol_list.append(fsolve(lambda x: BL(150, K_list[i], 0.25, 0.025, x)-c0_list[i],x0=0.5))
    return np.array(K_list),np.array(vol_list)
#

Strike,Vol_K = impliedvol_to_K()
plt.title("the implied volatility σ(K) as a function of strike")
plt.plot(Strike,Vol_K)
plt.savefig('1_4.jpg')
plt.show()
#
# vol1=dichotomy(llk,ccc)
# plt.plot(K_2,vol1)
# plt.show()

[cccc,[kk,CT]]=pre_fft_2(N,lllk)


# t
def call_price(K=250,n=10,alpha=1,t=0.5):
    t=t
    [cc,[K_list, price_list]]= pre_fft_2(n,lllk)
    return np.interp(K, K_list, price_list)

def impliedvol_to_T(K=150,n=10,alpha=1,power=7,t=0.25):
    t_list = np.arange(0.1,2,0.1)
    c0_list = []
    for ti in t_list:
        c0_list.append(call_price(K=K,n=n,alpha=alpha,t=ti))
        # print(ti)
        # print(call_price(K=K, n=n, alpha=alpha, t=ti))
    #plt.plot(t_list, c0_list)
    vol_list = []
    for i in np.arange(0,len(t_list),1):
        vol_list.append(fsolve(lambda x: BL(150, 150, t_list[i], 0.025, x)-c0_list[i],x0=0.5))
    return np.array(t_list),np.array(vol_list)


tlist, Vol_T = impliedvol_to_T()
plt.title("the implied volatility σ(K) as a function of T")
plt.plot(tlist,Vol_T)
plt.savefig('1_5.jpg')
plt.show()







# sigma = 0.4
# sigma_list = np.arange(-0.3,0.2,0.1) + sigma
# strike_list = []
# vol_list = []
# i = 0
# for si in sigma_list:
#     sigma = si
#     strike,Vol = impliedvol_to_K()
#     strike_list.append(strike)
#     vol_list.append(Vol)
# l = len(vol_list)
# for j in np.arange(0,l,1):
#     plt.plot(strike_list[j], vol_list[j],label='sigma='+ str(round(sigma_list[j],2)) +'')
# plt.legend()
# plt.title('Change sigma')
# plt.savefig('1_6.jpg')
# plt.show()




sigma=0.4
[cccc,[kk,CT]]=pre_fft_2(N,lllk)
print(cccc)

# sigma = 0.4
# v0 = 0.09
# v0_list = np.arange(-0.02,0.03,0.01) + v0
# strike_list = []
# vol_list = []
# i = 0
# for v0i in v0_list:
#     v0 = v0i
#     strike,Vol = impliedvol_to_K()
#     strike_list.append(strike)
#     vol_list.append(Vol)
# l = len(vol_list)
# for j in np.arange(0,l,1):
#     plt.plot(strike_list[j], vol_list[j],label='v0='+ str(round(v0_list[j],2)) +'')
# plt.legend()
# plt.title('Change v0')
# plt.savefig('1_7.jpg')
# plt.show()


# sigma = 0.4
# v0 = 0.09
# k = 0.5
# k_list = np.arange(-0.2,0.3,0.1) + k
# strike_list = []
# vol_list = []
# i = 0
# for ki in k_list:
#     k = ki
#     strike,Vol = impliedvol_to_K()
#     strike_list.append(strike)
#     vol_list.append(Vol)
# l = len(vol_list)
# for j in np.arange(0,l,1):
#     plt.plot(strike_list[j], vol_list[j],label='k='+ str(round(k_list[j],2)) +'')
# plt.legend()
# plt.title('Change $\kappa$')
# plt.savefig('1_8.jpg')
# plt.show()


# sigma = 0.4
# v0 = 0.09
# k = 0.5
# rho = 0.25
# rho_list = np.arange(-0.75,0.75,0.25) + rho
# vol_list = []
# strike_list = []
# i = 0
# for rhoi in rho_list:
#     rho = rhoi
#     strike,Vol = impliedvol_to_K()
#     strike_list.append(strike)
#     vol_list.append(Vol)
# l = len(vol_list)
# for j in np.arange(0,l,1):
#     plt.plot(strike_list[j], vol_list[j],label='rho='+ str(round(rho_list[j],2)) +'')
# plt.legend()
# plt.title('Change rho')
# plt.savefig('1_9.jpg')
# plt.show()

sigma = 0.4
v0 = 0.09
k = 0.5
rho = 0.25
theta = 0.12
theta_list = np.arange(-0.1,0.2,0.1) + theta
t_list = []
vol_list = []
strike_list = []
i = 0
for thetai in theta_list:
    theta = thetai
    strike,Vol = impliedvol_to_K()
    strike_list.append(strike)
    vol_list.append(Vol)
l = len(vol_list)
for j in np.arange(0,l,1):
    plt.plot(strike_list[j], vol_list[j],label='theta='+ str(round(theta_list[j],2)) +'')
plt.legend()
plt.title('Change theta')
plt.savefig('1_10.jpg')
plt.show()









