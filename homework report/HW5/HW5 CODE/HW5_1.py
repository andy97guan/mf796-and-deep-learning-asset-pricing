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

## Implementation of Breeden-Litzenberger:

## (a)
def call_option(K,sigma,T,S0,r):

    d1 = (np.log(S0 / K) + (r+sigma ** 2 / 2)*T) / (sigma* np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    c=S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

    return c


def put_option(K, sigma, T, S0, r):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    p = K * np.exp(-r * T) * norm.cdf(-d2)-S0 * norm.cdf(-d1)

    return p


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


put_par_list = [[0.1,0.3225],[0.25,0.2473],[0.4,0.2021],[0.5,0.1824]]
call_par_list = [[0.4,0.1574],[0.25,0.1370],[0.1,0.1148]]



strike_1m=[]
strike_3m=[]

for i in put_par_list:

    sol1 = fsolve(lambda x: put_delta(x,i[1],1/12,100,0)-i[0], [100])
    strike_1m.append(sol1[0])


for i in call_par_list:

    sol1 = fsolve(lambda x: call_delta(x,i[1],1/12,100,0)-i[0], [100])
    strike_1m.append(sol1[0])



put_par_list = [[0.1,0.2836],[0.25,0.2178],[0.4,0.1818],[0.5,0.1645]]
call_par_list = [[0.4,0.1462],[0.25,0.1256],[0.1,0.1094]]


for i in put_par_list:

    sol1 = fsolve(lambda x: put_delta(x,i[1],3/12,100,0)-i[0], [100])
    strike_3m.append(sol1[0])


for i in call_par_list:

    sol1 = fsolve(lambda x: call_delta(x,i[1],3/12,100,0)-i[0], [100])
    strike_3m.append(sol1[0])

K = {'1M': strike_1m, '3M': strike_3m}
strike = pd.DataFrame(K,index=['10DP','25DP','40DP','50D','40DC','25DC','10DC'])



##(b)
vol_1m=[0.3225,0.2473,0.2021,0.1824,0.1574,0.1370,0.1148]
vol_3m=[0.2836,0.2178,0.1818,0.1645,0.1462,0.1256,0.1094]




X_train=np.array(strike_1m)[:,np.newaxis]
X_test=np.array(strike_1m)[:,np.newaxis]
Y_train=np.array(vol_1m)[:,np.newaxis]
Y_test=np.array(vol_1m)[:,np.newaxis]

model=linear_model.LinearRegression()
model.fit(X_train,Y_train)
y_test_pred=model.predict(X_test)

b_1m=model.intercept_[0]
k_1m=model.coef_[0][0]




X_train=np.array(strike_3m)[:,np.newaxis]
X_test=np.array(strike_3m)[:,np.newaxis]
Y_train=np.array(vol_3m)[:,np.newaxis]
Y_test=np.array(vol_3m)[:,np.newaxis]

model=linear_model.LinearRegression()
model.fit(X_train,Y_train)
y_test_pred=model.predict(X_test)

b_3m=model.intercept_[0]
k_3m=model.coef_[0][0]




##(c)
# vol_1m=[0.3225,0.2473,0.2021,0.1824,0.1574,0.1370,0.1148]
# vol_3m=[0.2836,0.2178,0.1818,0.1645,0.1462,0.1256,0.1094]

K_list=np.arange(80,110,0.01)
vol_1m_new=k_1m*K_list+b_1m
vol_3m_new=k_3m*K_list+b_3m
density_1m=[]
density_3m=[]

for num,i in enumerate(K_list):

    if num<=len(K_list)-3:

        h=K_list[1]-K_list[0]
        K1_1m=call_option(i, vol_1m_new[num], 1 / 12, 100, 0)
        K2_1m=call_option(i+h, vol_1m_new[num+1], 1 / 12, 100, 0)
        K3_1m=call_option(i+2*h, vol_1m_new[num+2], 1 / 12, 100, 0)

        K1_3m=call_option(i, vol_3m_new[num], 3 / 12, 100, 0)
        K2_3m=call_option(i+h, vol_3m_new[num+1], 3 / 12, 100, 0)
        K3_3m=call_option(i+2*h, vol_3m_new[num+2], 3 / 12, 100, 0)



        density_1m.append((K1_1m-2*K2_1m+K3_1m)/h**2)
        density_3m.append((K1_3m-2*K2_3m+K3_3m)/h**2)


plt.plot(K_list[0:-2],density_1m,label='1M')
plt.plot(K_list[0:-2],density_3m,label='3M')
plt.title('risk neutral density')
plt.legend()
plt.savefig('1_1.jpg')
plt.show()


#(d)
K_list=np.arange(60,140,0.01)
vol_1m_2=vol_1m[3]
vol_3m_2=vol_3m[3]

density_1m_2=[]
density_3m_2=[]


for i in K_list:

    h=K_list[1]-K_list[0]
    K1_1m=call_option(i, vol_1m_2, 1 / 12, 100, 0)
    K2_1m=call_option(i+h, vol_1m_2, 1 / 12, 100, 0)
    K3_1m=call_option(i+2*h, vol_1m_2, 1 / 12, 100, 0)

    K1_3m=call_option(i, vol_3m_2, 3 / 12, 100, 0)
    K2_3m=call_option(i+h, vol_3m_2, 3 / 12, 100, 0)
    K3_3m=call_option(i+2*h, vol_3m_2, 3 / 12, 100, 0)



    density_1m_2.append((K1_1m-2*K2_1m+K3_1m)/h**2)
    density_3m_2.append((K1_3m-2*K2_3m+K3_3m)/h**2)

plt.plot(K_list,density_1m_2,label='1M')
plt.plot(K_list,density_3m_2,label='3M')
plt.title('risk neutral density with constant vol')
plt.legend()
plt.savefig('1_2.jpg')
plt.show()


#(e)
##(i)
K_list=np.arange(80,140,0.01)
vol_1m_new=k_1m*K_list+b_1m
vol_3m_new=k_3m*K_list+b_3m
density_1m=[]
density_3m=[]


for num,i in enumerate(K_list):

    if num<=len(K_list)-3:

        h=K_list[1]-K_list[0]
        K1_1m=put_option(i, vol_1m_new[num], 1 / 12, 100, 0)
        K2_1m=put_option(i+h, vol_1m_new[num+1], 1 / 12, 100, 0)
        K3_1m=put_option(i+2*h, vol_1m_new[num+2], 1 / 12, 100, 0)

        K1_3m=put_option(i, vol_3m_new[num], 3 / 12, 100, 0)
        K2_3m=put_option(i+h, vol_3m_new[num+1], 3 / 12, 100, 0)
        K3_3m=put_option(i+2*h, vol_3m_new[num+2], 3 / 12, 100, 0)



        density_1m.append((K1_1m-2*K2_1m+K3_1m)/h**2)
        density_3m.append((K1_3m-2*K2_3m+K3_3m)/h**2)

def digital_put(K,K_list,d_list,w):
    i = 0
    price = 0
    while i < len(d_list):
        if K_list[i] > K:
            price += 0
        else:
            price += 1*d_list[i]*w
        i += 1
    return price

def digital_call(K,K_list,d_list,w):
    i = 0
    price = 0
    while i < len(d_list):
        if K_list[i] < K:
            price += 0
        else:
            price += 1*d_list[i]*w
        i += 1
    return price

print(digital_put(110,K_list,density_1m,h))








##(ii)
K_list=np.arange(90,120,0.01)
vol_1m_new=k_1m*K_list+b_1m
vol_3m_new=k_3m*K_list+b_3m
density_1m=[]
density_3m=[]


for num,i in enumerate(K_list):

    if num<=len(K_list)-3:

        h=K_list[1]-K_list[0]
        K1_1m=call_option(i, vol_1m_new[num], 1 / 12, 100, 0)
        K2_1m=call_option(i+h, vol_1m_new[num+1], 1 / 12, 100, 0)
        K3_1m=call_option(i+2*h, vol_1m_new[num+2], 1 / 12, 100, 0)

        K1_3m=call_option(i, vol_3m_new[num], 3 / 12, 100, 0)
        K2_3m=call_option(i+h, vol_3m_new[num+1], 3 / 12, 100, 0)
        K3_3m=call_option(i+2*h, vol_3m_new[num+2], 3 / 12, 100, 0)



        density_1m.append((K1_1m-2*K2_1m+K3_1m)/h**2)
        density_3m.append((K1_3m-2*K2_3m+K3_3m)/h**2)


print(digital_call(105,K_list,density_3m,h))

##(iii)

def call(K,K_list,d_list,w):
    i = 0
    price = 0
    while i < len(d_list):
        price += np.maximum(K_list[i]-K,0)*d_list[i]*w
        i += 1
    return price

density_2m = [0.5*density_1m[i]+0.5*density_3m[i] for i in range(len(density_1m))]

print(call(100,K_list,density_2m,h/2))
