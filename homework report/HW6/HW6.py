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
import seaborn as sns
sns.set()

#(c)
strike = [280,283,286,289,292,295,298,301,304,307,310,313,316,319,322,325,328,331,334,337,340]
implied_vol = [0.2534,0.294,0.2594,0.2399,0.2715,0.2667,0.2585,0.2356,0.2383,0.2301,0.2261,0.2472,0.1926,0.2284,0.2207,0.2188,0.2246,0.1919,0.2237,0.2251,0.1799]

sigma = np.interp(317.5, strike, implied_vol)
sigma_1 = np.interp(315, strike, implied_vol)
sigma_2 = np.interp(320, strike, implied_vol)
sigma_new=0.5*(sigma_1+sigma_2)

#(e)


def Euler_discretization(sigma,r,Smin,Smax,T,N,M,K1,K2,S0,early_exercise):


    ht = T/N
    hs = Smax/M
    Si = np.arange(Smin,Smax+hs,hs) # j = 0,1,...,M-1


    ai = 1-(sigma**2)*(Si**2)*(ht/(hs**2)) - r*ht
    li = ((sigma**2)*(Si**2)/2)*(ht/(hs**2)) - r*Si*ht/2/hs
    ui = ((sigma**2)*(Si**2)/2)*(ht/(hs**2)) + r*Si*ht/2/hs

    ai = ai[1:M] # ai, i = 1,...,M-1
    li = li[2:M] # li, i = 2,...,M-1
    ui = ui[1:M-1] # ui, i = 1,...,M-2

    A = np.diag(ai)+np.diag(li,k=-1)+np.diag(ui,k=1)

    eig_vals, eig_vecs = np.linalg.eig(A)
    plt.plot(abs(eig_vals))
    plt.savefig("1_3.jpg")
    plt.show()
    plt.plot(np.sort(abs(eig_vals))[::-1])
    plt.savefig("1_4.jpg")
    plt.show()

    Ct1=np.maximum(Si-K1,0)
    Ct2=np.minimum(K2-Si,0)
    Ct=Ct1+Ct2  # get the price of call spread
    Ctt=Ct[1:M]



    for i in range(N):
        bj = ui[-1] * (K2 - K1) * (np.exp(-r * i * ht))
        Ctt = A.dot(Ctt)
        Ctt[-1] = Ctt[-1] + bj
        if early_exercise == True:
            Ctt = [max(x,y) for x,y in zip(Ctt,Ct[1:M])]

    C0 = np.interp(S0, Si[1:M], Ctt)





    return C0



r=0.0072
Smin = 0
Smax = 500
T = 7/12
N = 3000
M = 300
K1=315
K2=320
S0=312.8599853515625


C0=Euler_discretization(sigma_new,r,Smin,Smax,T,N,M,K1,K2,S0,early_exercise=True)
C1=Euler_discretization(sigma_new,r,Smin,Smax,T,N,M,K1,K2,S0,early_exercise=False)
print(C0-C1)






















