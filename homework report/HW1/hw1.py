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


def path_generate(S0, r, sigma):
    """
    Find the mean and std of terminal_value

    """
    I = 10000
    M = 500
    T = 1
    dt = T / M

    S = np.zeros((M + 1, I))

    S[0] = S0

    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(I))
    #    plt.figure()
    #    plt.hist(S[-1],bins=50)
    #    plt.figure()
    #    plt.plot(S[:,:10])

    return S

def path_generate2(S0, r, sigma):
    """
    Find the mean and std of terminal_value

    """
    I = 10000
    M = 500
    T = 1
    dt = T / M

    S = np.zeros((M + 1, I))

    S[0] = S0
    z= np.random.standard_normal(I)
    w = np.zeros((M + 1, I))
    w[0] = 0

    for t in range(1, M + 1):

        z = np.random.standard_normal(I)
        w[t] = w[t - 1] + z * (dt) ** (1 / 2)
        S[t] = S[t - 1] +r*S[t-1]*dt+sigma*S[t-1]*(w[t]-w[t-1])
    #    plt.figure()
    #    plt.hist(S[-1],bins=50)
    #    plt.figure()
    #    plt.plot(S[:,:10])

    return S

def port_path_generate(S0, r, sigma,beta,delta):
    """
    Find the mean and std of terminal_value

    """
    I = 10000
    M = 500
    T = 1
    dt = T / M

    S = np.zeros((M + 1, I))

    S[0] = S0
    z= np.random.standard_normal(I)
    w = np.zeros((M + 1, I))
    w[0] = 0

    for t in range(1, M + 1):

        z = np.random.standard_normal(I)
        w[t] = w[t - 1] + z * (dt) ** (1 / 2)
        S[t] = S[t - 1] +r*S[t-1]*dt+sigma*S[t-1]**beta*(w[t]-w[t-1])
    #    plt.figure()
    #    plt.hist(S[-1],bins=50)
    #    plt.figure()
    #    plt.plot(S[:,:10])

    payoff = np.maximum(S[-1]-K, 0)
    average_discounted_payoffs = payoff / (1 + r) ** T
    price=np.mean(average_discounted_payoffs)


    SS=payoff-delta*(S[-1]-S[0])
    plt.figure()
    plt.plot(S[:,:10])
    plt.show()

    payoff_end=np.mean(SS)








    return payoff_end

def payoff(K, S0, r, sigma):
    S = path_generate(S0, r, sigma)
    payoff = np.maximum(S[-1]-K, 0)
    #    print(payoff)

    return payoff

def payoff2(K, S0, r, sigma):
    S = path_generate2(S0, r, sigma)
    payoff = np.maximum(S[-1]-K, 0)
    #    print(payoff)

    return payoff


if __name__ == '__main__':
    T = 1
    S0 = 100
    r = 0
    sigma1 = 0.25
    K = 100
    S_1 = path_generate(S0, r, sigma1)

    plt.figure()
    plt.plot(S_1[:, :10])
    plt.show()

    S_2 = path_generate2(S0, r, sigma1)
    plt.figure()
    plt.plot(S_2[:, :10])
    plt.show()


    #    payoff
    payoff1 = payoff(K, S0, r, sigma1)
    payoff2 = payoff2(K, S0, r, sigma1)

    #    simulation approximation to the price
    average_discounted_payoff = payoff1 / (1 + r) ** T
    average_discounted_payoff2 = payoff2/(1+r)**T
    price1 = np.mean(average_discounted_payoff)
    price2 = np.mean(average_discounted_payoff2)
    print(price1)
    print(price2)

    # B-S
    d1 = (np.log(S0 / K) + sigma1 ** 2 * T / 2) / (sigma1 * np.sqrt(T))
    d2 = (np.log(S0 / K) - sigma1 ** 2 * T / 2) / (sigma1 * np.sqrt(T))

    p = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    print(p)

    #  delta
    d1=1/(sigma1*np.sqrt(T))*(np.log(S0/K)+(r+sigma1**2/2)*T)
    delta=norm.cdf(d1,0,1)
    print(delta)

    # simulation
    a=port_path_generate(S0,r,sigma1,1,delta)
    print(a)
    b=port_path_generate(S0,r,sigma1,0.5,delta)
    print(b)
    c=port_path_generate(S0,r,0.4,1,delta)
    print(c)








