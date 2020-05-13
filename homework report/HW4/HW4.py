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



class stock_basic:
    """
    every stock's data
    """

    def __init__(self, name, data):
        self.name = name  # name is code
        self.data = data  # data

    def annualized_return(self):
        annual_return = ((self.data.iloc[-1] - self.data.iloc[0]) / self.data.ix[0]) ** (365 / len(self.data))

        return annual_return

    # regression
    def regression(x, y):
        model = linear_model.LinearRegression()
        model.fit(x, y)

        return model.coef_

def e_val_percentage(e_val,percentage):
    a=0
    b=0
    e_val_sum=sum(e_val)
    for i in e_val:
        a=a+i
        b=b+1
        p=a/e_val_sum
        if p>percentage:
            return b








if __name__ == '__main__':
    code=pd.read_excel('ETF Symbol.xlsx')
    icode=np.array(code['Symbol']).tolist()
    data = pdr.get_data_yahoo(icode, start='2015-01-01', end='2020-01-01')  # For data
    data_close = data['Close']
    data_close.to_csv("data_close.csv")


    daily_returns = np.log(data_close.pct_change().shift(-1)+1)
    daily_returns=daily_returns.dropna(axis=0, how='all')
    Cov_daily = daily_returns.cov()
    e_vals, e_vecs = np.linalg.eig(Cov_daily)
    plt.plot(e_vals)
    plt.show()

    num=e_val_percentage(e_vals,0.9)

    u=e_vecs[0:num]
    HU=np.dot(daily_returns,u.T)



    # pca=PCA(n_components=1)
    # pca.fit(Cov_daily)
    # x=pca.components_

    model = linear_model.LinearRegression()
    model.fit(HU, np.array(daily_returns))
    model.intercept_ # 截距
    model.coef_  # 线性模型的系数

    residual=np.array(daily_returns)-np.dot(HU,model.coef_.T)-model.intercept_

    plt.plot(residual)
    plt.show()
    # a = model.predict([[12]])




























#(b)



    #annual_return2
    annual_return_value2 = daily_returns.mean()*252
    R=annual_return_value2
    a=1 #risk aversion


    g1=np.zeros(120)+1
    g2=np.zeros(120)

    for i in range(120):

        if i<17:
            g2[i]=1



    G = [g1, g2]
    G=np.array(G)
    c=[1,0.1]
    c=np.array(c)
    C=Cov_daily.values*252

    GCGT=np.dot(G,np.dot(np.linalg.inv(C),G.T))
    B=np.dot(G,np.dot(np.linalg.inv(C),R))-2*a*c

    lambda1=np.dot(np.linalg.inv(GCGT),B)

    BB=R-np.dot(G.T,lambda1)
    weight1=1/(2*a)*np.dot(np.linalg.inv(C),BB)

































