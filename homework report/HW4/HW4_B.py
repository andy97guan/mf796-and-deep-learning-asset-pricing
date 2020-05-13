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
import json
from cvxopt import matrix, solvers



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
    sp500 = pdr.get_data_yahoo('^GSPC', start='2015-02-12', end='2020-02-12')
    data = sp500.reset_index().loc[:, ['Date']]
    with open("sp500-historical-components.json", 'r') as load_f:
        load_dict = json.load(load_f)
    stk_list = load_dict[0]['Symbols']
    stk_list = stk_list[0:110]
    for ticker in stk_list:
        try:
            stk = pdr.get_data_yahoo(ticker, start='2015-02-12', end='2020-02-12')
        except:
            # print('no ticker: ', ticker)
            stk_list.remove(ticker)
        else:
            if stk.empty != True:
                stk = stk.reset_index().loc[:, ['Date', 'Close']]
                stk.rename(columns={'Close': ticker + '_Close'}, inplace=True)
                data = pd.DataFrame.merge(data, stk, how='left', on='Date')
    data.dropna(axis=1, how='all', inplace=True)
    values = dict([(col_name, col_mean) for col_name, col_mean in zip(data.columns.tolist(), data.mean().tolist())])
    data.fillna(value=values, inplace=True)



    tickers=data.columns[1:]
    daily_returns = np.log(data[tickers].pct_change().shift(-1)+1)
    daily_returns=daily_returns.dropna(axis=0, how='all')
    Cov_daily = daily_returns.cov()
    e_vals, e_vecs = np.linalg.eig(Cov_daily)
    plt.title('eigenvalue')
    plt.plot(e_vals)
    plt.savefig('1_2.jpg')
    plt.show()

    num=e_val_percentage(e_vals,0.9)

    u=e_vecs[0:num]
    HU=np.dot(daily_returns,u.T)


    model = linear_model.LinearRegression()
    model.fit(HU, np.array(daily_returns))
    model.intercept_ # 截距
    model.coef_  # 线性模型的系数

    residual=np.array(daily_returns)-np.dot(HU,model.coef_.T)-model.intercept_
    plt.title('residual')
    plt.plot(residual)
    plt.savefig('1_3.jpg')
    plt.show()


#B
    annual_return_value2 = daily_returns.mean()*252
    R=annual_return_value2
    a=1 #risk aversion

    g1=np.zeros(106)+1
    g2=np.zeros(106)

    for i in range(106):

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
