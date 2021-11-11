#python implementation of autoregression AR for time series prediction, just using OLS
import numpy as np
import numpy.linalg as la
from .utils import ese
from sklearn.model_selection import train_test_split
from sklearn import linear_model
"""
takes in time series of dimension p and length n. as n x p matrix.
outputs outlier score, from prediction error.
see: H. Lutkepohl, 'New introduction to multiple time series analysis. Springer-Verlag Berlin Heidelberg, 2005.
"""


def split(w, X):
    """
    p is the number of previous examples we will use to predict the next
    makes the target array Y
    makes the vectorised data array Z
    """
    T = X.shape[0]-w #number of data points we have enough data for
    Y = []
    Z = []
    for t in range(T):
        Y.append(X[t+w,:])
        Z.append(np.concatenate([[1],np.hstack([X[x].T for x in range(t+w-1,t-1,-1)] ) ]))
    return np.array(Y).T, np.array(Z).T

def get_estimate(w, X):
    """
    separates out the data matrix X in to Y and Z using the split function
    estimates each value using the previous values using the normal equation.
    """
    T = X.shape[0]-w #number of data points we have enough data for
    Y,Z = split(w,X)
    B_hat = Y.dot(Z.T).dot(la.inv(Z.dot(Z.T)))
    return Y, B_hat.dot(Z)


def get_VAR_os(X, w=3):
    """
    use VAR to make estimate then calculate elementwise squared error for each.
    can't do initial p data points as not enough data. they are set to the same
    as the first value.
    """
    Y, Y_hat = get_estimate(w,X)
    errs = ese(Y_hat, Y)
    # print(type(errs), errs.shape)
    errs = np.sum(errs, axis=0)
    errs = np.concatenate([np.ones(w)*errs[0], errs], axis=None)
    return errs


def OLS_err(X_train, y_train, X, y):
    """
    takes in train test split returns elementwise error for whole dataset.
    """
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    errs = reg.predict(X)
    # print(type(errs), errs.shape)
    return ese(errs, y)

def ridge_err(X_train, y_train, X, y, alpha):
    """
    takes in train test split returns elementwise error for whole dataset.
    """
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    pred = reg.predict(X)
    return ese(pred, y)

def lasso_err(X_train, y_train, X, y, alpha):
    """
    takes in train test split returns elementwise error for whole dataset.
    """
    reg = linear_model.Lasso(alpha=alpha)
    reg.fit(X_train, y_train)
    pred = reg.predict(X)
    return ese(pred, y)

def get_OLS_os(X):
    """
    finds OLS error on prediciton of each feature using the other.
    """
    n,p = X.shape
    err_sum = np.zeros(n)

    for i in range(p):
        inds = np.arange(p)
        X_x = np.delete(X, i, axis=1)
        y_y = X[:,i]
        X_train, X_test, y_train, y_test = train_test_split(X_x, y_y)
        err = OLS_err(X_train, y_train, X_x, y_y)
        err_sum +=err
    return err_sum/n

def get_ridge_os(X, alpha=1):
    """
    finds OLS with L2 regularisation error on prediction of each feature using the other.
    """
    n,p = X.shape
    err_sum = np.zeros(n)

    for i in range(p):
        inds = np.arange(p)
        X_x = np.delete(X, i, axis=1)
        y_y = X[:,i]
        X_train, X_test, y_train, y_test = train_test_split(X_x, y_y)
        err = ridge_err(X_train, y_train, X_x, y_y, alpha)
        err_sum +=err
    return err_sum/n

def get_LASSO_os(X, alpha=1):
    """
    finds OLS with L1 regularisation error on prediction of each feature using the other.
    """
    n,p = X.shape
    err_sum = np.zeros(n)

    for i in range(p):
        inds = np.arange(p)
        X_x = np.delete(X, i, axis=1)
        y_y = X[:,i]
        X_train, X_test, y_train, y_test = train_test_split(X_x, y_y)
        err = lasso_err(X_train, y_train, X_x, y_y, alpha)
        err_sum +=err
    return err_sum/n

if __name__ == '__main__':
    #for testing.
    from test_data import generate_test
    import matplotlib.pyplot as plt
    from utils import auc




    n = 1000
    p = 2
    p_lst = [2,4,8,16,32,64]
    gamma = 0.05
    p_frac = 0.3
    p_quant = 0.3
    r = 20
    aucs=[]
    params=[]
    #
    n = 100
    p = 32
    # X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
    # os = get_OLS_os(X)
    # print(os)
    # print(outs)


    for p in p_lst:
        X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
        os = get_VAR_os(X, *params)
        # print(os[:10])
        aucs.append(auc(os, outs))
    for i in range(len(p_lst)):
        print(p_lst[i], aucs[i])
    aucs=[]

    for p in p_lst:
        X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
        os = get_OLS_os(X, *params)
        # print(os[:10])
        aucs.append(auc(os, outs))
    for i in range(len(p_lst)):
        print(p_lst[i], aucs[i])

    for p in p_lst:
        X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
        os = get_ridge_os(X, *params)
        # print(os[:10])
        aucs.append(auc(os, outs))
    for i in range(len(p_lst)):
        print(p_lst[i], aucs[i])

    for p in p_lst:
        X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
        os = get_LASSO_os(X, *params)
        # print(os[:10])
        aucs.append(auc(os, outs))
    for i in range(len(p_lst)):
        print(p_lst[i], aucs[i])

    X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
    print(outs)

    os = get_VAR_os(X, *params)
    # print(os[:10])
    print(auc(os, outs))
    tim = np.arange(len(os))
    fig = plt.figure()
    plt.plot(tim, os)
    for out in outs:
        fig.text(out+1,2.5, '-', color='r')
        plt.plot(out, os[out], 'ro')
        # print(out)
    plt.show()
    X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
    print(outs)

    os = get_OLS_os(X, *params)
    # print(os[:10])
    print(auc(os, outs))
    tim = np.arange(len(os))
    plt.figure()
    plt.plot(tim, os)
    for out in outs:
        fig.text(out+1,2.5, '-', color='r')
        plt.plot(out, os[out], 'ro')
        # print(out)
    plt.show()
    X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
    print(outs)

    os = get_LASSO_os(X, *params)
    print(os[:10])
    print(auc(os, outs))
    tim = np.arange(len(os))
    fig = plt.figure()
    plt.plot(tim, os)
    for out in outs:
        fig.text(out+1,2.5, '-', color='r')
        plt.plot(out, os[out], 'ro')
        # print(out)
    plt.show()
    X, outs = generate_test(n,p,r, p_frac, p_quant,gamma)
    print(outs)
    os = get_ridge_os(X, *params)
    print(os[:10])
    print(auc(os, outs))
    tim = np.arange(len(os))
    fig = plt.figure()
    plt.plot(tim, os)
    for out in outs:
        fig.text(out+1,2.5, '-', color='r')
        plt.plot(out, os[out], 'ro')
        # print(out)
    plt.show()
