"""
all the functions take the Distribution object as input.
"""
import math
import numpy as np
from scipy import stats

def mean(rv):
    '''
    mean of linear data sets

    :param: Distribution object
    :return: float
    '''
    data = rv.get_data()
    return sum(data) / len(data)

def median(rv):
    '''
    median of linear data sets

    :param: Distribution object
    :return: float
    '''
    data = rv.get_data()
    return np.median(data)

def mode(rv):
    '''
    mode of linear data sets

    :param: Distribution object
    :return: float
    '''
    data = rv.get_data()
    return stats.mode(data)

def stdDv(rv):
    '''
    standard deviation of linear data sets

    :param: Distribution object
    :return: float
    '''
    data = rv.get_data()
    return np.nanstd(data)

def variance(rv):
    '''
    variance of linear data sets

    :param: Distribution object
    :return: float
    '''
    data = rv.get_data()
    return np.nanvar(data)

def covariance_2d(X, Y):
    '''
    Covariance of two different random variables

    :param: two distribution objects
    :return: float
    '''
    data1 = X.get_data()
    data2 = Y.get_data()

    return np.cov(data1, data2, bias = 1)

def correlation_2d(X, Y):
    '''
    Covariance of two different random variables

    :param: two distribution objects
    :return: float
    '''
    data1 = X.get_data()
    data2 = Y.get_data()

    return np.corrcoef(data1, data2)