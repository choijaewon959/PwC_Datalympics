"""
all the functions take the linear array(s) as input.
"""
import numpy as np
from scipy import stats

def mean(data):
    '''
    mean of linear data sets

    :param: data
    :return: float
    '''
    return sum(data) / len(data)

def median(data):
    '''
    median of linear data sets

    :param: data
    :return: float
    '''
    return np.median(data)

def mode(data):
    '''
    mode of linear data sets

    :param: data
    :return: float
    '''
    return stats.mode(data)

def stdDv(data):
    '''
    standard deviation of linear data sets

    :param: data
    :return: float
    '''
    return np.nanstd(data)

def variance(data):
    '''
    variance of linear data sets

    :param: data
    :return: float
    '''
    return np.nanvar(data)

def covariance(X, Y):
    '''
    Covariance of two different random variables

    :param: two distribution objects
    :return: float
    '''
    