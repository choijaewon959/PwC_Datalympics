"""
Distribution object that is initialized with linear datasets
"""
from . import Stat
import numpy as np
import pandas as pd

class Distribution:
    def __init__(self, dataFrame, key):
        '''
        Constructor.
        Update standard deviation and variance of the distribution.

        :param arr columData: raw data that includes name and values of data
        '''
        self.__dataName = key  #denotes the type of the data
        self.__data = dataFrame #data with actual values
        self.__stdDev = 0.0  #standard deviation of the distribution
        self.__variance = 0.0 #variance of the distribution

        self.__update_stdDv()
        self.__update_variance()

    def __str__(self):
        '''
        Stringify of distribution object.

        Prints out information of this distribution.
        :return string
        '''
        output = ""
        output += "Distribution of " + self.__dataName + '\n'
        output += "Standard deviation: " + self.__stdDev + '\n'
        output += "Variance: " + self.__variance + '\n'

        return output

    def __update_stdDv(self):
        '''
        Update the standard deviation of the distribution from data frame.

        :param arr newSample: one newly discovred datum.
        :return None
        '''
        key = self.__dataName
        dataTable = self.__data
        data = dataTable[key]

        self.__stdDev = data.std()


    def __update_variance(self):
        '''
        Update the variance of the distribution from data frame.

        :param arr newSample: one newly discovred datum.
        :return void
        '''
        key = self.__dataName
        dataTable = self.__data
        data = dataTable[key]

        self.__variance = data.var()

    def get_stdDv(self):
        '''
        Return standard deviation of this distribution

        :param None
        :return float
        '''
        return self.__stdDev

    def get_variance(self):
        '''
        Return variance of this distribution

        :param: None
        :return: float
        '''
        return self.__variance

    def get_data(self):
        '''
        Return the data frame

        :param None
        :return: str or int or float
        '''
        return self.__data
