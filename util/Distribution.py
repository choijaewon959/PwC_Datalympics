"""
Distribution object that is initialized with linear datasets
"""
from . import Stat

class Distribution:
    def __init__(self, columnData):
        '''
        Constructor

        :param arr columData: raw data that includes name and values of data
        '''
        self.__dataName = columnData[0]  #denotes the type of the data
        self.__data = columnData[1:] #data with actual values
        self.__stdDev = Stat.stdDv(self)  #standard deviation of the distribution
        self.__variance = Stat.variance(self) #variance of the distribution

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

    def update_stdDv(self, newSample):
        '''
        Update the standard deviation of the distribution when spotted new data sample.
        
        :param arr newSample: one newly discovred datum.
        :return void
        '''
        newList = self.__data + newSample
        self.__stdDev = Stat.stdDv(newList)

    def update_variance(self, newSample):
        '''
        Update the variance of the distribution when when spotted new data sample.

        :param arr newSample: one newly discovred datum.
        :return void
        '''
        newList = self.__data + newSample
        self.__variance = Stat.variance(newList)

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
        Return the array of data values

        :param None
        :return: str or int or float
        '''
        return self.__data