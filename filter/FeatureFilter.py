"""
Filter object that contains the distribution information as well
"""

from util.Distribution import Distribution
from data.FileManager import FileManager

class FeatureFilter:
    def __init__(self, data):
        '''
        Constructor

        :param: data file to be converted into Distribution objects
        '''
        self.__distributionTable = {} # Table having distribution objects (key: name of data, value: distribution object)
        self.__keys = set()    # string type keys for the table
        self.__numOfKeys = 0    # number of keys

    def data_to_distribution(self, data):
        '''
        Convert input data into distribution objects and store them into Table

        :param: data from csv file
        :return: None
        '''
        return None

    def get_meaningful_features(self):
        '''
        Return the meaningful features that are no/less dependent to other features.

        :param: None
        :return: hash set of keys (str)
        '''
        return None

    def get_distribution(self):
        '''
        Return the distribution table that contains all the distribution objects

        :param: None
        :return: dictionary (key: str, value: distribution object)
        '''
        return self.__distributionTable
