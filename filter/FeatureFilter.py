"""
Filter object that contains the distribution information as well
"""
import pandas as pd
from util.Distribution import Distribution
from data.FileManager import FileManager

class FeatureFilter:
    def __init__(self, dataFrame):
        '''
        Constructor

        :param: data file to be converted into Distribution objects
        '''
        self.__distributionTable = {} # Table having distribution objects (key: name of data, value: distribution object).
        self.__keys = set(dataFrame.columns.values)    # string type keys for the table.
        self.__meaingfulFeatures = set() # features to be used after filtering.
        self.__numOfKeys = 0    # number of keys.

    def data_to_distribution(self, dataFrame):
        '''
        Convert input data into distribution objects and store them into Table

        :param: data from csv file
        :return: None
        '''
        # TODO: Deal with string values
        for key in self.__keys:
            self.__distributionTable[key] = Distribution(dataFrame, key)

    def get_distribution(self):
        '''
        Return the distribution table that contains all the distribution objects

        :param: None
        :return: dictionary (key: str, value: distribution object)
        '''
        return self.__distributionTable

    def get_features(self):
        '''
        Return the all the features from the data frame

        :param:None
        :return: set of strings that represent each feature.
        '''
        return self.__keys

    def get_feature_size(self):
        '''
        Return the total number of all the features of the data.

        :param: None
        :return: Number of all the features (int)
        '''
        return self.__numOfKeys

    def get_meaningful_features(self):
        '''
        Return the meaningful features that are no/less dependent to other features.

        :param: None
        :return: hash set of keys (str)
        '''
        return None
