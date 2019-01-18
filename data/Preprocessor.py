"""
Data Preprocessor
"""
import pandas as pd
from util.Distribution import Distribution
from data.FileManager import FileManager
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, dataFrame):
        '''
        Constructor

        :param: data file to be converted into Distribution objects
        '''
        self.__distributionTable = {} # Table having distribution objects (key: name of data, value: distribution object).
        self.__keys = set(dataFrame.columns.values)    # string type keys for the table.
        self.__numOfKeys = 0    # number of keys.
        self.__dataFrame = dataFrame
        self.__trainDataFrame = None
        self.__testDataFrame = None

        self.__split_data()

    def data_to_distribution(self):
        '''
        Convert input data into distribution objects and store them into Table

        :param: data from csv file
        :return: None
        '''
        # TODO: Deal with string values
        for key in self.__keys:
            self.__distributionTable[key] = Distribution(self.__dataFrame, key)

    def __split_data(self):
        '''
        Split the dataframe into two datasets: Traning data, test data.

        :param: whole given data frame
        :return: None
        '''
        self.__trainDataFrame , self.__testDataFrame = train_test_split(self.__dataFrame, test_size = 0.2)

    def get_train_data(self):
        '''
        Retrieve the data frame for training

        :param: None
        :return: training data frame
        '''
        return self.__trainDataFrame

    def get_test_data(self):
        '''
        Retrieve the data frame for testing

        :param: None
        :return: test data frame
        '''
        return self.__testDataFrame

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
