"""
Data Preprocessor
"""
import pandas as pd
from util.Distribution import Distribution
from data.FileManager import FileManager
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self):
        '''
        Constructor

        :param: data file to be converted into Distribution objects
        '''
        self.__distributionTable = {} # Table having distribution objects (key: name of data, value: distribution object).
        self.__keys = None   # string type keys for the table.
        self.__numOfKeys = 0    # number of keys.
        self.__loanData = None # data mainly used.

        self.__attributes_train = None
        self.__labels_train = None

        self.__attributes_test = None
        self.__labels_test = None

        self.__retrieve_data()
        self.__split_data()

    def __retrieve_data(self):
        '''
        Retrieve the data from the csv file and process to store data to datastructures.
        Update the row size and colum size.

        :param: name of file (str)
        :return: data from file
        '''
        # TODO: file name should be converted to file path
        data = pd.read_csv('../loan_data/data/loan.csv', names= self.__keys)
        self.__keys = data.columns.values
        self.__loanData = data

    def data_to_distribution(self):
        '''
        Convert input data into distribution objects and store them into Table

        :param: data from csv file
        :return: None
        '''
        # TODO: Deal with string values
        for key in self.__keys:
            self.__distributionTable[key] = Distribution(self.__loan_data, key)

    def __split_data(self):
        '''
        Split the dataframe into two datasets: Traning data, test data.

        :param: whole given data frame
        :return: None
        '''
        # TODO: loan status may not be the label -> change to label accordingly.
        X = self.__loanData.drop('load_status', axis = 1)
        y = self.__loanData['load_status']

        self.__attributes_train, self.__attributes_test, self.__labels_train, self.__labels_test = train_test_split(X, y, test_size=0.2)

    def get_train_attributes(self):
        '''
        Return the attributes of the data for training.

        :param: None
        :return: data attributes
        '''
        return self.__attributes_train

    def get_train_labels(self):
        '''
        Return the labels of the data for training.

        :param: None
        :return: categorical labels
        '''
        return self.__labels_train

    def get_test_attributes(self):
        '''
        Return the attributes of the data for test.

        :param: None
        :return: data attributes
        '''
        return self.__attributes_test

    def get_test_labels(self):
        '''
        Return the labels of the data for test.

        :param: None
        :return: categorical labels
        '''
        return self.__labels_test

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
