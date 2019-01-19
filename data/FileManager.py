import pandas as pd

from .Column import Column
from .Row import Row
from .Config import *

class FileManager:
    def __init__(self, fileName):
        '''
        Constructor

        :param: name of csv file to be managed
        '''
        self.__rowSize = 0
        self.__colSize = 0

    def retrieve_data(self):
        '''
        Retrieve the data from the csv file and process to store data to datastructures.
        Update the row size and colum size.

        :param: name of file (str)
        :return: data from file
        '''
        dataFile = pd.read_csv('../loan_data/data/loan.csv') # TODO: file name should be converted to file path
        self.__rowSize = dataFile.shape[0]
        self.__colSize = dataFile.shape[1]

        return dataFile

    def get_row_size(self):
        '''
        Return the row count of the data file

        :param: None
        :return: Size of the row of data file (int)
        '''
        return self.__rowSize

    def get_col_size(self):
        '''
        Return the column count of the data file

        :param: None
        :return: Size of the column of data file (int)
        '''
        return self.__colSize
