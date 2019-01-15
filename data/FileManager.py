import csv

from Column import Column
from Row import Row

class FileManager:
    def __init__(self, fileName):
        '''
        Constructor

        :param: name of csv file to be managed
        '''
        self.__col = Column()
        self.__row = Row()
        retrieve_data(fileName)

    def retrieve_data(fileName):
        '''
        Retrieve the data from the csv file and process to store data to datastructures.

        :param: name of file (str)
        :return: None
        '''

        with open('../loan_data/data/loan.csv') as infile:
            