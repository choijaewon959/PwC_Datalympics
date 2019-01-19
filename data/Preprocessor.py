"""
Data Preprocessor
"""
import pandas as pd
from util.Distribution import Distribution
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self):
        '''
        Constructor

        :param: data file to be converted into Distribution objects
        '''
        self.__distributionTable = {} # Table having distribution objects (key: name of data, value: distribution object).
        self.__colnames = None # string type keys for the table.
        self.__numOfKeys = 0    # number of keys.
        self.__loanData = None # data mainly used.

        self.__attributes_train = None
        self.__labels_train = None

        self.__attributes_test = None
        self.__labels_test = None

        self.__retrieve_data()
        # TODO: function call for preprocessing data
        self.__temp_data_process()
        self.__split_data()

    def __retrieve_data(self):
        '''
        Retrieve the data from the csv file and process to store data to datastructures.
        Update the row size and colum size.

        :param: name of file (str)
        :return: data from file
        '''
        # TODO: file name should be converted to file path
        data = pd.read_csv(r"C:\Users\user\Desktop\proj\Data_lympics\Deeplearning\loan.csv")
        self.__colnames= data.columns.values
        self.__loanData = data

    def data_to_distribution(self):
        '''
        Convert input data into distribution objects and store them into Table

        :param: data from csv file
        :return: None
        '''
        # TODO: Deal with string values
        for key in self.__colnames:
            self.__distributionTable[key] = Distribution(self.__dataFrame, key)

    def __split_data(self):
        '''
        Split the dataframe into two datasets: Traning data, test data.

        :param: whole given data frame
        :return: None
        '''
        # TODO: loan status may not be the label -> change to label accordingly.
        X = self.__loanData.drop('loan_status', axis = 1)
        y = self.__loanData['loan_status']

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
        return self.__colnames

    def get_feature_size(self):
        '''
        Return the total number of all the features of the data.

        :param: None
        :return: Number of all the features (int)
        '''
        return self.__numOfKeys

    def __temp_data_process(self):

        dfTrain = self.__loanData
        # TODO: when dealing with real data, columns has to be selected otherwise
        #erase unrelated columns
        dfTrain= dfTrain[['member_id', 'loan_amnt', 'funded_amnt',
               'term', 'int_rate', 'installment', 'sub_grade',
               'emp_length', 'annual_inc', 'loan_status']]

        # TODO: Feature transformation can be done beforehand or after
        # when the data is normalized to numerical data, these steps should be omitted.
        dfTrain['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
        dfTrain['term']= pd.to_numeric(dfTrain['term'], errors='coerce')

        #print('Transform: sub_grade...')
        dfTrain['sub_grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
        dfTrain['sub_grade'].replace(to_replace='B', value='1', regex=True, inplace=True)
        dfTrain['sub_grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
        dfTrain['sub_grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
        dfTrain['sub_grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
        dfTrain['sub_grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
        dfTrain['sub_grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
        dfTrain['sub_grade'] = pd.to_numeric(dfTrain['sub_grade'], errors='coerce')

        #print('Transform: emp_length...')
        dfTrain['emp_length'].replace('n/a', '0', inplace=True)
        dfTrain['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
        dfTrain['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
        dfTrain['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
        dfTrain['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
        dfTrain['emp_length'] = pd.to_numeric(dfTrain['emp_length'], errors='coerce')

        #print('Transform: annual_inc...')
        dfTrain['annual_inc']= pd.to_numeric(dfTrain['annual_inc'], errors='coerce')

        #print('Transform: loan_status...')
        # for loan status just gave random 0 / 1 of binary representation of good or bad loan
        dfTrain['loan_status'].replace('n/a', '0', inplace=True)
        dfTrain['loan_status'].replace(to_replace='Fully Paid', value='0', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Current', value='1', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Charged Off', value='2', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='In Grace Period', value='3', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Late (31-120 days)', value='4', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Late (16-30 days)', value='5', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Issued', value='6', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Default', value='7', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Does not meet the credit policy. Status:Fully Paid Off', value='8', regex=True, inplace=True)
        dfTrain['loan_status'].replace(to_replace='Does not meet the credit policy. Status:Charged Off', value='9', regex=True, inplace=True)
        dfTrain['loan_status'] = pd.to_numeric(dfTrain['loan_status'], errors='coerce')


        '''
        #data imputation
        '''
        cols = ['term', 'loan_amnt', 'funded_amnt', 'int_rate', 'sub_grade', 'annual_inc', 'emp_length', 'installment']
        for col in cols:
            print('Imputation with Median: %s' % (col))
            dfTrain[col].fillna(dfTrain[col].median(), inplace=True)

        cols=['member_id', 'loan_status']
        for col in cols:
            print('Imputation with Zero: %s' % (col))
            dfTrain[col].fillna(0, inplace=True)
        print('Missing value imputation done.')


        self.__loanData = dfTrain