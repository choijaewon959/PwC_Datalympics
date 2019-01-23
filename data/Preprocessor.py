"""
Data Preprocessor
"""
import pandas as pd
import numpy as np
from util.Distribution import Distribution
from sklearn.model_selection import train_test_split
from sklearn import datasets


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

        self.__smallData = None

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
        print("retrieve_data running...")
        # TODO: file name should be converted to file path

        """
        DONT ERASE THE COMMENTED FILE PATH.
        USE YOUR OWN FILE PATH AND COMMENT OUT WHEN YOU PUSH.
        """
        #data = pd.read_csv(r"C:\Users\lasts\Google Drive\Etc\Coding\Data_lympics\Deeplearning\loan.csv")
        #data = pd.read_csv("../loan_data/data/loan.csv")
        data = pd.read_csv("../loan_data/data/loanfull.csv")
        #low_memory was added to avoid data compression


        # #Using sklearn datasets
        # iris = datasets.load_wine()
        
        # data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
        #              columns= iris['feature_names'] + ['target'])

        #Taemin's debugging tool@!!
        # data = pd.read_csv("Deeplearning/loan.csv")

        self.__colnames= data.columns.values
        self.__loanData = data

    def __split_data(self):
        '''
        Split the dataframe into two datasets: Traning data, test data.

        :param: whole given data frame
        :return: None
        '''
        print("split_data running...")
        # TODO: loan status may not be the label -> change to label accordingly.
        X = self.__loanData.drop('loan_status', axis = 1)
        y = self.__loanData['loan_status']

        self.__attributes_train, self.__attributes_test, self.__labels_train, self.__labels_test = train_test_split(X, y, test_size=0.2, random_state = 1)

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

    def convert_label(self,Y):
        '''
        Converting the label into binary vector forms for keras neural network output layer.

        :param: label
        :return: converted label (int vector)
        '''
        l = np.array([[0,0,0,0,0,0,0,0,0,0]])
        tmp = np.array([0,0,0,0,0,0,0,0,0,0])
        for i in Y:
            tmp[int(i)] = 1
            l = np.append(l,[tmp],axis=0)
            tmp = np.array([0,0,0,0,0,0,0,0,0,0])
        l = np.delete(l,0,0)
        YY = pd.DataFrame(l)
        return YY

    def __temp_data_process(self):
        '''
        temporary data processor for loan.csv file
        erase unrelated columns and imputation is done.
        prints some debugging messages.

        :param: none
        :return: none
        '''
        dfTrain = self.__loanData
        #copied data to refrain from warnings
        dfTrain= dfTrain.copy()

        # TODO: when dealing with real data, columns has to be selected otherwise
        #erase unrelated columns
        dfTrain= dfTrain[['loan_amnt', 'funded_amnt',
               'term', 'int_rate', 'installment', 'sub_grade',
               'emp_length', 'annual_inc', 'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths'
               ,'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec'
               ,'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt'
               ,'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'
               ,'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt'
            ]]

        # TODO: Feature transformation can be done beforehand or after
        # when the data is normalized to numerical data, these steps should be omitted.
        dfTrain['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
        dfTrain['term'] = dfTrain['term'].astype(int)

        # dfTrain['term']= pd.to_numeric(dfTrain['term'], errors='coerce')
        #dfTrain['term'] = dfTrain.term.astype(float)


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
        cols = ['loan_amnt', 'funded_amnt',
               'term', 'int_rate', 'installment', 'sub_grade',
               'emp_length', 'annual_inc', 'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths'
               ,'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec'
               ,'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt'
               ,'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'
               ,'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt'
            ]
        for col in cols:
            print('Imputation with Median: %s' % (col))
            dfTrain[col].fillna(dfTrain[col].median(), inplace=True)

        cols=['loan_status']
        for col in cols:
            print('Imputation with Zero: %s' % (col))
            dfTrain[col].fillna(0, inplace=True)
        print('Missing value imputation done.')

        print(dfTrain['loan_status'].unique())

        self.__loanData = dfTrain

    def get_labels(self):
        print(self.__loanData['loan_status'].unique())
        return self.__loanData['loan_status'].unique()


    def get_data(self):
        return self.__loanData
