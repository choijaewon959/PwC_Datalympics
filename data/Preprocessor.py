"""
Data Preprocessor
"""
import pandas as pd
import numpy as np
import time
import math
from util.Distribution import Distribution
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
from imblearn.combine import SMOTETomek
from evaluation.Visualization import *
from data.FeatureFilter import FeatureFilter
from data.datehandler import *


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
        self.__meaningfulfeatures=[]

        self.__smallData = None
        self.__currentData = None

        self.__attributes_train = None
        self.__labels_train = None

        self.__attributes_test = None
        self.__labels_test = None

        self.__true_y = None

        #second classifier input
        self.sec_att_train = None
        self.sec_lab_train = None

        self.sec_att_test = None
        self.sec_lab_train = None

        #self.__featurefilter = FeatureFilter()

        self.__retrieve_data()
        self.__loanData = datetime_data(self.__loanData)

        self.__data_preprocess()


        #print(self.__loanData)

        #self.__dominant_feature_filter()
        # TODO: function call for preprocessing data

        #self.__temp_data_process()
        #self.add_nodes()

        #self.__select_k_best()
        #self.__extra_tree_classify()

        self.__split_data()
        #self.__resample_data_SMOTE()

        #self.__scale_data()
        #self.__graph()

    def __dominant_feature_filter(self):
        '''
        Filter out the dominant term to avoid overfitting

        :param: None
        :return: None
        '''
        self.__loanData = self.__featurefilter.dominant_feature_filter(self.__loanData)


    def __scale_data(self):
        '''
        Normalize data.

        :param: data to be normalized. (Data frame)
        :return: nomalized data. (Data frame)
        '''
        names_train = self.__attributes_train.columns
        names_test = self.__attributes_test.columns

        # #Standard Scaler
        # scaling = preprocessing.StandardScaler()
        # scaled = scaling.fit_transform(X_train)

        #Minimax Scaler
        scaling = preprocessing.MinMaxScaler(feature_range= (-1,1))

        scaled_train = scaling.fit_transform(self.__attributes_train)
        scaled_test = scaling.fit_transform(self.__attributes_test)

        self.__attributes_train = pd.DataFrame(scaled_train, columns = names_train)
        self.__attributes_test = pd.DataFrame(scaled_test, columns = names_test)

    def __select_k_best(self):


        self.__meaningfulfeatures = self.__featurefilter.feature_score(self.__loanData.drop('loan_status',axis=1))

        cols= self.__meaningfulfeatures
        cols.append('new_loan_status')
        cols.append('loan_status')

        dfdataset=self.__loanData
        dfdataset= dfdataset[cols]

        self.__loanData=dfdataset
        print("Select K Best features replaced original feature list")

    def __extra_tree_classify(self):

        self.__meaningfulfeatures = self.__featurefilter.feature_score(self.__loanData)

        cols= self.__meaningfulfeatures
        cols.append('loan_status')

        dfdataset=self.__loanData
        dfdataset= dfdataset[cols]

        self.__loanData=dfdataset
        print("Extra_tree_classify() features replaced original feature list")


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
        #data = pd.read_csv("Deeplearning/loan.csv")
        #data = pd.read_csv("../loan_data/data/loanfull.csv")
        #low_memory was added to avoid data compression


        # #Using sklearn datasets
        # iris = datasets.load_wine()

        # data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
        #              columns= iris['feature_names'] + ['target'])

        #Taemin's debugging tool@!!
        #data = pd.read_csv("Deeplearning/loan.csv")
        data = pd.read_csv("../InvoicePayment-training.csv")

        self.__colnames= data.columns.values
        self.__loanData = data
        print("[retrieve_data finished]")

    def __split_data(self):
        '''
        Split the dataframe into two datasets: Traning data, test data.

        :param: whole given data frame
        :return: None
        '''
        print("split_data running...")
        # TODO: loan status may not be the label -> change to label accordingly.
        X = self.__loanData.drop(['difference', 'label'], axis = 1)
        y = self.__loanData['label']

        self.__true_y = self.__loanData['label']

        self.__attributes_train, self.__attributes_test, self.__labels_train, self.__labels_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle =True, stratify=y)
        print("[split_data finished]")

    def __resample_data_SMOTE(self):
        '''
        Resampling imbalanced data with smote algorithm. (Oversampling)
        Update train attributes, train labels

        :param: None
        :return: None
        '''
        name_train = self.__attributes_train.columns
        print("resampling data...")

        sm = SMOTE(random_state=12)
        X_train_res, y_train_res = sm.fit_resample(self.__attributes_train, self.__labels_train)
        self.__attributes_train, self.__labels_train = pd.DataFrame(X_train_res, columns=name_train), pd.Series(y_train_res)

        print("[respamling finished]")

    def __resample_data_NearMiss(self):
        '''
        Resampling imbalanced data with near miss algorithm. (Undersampling)

        :param: None
        :return: None
        '''
        name_train = self.__attributes_train.columns

        print("resampling data...")
        nm = NearMiss(random_state=6)
        X_train_res, y_train_res = nm.fit_resample(self.__attributes_train, self.__labels_train)
        self.__attributes_train, self.__labels_train = pd.DataFrame(X_train_res, columns = name_train), pd.Series(y_train_res)
        print("[respamling finished]")

    def __scale_data(self):
        '''
        Normalize data.

        :param: data to be normalized. (Data frame)
        :return: nomalized data. (Data frame)
        '''
        X_train = self.__attributes_train
        X_test = self.__attributes_test

        names_train = X_train.columns
        names_test = X_test.columns

        # #Standard Scaler
        # scaling = preprocessing.StandardScaler()
        # scaled = scaling.fit_transform(X_train)

        #Minimax Scaler
        scaling = preprocessing.MinMaxScaler(feature_range= (-1,1))

        X_train_scaled = scaling.fit_transform(X_train)
        X_test_scaled = scaling.fit_transform(X_test)

        self.__attributes_train = pd.DataFrame(X_train_scaled, columns = names_train)
        self.__attributes_test = pd.DataFrame(X_test_scaled, columns = names_test)

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
        print("__temp_data_process running...")
        #--------other debugging messages are omitted/ commented for simplifying purposes -------
        start_time = time.time()

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
        mapping = {'loan_status': {'Fully Paid': 0 , 'Current': -1, 'Charged Off': 2,
                    'In Grace Period': 3, 'Late (31-120 days)': 4, 'Late (16-30 days)': 5,
                    'Issued': 6, 'Default': 7, 'Does not meet the credit policy. Status:Fully Paid': 8,
                    'Does not meet the credit policy. Status:Charged Off': 9}
        }
        dfTrain= dfTrain.replace(mapping)

        # dfTrain['loan_status'].replace('n/a', '0', inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Fully Paid', value='0', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Current', value='1', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Charged Off', value='2', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='In Grace Period', value='3', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Late (31-120 days)', value='4', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Late (16-30 days)', value='5', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Issued', value='6', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Default', value='7', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Does not meet the credit policy. Status:Fully Paid Off', value='8', regex=True, inplace=True)
        # dfTrain['loan_status'].replace(to_replace='Does not meet the credit policy. Status:Charged Off', value='9', regex=True, inplace=True)
        # dfTrain['loan_status'] = pd.to_numeric(dfTrain['loan_status'], errors='coerce')

        # print(dfTrain['loan_status'].unique())

        '''
        #data imputation
        '''
        cols = ['loan_amnt', 'funded_amnt',
               'term', 'int_rate', 'installment', 'sub_grade',
               'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths'
               ,'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec'
               ,'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt'
               ,'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'
               ,'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt'
            ]

        for col in cols:
            #print('Imputation with Median: %s' % (col))
            dfTrain[col].fillna(dfTrain[col].median(), inplace=True)

        cols=['loan_status']
        for col in cols:
            #print('Imputation with Zero: %s' % (col))
            dfTrain[col].fillna(0, inplace=True)
        #print('Missing value imputation done.')
        self.__currentData = dfTrain[dfTrain.loan_status < 0]
        dfTrain = dfTrain.drop(dfTrain[dfTrain.loan_status < 0].index)
        self.__loanData = dfTrain

        #add new column which merges certain labels
        self.__loanData['new_loan_status'] = self.__loanData['loan_status'].apply(self.add_column)

        print(self.__loanData['new_loan_status'].unique())
        tempProcessTime= time.time() - start_time
        print("[tempProcessTime finished with %.2f seconds]"  % tempProcessTime)

    def get_data(self):
        return self.__loanData

    def change(self,val):
        return int(val[-2:])
    def change2(self,val):
        return int(val[-1:])
    def __data_preprocess(self):

        dfTrain = self.__loanData
        #copied data to refrain from warnings
        dfTrain= dfTrain.copy()


        dfTrain= dfTrain[['PwC_RowID', 'BusinessTransaction', 'CompanyCode', 'CompanyName',
       'DocumentNo', 'DocumentType', 'DocumentTypeDesc', 'EntryDate',
       'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
       'InvoiceItemDesc', 'LocalCurrency', 'PaymentDate', 'PaymentDocumentNo',
       'Period', 'PO_FLag', 'PO_PurchasingDocumentNumber', 'PostingDate',
       'PurchasingDocumentDate', 'ReferenceDocumentNo', 'ReportingAmount',
       'TransactionCode', 'TransactionCodeDesc', 'UserName', 'VendorName',
       'VendorCountry', 'Year', 'PaymentDueDate', 'difference', 'label']]

        print(dfTrain['VendorCountry'].unique().tolist())
        li= dfTrain['VendorCountry'].unique().tolist()

        mapping={}

        num=0
        for i in li:
            mapping['VendorCountry']= { i : num }


        mapping = {'BusinessTransaction': {'Business transaction type 0002': 2 , 'Business transaction type 0003': 3, 'Business transaction type 0001': 1},
        'CompanyCode' : {'C002':2, 'C001':1, 'C003':3},
        'DocumentType': {'T03':3,'T04':4,'T02':2,'T01':1,'T09':9,'T07':7,'T06':6,'T08':8},
        'DocumentTypeDesc': {'Vendor invoice': 0, 'Invoice receipt':1,'Vendor credit memo':2,'Vendor document':3,'TOMS (Jul2003)/ TWMS':4 ,'Interf.with SMIS-CrM':5,'Interf.with SMIS-IV':6 ,'Interface with PIMS':7},
        'PO_FLag': {'N': 0 , 'Y':1},
        'TransactionCode': {'TR 0005':0,'TR 0006':1,'TR 0002':2,'TR 0008':3,'TR 0007':4,'TR 0003':5,'TR 0004':6, 'TR 0001':7},

        }

        col  = ['CompanyName', 'DocumentNo', 'EntryDate', 'DocumentTypeDesc', 'EntryTime',
                'InvoiceDate', 'InvoiceDesc', 'InvoiceItemDesc', 'LocalCurrency', 'PaymentDocumentNo',
                'Period', 'PO_PurchasingDocumentNumber', 'PostingDate', 'PurchasingDocumentDate', 'ReferenceDocumentNo',
                'ReportingAmount', 'TransactionCodeDesc', 'Year', 'VendorName' , 'VendorCountry', 'PaymentDate', 'PaymentDueDate'
                ]

        dfTrain['UserName'] = dfTrain['UserName'].apply(self.change)
        dfTrain['TransactionCodeDesc'] = dfTrain['TransactionCodeDesc'].apply(self.change2)
        dfTrain = dfTrain.replace(mapping)
        dfTrain = dfTrain.drop(col, axis=1)

        print(dfTrain.dtypes)

        self.__loanData = dfTrain

    def get_labels(self):
        print(self.__loanData['loan_status'].unique())
        return self.__loanData['loan_status'].unique()

    def additional_feature(self,val,unique):
        if(val == unique):
            return 1
        return 0

    def add_nodes(self):
        '''
        'dummy' nodes added
        '''
        stop = ['sub_grade','emp_length','loan_status','annual_inc','term','grade', 'delinq_2yrs','inq_last_6mths', 'pub_rec']
        for col in list(self.__loanData.columns.values):
            try:
                if(stop.index(col) != -1):
                    continue
            except:
                if(len(self.__loanData[col].unique()) < 30):
                    for uniq in self.__loanData[col].unique():
                        self.__loanData[col+' '+str(uniq)] = self.__loanData[col].apply(self.additional_feature,args=(uniq,))
        self.__loanData = self.__loanData.drop(['home_ownership', 'initial_list_status','application_type'], axis=1)
        print(self.__loanData.columns.values)
        print(len(self.__loanData.columns.values))

    def __graph(self):
        visual = Visualization(self.__loanData)
        visual.plot_heatmap()

    def add_column(self,val):
        if(val == 5 or val == 4):
            return 3
        return val

    def get_true_y(self):
        '''
        return the y value from the raw data.

        :param: None
        :return: true_y
        '''
        return self.__true_y

    def change(self,val):
        return int(val[-2:])
    def change2(self,val):
        return int(val[-1:])
    def __data_process(self):
            mapping = {'BusinessTransaction': {'Business transaction type 0002': 2 , 'Business transaction type 0003': 3, 'Business transaction type 0001': 1},
                       'CompanyCode' : {'C002':2, 'C001':1, 'C003':3},
                       'DocumentType': {'T03':3,'T04':4,'T02':2,'T01':1,'T09':9,'T07':7,'T06':6,'T08':8},
                       'DocumentTypeDesc': {'Vendor invoice': 0, 'Invoice receipt':1,'Vendor credit memo':2,'Vendor document':3,'TOMS (Jul2003)/ TWMS':4 ,'Interf.with SMIS-CrM':5,'Interf.with SMIS-IV':6 ,'Interface with PIMS':7},
                       'PO_FLag': {'N': 0 , 'Y':1},
                       'TransactionCode': {'TR 0005':0,'TR 0006':1,'TR 0002':2,'TR 0008':3,'TR 0007':4,'TR 0003':5,'TR 0004':6, 'TR 0001':7}
            }
            col  = ['CompanyName', 'DocumentNo']
            self.__loanData['UserName'] = self.__loanData['UserName'].apply(self.change)
            self.__loanData['TransactionCodeDesc'] = self.__loanData['TransactionCodeDesc'].apply(self.change2)
            self.__loanData = self.__loanData.replace(mapping)
            self.__loanData.drop(col,axis=1)
    def classify(self,val,early_nodes, late_nodes):
        if(val == 0):
            return 0
        elif(val > 0):
            if(val >= early_nodes[9]):
                return 19
            elif(val>= early_nodes[8]):
                return 19
            elif(val>= early_nodes[7]):
                return 18
            elif(val>= early_nodes[6]):
                return 17
            elif(val>= early_nodes[5]):
                return 16
            elif(val>= early_nodes[4]):
                return 15
            elif(val>= early_nodes[3]):
                return 14
            elif(val>= early_nodes[2]):
                return 13
            elif(val>= early_nodes[1]):
                return 12
            elif(val>= early_nodes[0]):
                return 11
            return 10
        elif(val < 0):
            if(val >= late_nodes[9]):
                return 40
            elif(val>= late_nodes[8]):
                return 41
            elif(val>= late_nodes[7]):
                return 42
            elif(val>= late_nodes[6]):
                return 43
            elif(val>= late_nodes[5]):
                return 44
            elif(val>= late_nodes[4]):
                return 45
            elif(val>= late_nodes[3]):
                return 46
            elif(val>= late_nodes[2]):
                return 47
            elif(val>= late_nodes[1]):
                return 48
            elif(val>= late_nodes[0]):
                return 49
            return 50
    def classify_label(self):
        '''
        converts payment time into labels according to the distribution of payment time
        nodes classifying the label is defined by the quantile values of the distribution
        :param = list/dataframe of the payment time, loanData
        :return = loanData with a new column of label
        '''
        tmp = 0.1
        early_nodes = dict()
        late_nodes = dict()
        dfTrain = self.__loanData
        for i in range(0,10):
            early_nodes[i] = dfTrain.drop(dfTrain[dfTrain.difference < 1].index)['difference'].quantile(tmp)
            late_nodes[i] = dfTrain.drop(dfTrain[dfTrain.difference > -1].index)['difference'].quantile(tmp)
            tmp += 0.1
        self.__loanData['payment_label'] = self.__loanData['difference'].apply(self.classify, args=(early_nodes,late_nodes,))
        print(early_nodes, late_nodes)
        print(self.__loanData[['payment_label','difference']])
        print(self.__loanData['payment_label'].value_counts())
    def vendor_apply(self,val,name):
        if(val == name):
            return 1
        return 0
    def vendor_column(self):
        for name in list(self.__loanData['VendorName'].unique()):
            if(len(self.__loanData[self.__loanData.VendorName == name].index)> 10000):
                self.__loanData[name] = self.__loanData['VendorName'].apply(self.vendor_apply, args=(name,))
