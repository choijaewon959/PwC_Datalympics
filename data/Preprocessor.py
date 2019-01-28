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
        self.__transactionData = None # data mainly used.
        self.__meaningfulfeatures=[]

        self.__smallData = None
        self.__currentData = None

        self.__attributes_train = None
        self.__labels_train = None

        self.__attributes_test = None
        self.__labels_test = None

        self.__true_y = None
        self.late_nodes = None 
        self.early_nodes = None
        self.__featurefilter = FeatureFilter()

        self.__retrieve_data()
        self.__transactionData = datetime_data(self.__transactionData)

        self.__data_preprocess()

        #self.__dominant_feature_filter()

        #self.__extra_tree_classify()
        self.vendor_column()
        self.__select_k_best()
        self.classify_label()
        self.__split_data()
        self.__resample_data_SMOTE()

        print(self.__attributes_train)

        #self.__scale_data()
        #self.__graph()


    def __dominant_feature_filter(self):
        '''
        Filter out the dominant term to avoid overfitting

        :param: None
        :return: None
        '''
        self.__transactionData = self.__featurefilter.dominant_feature_filter(self.__transactionData)

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

        self.__meaningfulfeatures = self.__featurefilter.feature_score(self.__transactionData)

        cols= self.__meaningfulfeatures
        cols.append('label')
        cols.append('difference')
        dfdataset=self.__transactionData
        dfdataset= dfdataset[cols]

        self.__transactionData=dfdataset
        print(self.__meaningfulfeatures)
        print("Select K Best features replaced original feature list")

    def __extra_tree_classify(self):

        self.__meaningfulfeatures = self.__featurefilter.feature_score(self.__transactionData)

        cols= self.__meaningfulfeatures
        cols.append('loan_status')

        dfdataset=self.__transactionData
        dfdataset= dfdataset[cols]

        self.__transactionData=dfdataset
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

        data = pd.read_csv("../InvoicePayment-training.csv")

        self.__colnames= data.columns.values
        self.__transactionData = data
        print("[retrieve_data finished]")

    def __split_data(self):
        '''
        Split the dataframe into two datasets: Training data, test data.

        :param: whole given data frame
        :return: None
        '''
        print("split_data running...")
        # TODO: loan status may not be the label -> change to label accordingly.
        X = self.__transactionData.drop(['label', 'PwC_RowID','difference','payment_label'], axis = 1)
        y = self.__transactionData['label']

        self.__true_y = self.__transactionData['label']

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

    def change(self,val):
        return int(val[-2:])
    def change2(self,val):
        return int(val[-1:])

    def change3(self,val):
        return int(val[-6:])

    def __data_preprocess(self):

        dfTrain = self.__transactionData
        #copied data to refrain from warnings
        #dfTrain= dfTrain.copy()

        dfTrain= dfTrain[['PwC_RowID', 'BusinessTransaction', 'CompanyCode', 'CompanyName',
       'DocumentNo', 'DocumentType', 'DocumentTypeDesc', 'EntryDate',
       'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
       'InvoiceItemDesc', 'LocalCurrency', 'PaymentDate', 'PaymentDocumentNo',
       'Period', 'PO_FLag', 'PO_PurchasingDocumentNumber', 'PostingDate',
       'PurchasingDocumentDate', 'ReferenceDocumentNo', 'ReportingAmount',
       'TransactionCode', 'TransactionCodeDesc', 'UserName', 'VendorName',
       'VendorCountry', 'Year', 'PaymentDueDate', 'difference', 'label']]

        # print(dfTrain['VendorCountry'].unique().tolist())
        # li= dfTrain['VendorCountry'].unique().tolist()

        mapping = {'BusinessTransaction': {'Business transaction type 0002': 2 , 'Business transaction type 0003': 3, 'Business transaction type 0001': 1},
        'CompanyCode' : {'C002':2, 'C001':1, 'C003':3},
        'DocumentType': {'T03':3,'T04':4,'T02':2,'T01':1,'T09':9,'T07':7,'T06':6,'T08':8, 'T05':5},
        'DocumentTypeDesc': {'Vendor invoice': 0, 'Invoice receipt':1,'Vendor credit memo':2,'Vendor document':3,'TOMS (Jul2003)/ TWMS':4 ,'Interf.with SMIS-CrM':5,'Interf.with SMIS-IV':6 ,'Interface with PIMS':7},
        'PO_FLag': {'N': 0 , 'Y':1},
        'TransactionCode': {'TR 0005':0,'TR 0006':1,'TR 0002':2,'TR 0008':3,'TR 0007':4,'TR 0003':5,'TR 0004':6, 'TR 0001':7},
        }

        col  = ['CompanyName', 'EntryDate', 'DocumentTypeDesc', 'EntryTime',
                'InvoiceDate', 'LocalCurrency',
                'PO_PurchasingDocumentNumber', 'PostingDate', 'PurchasingDocumentDate',
                'ReportingAmount', 'TransactionCodeDesc', 'Year', 'PaymentDate', 'PaymentDueDate'
                ]
        dfTrain['UserName'] = dfTrain['UserName'].apply(self.change)
        dfTrain['TransactionCodeDesc'] = dfTrain['TransactionCodeDesc'].apply(self.change2)
        dfTrain['ReferenceDocumentNo'] = dfTrain['ReferenceDocumentNo'].apply(self.change3)
        dfTrain['DocumentNo'] = dfTrain['DocumentNo'].apply(self.change3)
        dfTrain['PaymentDocumentNo'] = dfTrain['PaymentDocumentNo'].apply(self.change3)
        dfTrain['InvoiceItemDesc'] = dfTrain['InvoiceItemDesc'].apply(self.change3)
        dfTrain['InvoiceDesc'] = dfTrain['InvoiceDesc'].apply(self.change3)

        print(dfTrain)
        dfTrain = dfTrain.replace(mapping)
        dfTrain = dfTrain.drop(col, axis=1)

        # dfTrain= dfTrain.loc[dfTrain['VendorCountry'] == 'HK']

        #print(dfTrain.columns)

        cols = ['PwC_RowID', 'BusinessTransaction', 'CompanyCode', 'DocumentType',
       'InvoiceAmount', 'PO_FLag', 'TransactionCode', 'UserName', 'difference',
       'label']

        for col in cols:
            print('Imputation with Median: %s' % (col))
            dfTrain[col].fillna(dfTrain[col].median(), inplace=True)

        print(dfTrain.describe)
        self.__transactionData = dfTrain

    # def get_labels(self):
    #     print(self.__transactionData['loan_status'].unique())
    #     return self.__transactionData['loan_status'].unique()

    def get_data(self):
        return self.__transactionData

    def additional_feature(self,val,unique):
        if(val == unique):
            return 1
        return 0

    def add_nodes(self):
        '''
        'dummy' nodes added
        '''
        stop = ['sub_grade','emp_length','loan_status','annual_inc','term','grade', 'delinq_2yrs','inq_last_6mths', 'pub_rec']
        for col in list(self.__transactionData.columns.values):
            try:
                if(stop.index(col) != -1):
                    continue
            except:
                if(len(self.__transactionData[col].unique()) < 30):
                    for uniq in self.__transactionData[col].unique():
                        self.__transactionData[col+' '+str(uniq)] = self.__transactionData[col].apply(self.additional_feature,args=(uniq,))
        self.__transactionData = self.__transactionData.drop(['home_ownership', 'initial_list_status','application_type'], axis=1)
        print(self.__transactionData.columns.values)
        print(len(self.__transactionData.columns.values))

    def __graph(self):
        visual = Visualization(self.__transactionData)
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

    def classify(self,val,early_nodes, late_nodes):
        if(val == 0):
            return 1
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
        dfTrain = self.__transactionData
        for i in range(0,10):
            early_nodes[i] = dfTrain.drop(dfTrain[dfTrain.difference < 1].index)['difference'].quantile(tmp)
            late_nodes[i] = dfTrain.drop(dfTrain[dfTrain.difference > -1].index)['difference'].quantile(tmp)
            tmp += 0.1
        self.__transactionData['payment_label'] = self.__transactionData['difference'].apply(self.classify, args=(early_nodes,late_nodes,))
        self.early_nodes = early_nodes
        self.late_nodes = late_nodes

    def vendor_apply(self,val,name):
        if(val == name):
            return 1
        return 0

    def vendor_column(self):
        for name in list(self.__transactionData['VendorName'].unique()):
            if(len(self.__transactionData[self.__transactionData.VendorName == name].index)> 10000):
                self.__transactionData[name] = self.__transactionData['VendorName'].apply(self.vendor_apply, args=(name,))
        #print(self.__transactionData)
        for country in list(self.__transactionData['VendorCountry'].unique()):
            if(len(self.__transactionData[self.__transactionData.VendorCountry == country].index)> 10000):
                self.__transactionData[country] = self.__transactionData['VendorCountry'].apply(self.vendor_apply, args=(name,))

        dfTrain =self.__transactionData.copy()
        #print(dfTrain.loc[dfTrain.index[dfTrain['VendorName'] == 'Vendor 01024'].tolist()])
        #print(dfTrain['Vendor 01899'].value_counts())
        dfTrain=dfTrain.drop('VendorName', axis=1)
        dfTrain=dfTrain.drop('VendorCountry', axis=1)
        print(dfTrain.columns.values)
        self.__transactionData = dfTrain
