'''
Class preprocessing data obtained from the first training, merging data at the end for the final evaluation of the model.
'''
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import datasets, preprocessing

class MiniProcessor:
    def __init__(self, Data):
        self.__transactionData = Data

        self.__filteredData = None

        #second classifier input
        self.__sec_att_train = None
        self.__sec_lab_train = None

        self.__sec_att_test = None
        self.__sec_lab_train = None

    def __resample_data_SMOTE(self):
        '''
        Resampling imbalanced data with smote algorithm. (Oversampling)
        Update train attributes, train labels

        :param: None
        :return: None
        '''
        name_train = self.__sec_att_train.columns
        print("resampling data...")

        sm = SMOTE(random_state=6)
        X_train_res, y_train_res = sm.fit_resample(self.__sec_att_train, self.__sec_lab_train)
        self.__sec_att_train, self.__sec_lab_train = pd.DataFrame(X_train_res, columns=name_train), pd.DataFrame(y_train_res)
        
        # self.__sec_att_test = pd.DataFrame(self.__sec_att_test)
        # self.__sec_att_train = pd.DataFrame(self.__sec_att_train)

        print("[respamling finished]")
 
    def __scale_data(self):
        '''
        Normalize data.

        :param: data to be normalized. (Data frame)
        :return: nomalized data. (Data frame)
        '''
        X_train = self.__sec_att_train
        X_test = self.__sec_att_test

        names_train = X_train.columns
        names_test = X_test.columns

        # #Standard Scaler
        # scaling = preprocessing.StandardScaler()
        # scaled = scaling.fit_transform(X_train)

        #Minimax Scaler
        scaling = preprocessing.StandardScaler()

        X_train_scaled = scaling.fit_transform(X_train)
        X_test_scaled = scaling.fit_transform(X_test)

        self.__sec_att_train = pd.DataFrame(X_train_scaled, columns = names_train)
        self.__sec_att_test = pd.DataFrame(X_test_scaled, columns = names_test)

    def get_second_data(self,n):
        '''
        Split the data into train and test data for second classifier.
        Only used for classification model.

        :parameter: Label number
        :return : tuple of label n data, splited into test, train data
        '''
        dfTrain = self.__transactionData
        if(n == 0):
            dfTrain = dfTrain.drop(dfTrain[dfTrain.payment_label < 10].index)
            dfTrain = dfTrain.drop(dfTrain[dfTrain.payment_label > 19].index)
        elif(n==2):
            dfTrain = dfTrain.drop(dfTrain[dfTrain.payment_label < 40].index)
            dfTrain = dfTrain.drop(dfTrain[dfTrain.payment_label > 50].index)
        y = dfTrain['payment_label']
        X = dfTrain.drop(['label','payment_label','difference'], axis=1)

        self.__sec_att_train ,self.__sec_att_test, self.__sec_lab_train,self.__sec_lab_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle =True, stratify=y)
        self.__resample_data_SMOTE()

        # self.__sec_att_test = pd.DataFrame(self.__sec_att_test)
        # self.__sec_att_train = pd.DataFrame(self.__sec_att_train)

        print("[split_data finished]")
        return (self.__sec_att_test, self.__sec_lab_test,self.__sec_att_train,self.__sec_lab_train)

    def get_poly_data(self,n):
        '''
        Split the data into train and test data for second classifier.
        Only used for regression model.

        :parameter: Label number
        :return : tuple of label n data, splited into test, train data
        '''

        dfTrain = self.__transactionData
        if(n == 0):
            dfTrain = dfTrain.drop(dfTrain[dfTrain.label < n].index)
            dfTrain = dfTrain.drop(dfTrain[dfTrain.label > n].index)
        elif(n==2):
            dfTrain = dfTrain.drop(dfTrain[dfTrain.label < n].index)
            dfTrain = dfTrain.drop(dfTrain[dfTrain.label > n].index)
        y = dfTrain['difference']
        X = dfTrain.drop(['label','payment_label','difference'], axis=1)

        self.__sec_att_train ,self.__sec_att_test, self.__sec_lab_train,self.__sec_lab_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle =True)
        #self.__resample_data_SMOTE()

        #scale to use PCA
        self.__scale_data()
        print("[split_data finished]")
        return (self.__sec_att_test, self.__sec_lab_test,self.__sec_att_train,self.__sec_lab_train)

    