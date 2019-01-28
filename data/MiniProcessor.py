'''
Class preprocessing data obtained from the first training, merging data at the end for the final evaluation of the model.
'''
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class MiniProcessor:
    def __init__(self, Data):
        self.__transactionData = Data
        self.__currentData = None

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
        self.__sec_att_train, self.__sec_lab_train = pd.DataFrame(X_train_res, columns=name_train), pd.Series(y_train_res)

        print("[respamling finished]")

    # def merge_dataframes(self):
    #     '''
    #     Merge all the columns that were previously plitted to obtain more specific labels

    #     :param: df1: left data columns to be merged.    (Dataframe)
    #             df2: middle data columns to be merged.   (Dataframe)
    #             df3: right data columns to be merged.  (Datafrmae)
    #     :return: mergedFrame: Merged data columns. (Data Frame)
    #     '''
    #     frames = [df1, df2, df3]
    #     mergedFrame = pd.concat(frames)

    #     return mergedFrame.sort_index()

    def get_second_data(self,n):
        '''
        Split the data into train and test data for second classifier
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
        X = dfTrain.drop(['label','payment_label'], axis=1)
        self.__sec_att_train ,self.__sec_att_test, self.__sec_lab_train,self.__sec_lab_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle =True, stratify=y)
        print(self.__sec_lab_test)
        print(self.__sec_lab_train)
        self.__resample_data_SMOTE()

        print("[split_data finished]")
        return (self.__sec_att_test, self.__sec_lab_test,self.__sec_att_train,self.__sec_lab_train)

    def finalize_label(self, y_first, y_early, y_late):
        '''
        Convert the virtual label into the real label.

        :param: None
        :return: None
        '''
        finalLabel = y_first
        j=0
        k=0

        #TODO:  Change the value 3 into changable form so as to convert the corresponding value.
        for i in range(len(finalLabel)):
            if finalLabel.iloc[i] == 0: #early
                finalLabel.iloc[i] = y_early.iloc[j]
                j+=1

            if finalLabel.iloc[i] == 2: #late
                finalLabel.iloc[i] == y_late.iloc[k]
                k+=1

        return finalLabel
