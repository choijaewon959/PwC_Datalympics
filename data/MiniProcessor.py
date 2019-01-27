'''
Class preprocessing data obtained from the first training, merging data at the end for the final evaluation of the model.
'''
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class MiniProcessor:
    def __init__(self, loanData):
        self.__loanData = loanData
        self.__currentData = None

        self.__attributes_train = None
        self.__labels_train = None

        self.__attributes_test = None
        self.__labels_test = None

        #second classifier input
        self.sec_att_train = None
        self.sec_lab_train = None

        self.sec_att_test = None
        self.sec_lab_train = None

        self.__split_second_data()

    def __split_second_data(self) :
        '''
        Split the dataframe into two.
        '''
        dfTrain = self.__loanData
        dfTrain = dfTrain.drop(dfTrain[dfTrain.loan_status < 3].index)
        dfTrain = dfTrain.drop(dfTrain[dfTrain.loan_status > 5].index)
        y = dfTrain['loan_status']
        X = dfTrain.drop('loan_status', axis=1)

        self.__sec_att_train, self.__sec_att_test, self.__sec_lab_train, self.__sec_lab_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle =True, stratify=y)
        # print(self.__sec_att_train)
        # print(self.__sec_att_test)
        # print(self.__sec_lab_test)
        # print(self.__sec_lab_train)
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

        sm = SMOTE(random_state=6)
        X_train_res, y_train_res = sm.fit_resample(self.__attributes_train, self.__labels_train)
        self.__attributes_train, self.__labels_train = pd.DataFrame(X_train_res, columns=name_train), pd.Series(y_train_res)

        print("[respamling finished]")

    def glue_pred_to_att(self, attWithoutLabel, virtualLabel):
        '''
        Merge the predicted y labels from the evaluation and merge it into the test attributes

        :param: attWithoutLabel:    attribute values with no labels (Dataframe)
                virtualLabel:   label values predicted by first model   (Series)
        :return: attributesWithVirtualLabel (attributes with predicted labels)
        '''
        attributesWithVirtualLabel = pd.concat([attWithoutLabel, virtualLabel], axis=1, join='inner')
        return attributesWithVirtualLabel

    def merge_dataframes(self, df1, df2, df3):
        '''
        Merge all the columns that were previously plitted to obtain more specific labels

        :param: df1: left data columns to be merged.    (Dataframe)
                df2: middle data columns to be merged.   (Dataframe)
                df3: right data columns to be merged.  (Datafrmae)
        :return: mergedFrame: Merged data columns. (Data Frame)
        '''
        frames = [df1, df2, df3]
        mergedFrame = pd.concat(frames)

        return mergedFrame.sort_index()

    def get_second_data(self,n):
        '''
        Split the data into train and test data for second classifier
        :parameter: Label number
        :return : tuple of label n data, splited into test, train data
        '''
        dfTrain = self.__loanData
        out = ()
        dfTrain = dfTrain.drop(dfTrain[dfTrain.loan_status < n].index)
        dfTrain = dfTrain.drop(dfTrain[dfTrain.loan_status > n].index)
        y = dfTrain['loan_status']
        X = dfTrain.drop('loan_status', axis=1)
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle =True, stratify=y)
        out = (X_test, X_train, y_test, y_train)
        # print(X_test)
        # print(X_train)
        # print(y_test)
        # print(y_train)
        print("[split_data finished]")
        return out
