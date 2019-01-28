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
        dfTrain = self.__loanData
        dfTrain = dfTrain.drop(dfTrain[dfTrain.loan_status < 3].index)
        dfTrain = dfTrain.drop(dfTrain[dfTrain.loan_status > 5].index)
        y = dfTrain['loan_status']
        X = dfTrain.drop(['loan_status','new_loan_status'], axis=1)
        self.__sec_att_train ,self.__sec_att_test, self.__sec_lab_train,self.__sec_lab_test = train_test_split(X, y, test_size=0.2, random_state = 1, shuffle =True, stratify=y)
        print(self.__sec_lab_test)
        print(self.__sec_lab_train)
        self.__resample_data_SMOTE()

        print("[split_data finished]")
        return (self.__sec_att_test, self.__sec_lab_test,self.__sec_att_train,self.__sec_lab_train)

    def finalize_label(self, y_first, y_second):
        '''
        Convert the virtual label into the real label.

        :param: None
        :return: None
        '''

        finalLabel = y_first
        j=0

        for i in range(len(finalLabel)):
            if finalLabel.iloc[i] == 3:
                finalLabel.iloc[i] = y_second.iloc[j]
                j+=1

        return finalLabel

    def classify_label(self, loanData):
        '''
        converts payment time into labels according to the distribution of payment time
        nodes classifying the label is defined by the quantile values of the distribution
        :param = list/dataframe of the payment time, loanData
        :return = loanData with a new column of label
        '''
        tmp = 0.1
        nodes = dict() 
        for i in range(0,10):
            nodes[i] = loanData['payment_time'].quantile(tmp)
            tmp += 0.1
        print(nodes)
        loanData['payment_label'] = loanData['payment_time'].apply(self.classify, args=(nodes,))
    # def classify(self,val,nodes):
    #     tmp = 0
    #     tmp2 = 0
    #     list_class = range(len(nodes)+ 1)
    #     for node in nodes:
    #         if(val > node):
    #             tmp = 1
    #         elif(tmp == 1 and val < node):
    #             return list_class[]
    #         else:
    #             tmp = 0
    #         tmp2 += 1