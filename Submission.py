'''
Class only used for the making the submission csv file.
'''
import numpy as np 
import pandas as pd 
import csv

class Submission:
    def __init__(self):
        self.__row = ['PwC_RowID','PredictedPaymentDate','PaymentTiming']
        self.__dataFrame = pd.DataFrame(columns=self.__row)
        
    def update_PwC_RowID(self, values):
        '''
        Update the values of PwC rowId column in the dataframe.

        :param: values: row containing values of the payment.
        :return None
        '''
        for i in range(len(values)):
            self.__dataFrame[i]['PwC_RowID'] = i+1
        
    def update_paymentTiming(self, values):
        '''
        Update the values of payment timing column in the dataframe.

        :param: values: predicted payment timing values.(early, on time, late)
        :return None
        '''
        self.__dataFrame['PaymentTiming'] = values

    def update_predictedPaymentDate(self, values):
        '''
        Update the values of predictedPaymentDate column in the dataframe.

        :param: values: predicted payment date values.  (str)
        :return: None
        '''
        self.__dataFrame['PredictedPaymentDate'] = values

    def make_csv(self):
        '''
        Convert the dataframe into the csv file to be submitted.

        :param: None
        :return: None
        '''
        self.__dataFrame.to_csv('HKU_KD_result.csv')
