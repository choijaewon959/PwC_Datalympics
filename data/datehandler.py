import pandas as pd
import numpy as np
import time
import math
import datetime


def datetime_data(data):
    """
    parse input data(datetime format)
    into datetimeIndex dataframes

    calculate late days or early days if needed (on time)

    :param: dataset (pandas dataframe)
    :return: dataset (pandas dataframe) column added
    """
    # data = pd.read_csv("../paymentdata.csv"

    #process each column into datetimeIndex dataframe
    data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])
    data['PaymentDate']= pd.to_datetime(data['PaymentDate'])
    data['EntryDate']= pd.to_datetime(data['EntryDate'])
    data['PaymentDueDate']= pd.to_datetime(data['PaymentDueDate'])

    #normalize each date time into same format
    #data['paid_off_time']=data['paid_off_time'].dt.strftime('%Y-%m-%d')
    #data['paid_off_time']=pd.to_datetime(data['paid_off_time'])

    #make new column with result which keep track of dates belated or early
    data['difference']=data['PaymentDueDate']-data['PaymentDate']
    #format into integer format
    data['difference']=data['difference'].dt.days

    data['label'] = data['difference'].apply(add_label)

    print(data['difference'].value_counts())
    #return those with negative days (days passed due date)
    #lateday = data.loc[data['result'] < 0]

    # print(data['difference'])
    # print(data['label'])

    #print(data)

    """
    return data with difference/ label with it
    """
    #print(data['diff'].value_counts())
    print("[datetime conversion finished]")

    return data

def add_label(val):
    if val > 0 :
        return 0
    elif val < 0:
        return 2
    else:
        return 1

#data,lateday = retrieve_data()
#data=comparison_check(data)
