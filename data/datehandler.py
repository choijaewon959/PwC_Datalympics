import pandas as pd
import numpy as np
import time
import math
import datetime


<<<<<<< HEAD
def datetime_data(data):
    """
    parse input data(datetime format)
    into datetimeIndex dataframes

    calculate late days or early days if needed (on time)

    :param: dataset (pandas dataframe)
    :return: dataset (pandas dataframe) column added
    """
=======
def retrieve_data(data):

>>>>>>> 083eed2b8d1d602e06d13dfa5e53c7f6f85c6c34
    # data = pd.read_csv("../paymentdata.csv")

    print("[retrieve_data finished]")

    #process each column into datetimeIndex dataframe
    data['effective_date']=pd.to_datetime(data['effective_date'])
    data['due_date']= pd.to_datetime(data['due_date'])
    data['paid_off_time']= pd.to_datetime(data['paid_off_time'])

    #normalize each date time into same format
    data['paid_off_time']=data['paid_off_time'].dt.strftime('%Y-%m-%d')
    data['paid_off_time']=pd.to_datetime(data['paid_off_time'])

    #make new column with result which keep track of dates belated or early
    data['result']=data['due_date']-data['paid_off_time']
    #format into integer format
    data['result']=data['result'].dt.days

    #return those with negative days (days passed due date)
    lateday = data.loc[data['result'] < 0]
    #not returned yet


    return data

#data,lateday = retrieve_data()
#data=comparison_check(data)
