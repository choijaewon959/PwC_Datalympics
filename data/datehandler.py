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
    data['duration'] = data['PaymentDueDate']-data['InvoiceDate']

    #format into integer format
    data['difference']=data['difference'].dt.days
    data['duration']= data['duration'].dt.days

    data['label'] = data['difference'].apply(add_label)

    #print(data['label'].value_counts())
    #print(data['difference'].value_counts())
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
def convert_back(val,early_nodes,late_nodes):
    '''
    function converting payment_label to integer which estimates how many days late/early
    '''
    early_nodes = [round(i) for i in early_nodes]
    late_nodes = [round(i) for i in late_nodes]
    map = {1:0, 19:early_nodes[8], 18:early_nodes[7], 17:early_nodes[6],16:early_nodes[5],
            15:early_nodes[4],14:early_nodes[3],13:early_nodes[2],12:early_nodes[1],11:early_nodes[0],10: int(early_nodes[0]/2),
            40 : late_nodes[9] , 41 : late_nodes[8], 42: late_nodes[7] ,43: late_nodes[6],44: late_nodes[5],45: late_nodes[4],
            46 : late_nodes[3], 47: late_nodes[2], 48: late_nodes[1], 49 : late_nodes[0], 50: int(late_nodes[0]/2)
    }
    return map[val]
def label2result(val):
    '''
    function converting column label into string indicating early/on time/ late
    '''
    if(val == 0):
        return 'Early'
    elif(val == 1):
        return 'On time'
    if(val == 2):
        return 'Late'
    
def final_result(data, y_predicted):
    '''
    :param - Dataframe type column vector y_predicted 
    :return - transaction data with column 'predicted date' appended
    '''
    data['predicted_date'] = data['PaymentDueDate'] + y_predicted.apply(convert_back)
    data['label'] = data['label'].apply(label2result)
def result2csv(data):
    '''
    save result in format of PwC's format
    '''
    data[['PwC_RowID','label','predicted_date']].to_csv('HKU_KD_result.csv')
def csv2score(result_dir,answer):
    '''
    evaluate score of result
    :param - dir to result csv file, dataframe column vector including true value of payment date
    :return - score
    '''
    result = pd.read_csv(result_dir)
    result['predicted_date'] = pd.to_datetime(result['predicted_date'])
    answer['payment_date'] = pd.to_datetime(answer['payment_date'])

    answer['error'] = answer['payment_date'] - result['predicted_date']
    answer['error'] = answer['error'].abs()
    error = answer['error'].sum()
    print(error)
    return score