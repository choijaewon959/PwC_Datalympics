import pandas as pd
import numpy as np
import datetime


def change(val):
    return int(val[-2:])
def change2(val):
    return int(val[-1:])
def change3(val):
    return int(val[-6:])
def change4(val):
    return int(val[-3:])
def vendor_apply(val,name):
    if(val == name):
        return 1
    return 0

def clean_data(data, featurelist):
    '''
    temporary data processor for loan.csv file
    erase unrelated columns and imputation is done.
    prints some debugging messages.

    :param: DataFrame
    :return: DataFrame
    '''
    print("Cleaning_data running...")
    dfTest= data
    temp=['PwC_RowID', 'BusinessTransaction', 'CompanyCode', 'CompanyName',
   'DocumentNo', 'DocumentType', 'DocumentTypeDesc', 'EntryDate',
   'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
   'InvoiceItemDesc', 'LocalCurrency', 'PaymentDate', 'PaymentDocumentNo',
   'Period', 'PO_FLag', 'PO_PurchasingDocumentNumber', 'PostingDate',
   'PurchasingDocumentDate', 'ReferenceDocumentNo', 'ReportingAmount',
   'TransactionCode', 'TransactionCodeDesc', 'UserName', 'VendorName',
   'VendorCountry', 'Year', 'PaymentDueDate', 'difference', 'label','duration']

    dfTest= dfTest[['PwC_RowID', 'BusinessTransaction', 'CompanyCode', 'CompanyName',
   'DocumentNo', 'DocumentType', 'DocumentTypeDesc', 'EntryDate',
   'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
   'InvoiceItemDesc', 'LocalCurrency', 'PaymentDate', 'PaymentDocumentNo',
   'Period', 'PO_FLag', 'PO_PurchasingDocumentNumber', 'PostingDate',
   'PurchasingDocumentDate', 'ReferenceDocumentNo', 'ReportingAmount',
   'TransactionCode', 'TransactionCodeDesc', 'UserName', 'VendorName',
   'VendorCountry', 'Year', 'PaymentDueDate', 'difference', 'label','duration']]

    mapping = {'BusinessTransaction': {'Business transaction type 0001': 1,'Business transaction type 0002': 2 , 'Business transaction type 0003': 3},
    'CompanyCode' : {'C002':2, 'C001':1, 'C003':3},
    'DocumentType': {'T03':3,'T04':4,'T02':2,'T01':1,'T09':9,'T07':7,'T06':6,'T08':8, 'T05':5},
    'DocumentTypeDesc': {'Vendor invoice': 0, 'Invoice receipt':1,'Vendor credit memo':2,'Vendor document':3,'TOMS (Jul2003)/ TWMS':4 ,'Interf.with SMIS-CrM':5,'Interf.with SMIS-IV':6 ,'Interface with PIMS':7},
    'PO_FLag': {'N': 0 , 'Y':1},
    'TransactionCode': {'TR 0005':0,'TR 0006':1,'TR 0002':2,'TR 0008':3,'TR 0007':4,'TR 0003':5,'TR 0004':6, 'TR 0001':7},
    }

    dropcol  = ['CompanyName', 'EntryDate', 'DocumentTypeDesc', 'EntryTime',
            'InvoiceDate', 'LocalCurrency','PwC_RowID',
            'PO_PurchasingDocumentNumber', 'PostingDate', 'PurchasingDocumentDate',
            'ReportingAmount', 'Year', 'PaymentDate', 'PaymentDueDate','TransactionCodeDesc','difference'
            ]


    dfTest['UserName'] = dfTest['UserName'].apply(change)
    dfTest['TransactionCodeDesc'] = dfTest['TransactionCodeDesc'].apply(change2)
    dfTest['ReferenceDocumentNo'] = dfTest['ReferenceDocumentNo'].apply(change3)
    dfTest['DocumentNo'] = dfTest['DocumentNo'].apply(change3)
    dfTest['PaymentDocumentNo'] = dfTest['PaymentDocumentNo'].apply(change3)
    dfTest['InvoiceItemDesc'] = dfTest['InvoiceItemDesc'].apply(change3)
    dfTest['InvoiceDesc'] = dfTest['InvoiceDesc'].apply(change3)

    #self.__rowid= dfTest['PwC_RowID']

    dfTest = dfTest.replace(mapping)
    dfTest = dfTest.drop(dropcol, axis=1)

    cols = [ 'CompanyCode','DocumentNo','PaymentDocumentNo','InvoiceItemDesc','ReferenceDocumentNo',
   'InvoiceAmount', 'PO_FLag', 'TransactionCode', 'UserName','InvoiceDesc',
   'label','duration','Period']

    for col in cols:
        print('Imputation with Median: %s' % (col))
        dfTest[col].fillna(dfTest[col].median(), inplace=True)

    for name in featurelist:
        if name in list(dfTest['VendorName'].unique()):
            dfTest[name] = dfTest['VendorName'].apply(vendor_apply, args=(name,))
        elif "Vendor " in name and name not in list(data['VendorName'].unique()):
            dfTest[name] = dfTest['VendorName'].apply(vendor_apply, args=(name,))

    for name in featurelist:
        if name in list(dfTest['VendorCountry'].unique()):
            dfTest[name] = dfTest['VendorCountry'].apply(vendor_apply, args=(name,))
        elif name not in temp and name not in list(data['VendorCountry'].unique()):
            dfTest[name] = dfTest['VendorCountry'].apply(vendor_apply, args=(name,))

    dfTest=dfTest.drop('VendorName', axis=1)
    dfTest=dfTest.drop('VendorCountry', axis=1)
    dfTest['InvoiceAmount'].fillna(dfTest['InvoiceAmount'].mean(), inplace=True)
    # for i in dfTest.columns.values:
    #print(dfTest['InvoiceAmount'].unique())


    # dfTest['InvoiceAmount'] = pd.to_numeric(dfTest['InvoiceAmount'], downcast="integer", errors="raise")
    #dfTest['duration'] = pd.to_numeric(dfTest['duration'], downcast="integer")

    # print(dfTest)
    # print(dfTest.columns)
    print(dfTest.dtypes)
    #print(dfTest.drop('label', axis=1) , dfTest['label'])
    return (dfTest.drop('label', axis=1) , dfTest['label'])


def get_specific_label(data, labelnum):
    '''
    Receive data with labels with predicted

    :parameter: DataFrame: data, integer: labelnum
    :return : dataset with labels with specific numbering
    '''
    #print(data)
    dfTest = data
    dfTest = dfTest.drop(dfTest[dfTest.label < labelnum].index)
    dfTest = dfTest.drop(dfTest[dfTest.label > labelnum].index)
    #print("get_specific_label : " , dfTest)
    return dfTest
def finalize_label(y_first, y_early, y_late):
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
                finalLabel.iloc[i] = y_late.iloc[k]
                k+=1
        return finalLabel
def convert_back(val,early_nodes,late_nodes):
    '''
    function converting payment_label to integer which estimates how many days late/early
    '''
    map = {1:0, 19:early_nodes[8], 18:early_nodes[7], 17:early_nodes[6],16:early_nodes[5],
            15:early_nodes[4],14:early_nodes[3],13:early_nodes[2],12:early_nodes[1],11:early_nodes[0],10: early_nodes[0],
            40 : late_nodes[9] , 41 : late_nodes[8], 42: late_nodes[7] ,43: late_nodes[6],44: late_nodes[5],45: late_nodes[4],
            46 : late_nodes[3], 47: late_nodes[2], 48: late_nodes[1], 49 : late_nodes[0], 50: late_nodes[0]
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

def regression2label(val):
    if(val > 0):
        return 'Early'
    elif(val == 0):
        return 'On time'
    elif(val < 0):
        return 'Late'

def regression_final_result(data,y_predicted):
    '''
    conversion of regression result to predicted date
    :param - data including duedate, regression result
    :return - data appended with predicted date
    '''
    y_predicted = y_predicted.round()
    data['predicted_date'] = data['PaymentDueDate'] - y_predicted.map(pd.offsets.Day) 
    data['label'] = y_predicted.apply(regression2label)

def final_result(data, y_predicted,early_nodes,late_nodes):
    '''
    :param - Dataframe type column vector y_predicted 
    :return - transaction data with column 'predicted date' appended
    '''
    data['predicted_date'] = data['PaymentDueDate'] - y_predicted.apply(convert_back,args=(early_nodes,late_nodes,)).map(pd.offsets.Day) 
    data['label'] = data['label'].apply(label2result)

def result2csv(data):
    '''
    save result in format of PwC's format
    '''
    data[['PwC_RowID','predicted_date','label']].to_csv('HKU_KD_result.csv', index=False)

def csv2score(result_dir,data):
    '''
    evaluate score of result
    :param - dir to result csv file, dataframe column vector including true value of payment date
    :return - score
    '''
    result = pd.read_csv(result_dir)
    result['predicted_date'] = pd.to_datetime(result['predicted_date'])
    data['PaymentDate'] = pd.to_datetime(data['PaymentDate'])

    data['error'] = data['PaymentDate'] - result['predicted_date']
    print(data['error'])
    data['error'] = data['error'].astype('timedelta64[h]').abs()
    print(data['error'])
    error = data['error'].sum()
    print(error)
    return error/24

def generate_node(dfTrain):
    tmp = 0.1
    early_nodes = dict()
    late_nodes = dict()
    for i in range(0,10):
        early_nodes[i] = dfTrain.drop(dfTrain[dfTrain.difference < 1].index)['difference'].quantile(tmp)
        late_nodes[i] = dfTrain.drop(dfTrain[dfTrain.difference > -1].index)['difference'].quantile(tmp)
        tmp += 0.1
    return (early_nodes, late_nodes)
