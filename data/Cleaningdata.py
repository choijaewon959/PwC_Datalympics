import pandas as pd
import numpy as np

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

    dfTest= dfTest[['PwC_RowID', 'BusinessTransaction', 'CompanyCode', 'CompanyName',
   'DocumentNo', 'DocumentType', 'DocumentTypeDesc', 'EntryDate',
   'EntryTime', 'InvoiceAmount', 'InvoiceDate', 'InvoiceDesc',
   'InvoiceItemDesc', 'LocalCurrency', 'PaymentDate', 'PaymentDocumentNo',
   'Period', 'PO_FLag', 'PO_PurchasingDocumentNumber', 'PostingDate',
   'PurchasingDocumentDate', 'ReferenceDocumentNo', 'ReportingAmount',
   'TransactionCode', 'TransactionCodeDesc', 'UserName', 'VendorName',
   'VendorCountry', 'Year', 'PaymentDueDate', 'difference', 'label','duration']]

    mapping = {'BusinessTransaction': {'Business transaction type 0002': 2 , 'Business transaction type 0003': 3, 'Business transaction type 0001': 1},
    'CompanyCode' : {'C002':2, 'C001':1, 'C003':3},
    'DocumentType': {'T03':3,'T04':4,'T02':2,'T01':1,'T09':9,'T07':7,'T06':6,'T08':8, 'T05':5},
    'DocumentTypeDesc': {'Vendor invoice': 0, 'Invoice receipt':1,'Vendor credit memo':2,'Vendor document':3,'TOMS (Jul2003)/ TWMS':4 ,'Interf.with SMIS-CrM':5,'Interf.with SMIS-IV':6 ,'Interface with PIMS':7},
    'PO_FLag': {'N': 0 , 'Y':1},
    'TransactionCode': {'TR 0005':0,'TR 0006':1,'TR 0002':2,'TR 0008':3,'TR 0007':4,'TR 0003':5,'TR 0004':6, 'TR 0001':7},
    }

    dropcol  = ['CompanyName', 'EntryDate', 'DocumentTypeDesc', 'EntryTime',
            'InvoiceDate', 'LocalCurrency','PwC_RowID','BusinessTransaction',
            'PO_PurchasingDocumentNumber', 'PostingDate', 'PurchasingDocumentDate',
            'ReportingAmount', 'Year', 'PaymentDate', 'PaymentDueDate','DocumentType','difference'
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
   'InvoiceAmount', 'PO_FLag', 'TransactionCode', 'TransactionCodeDesc', 'UserName','InvoiceDesc',
   'label','duration','Period']

    for col in cols:
        print('Imputation with Median: %s' % (col))
        dfTest[col].fillna(dfTest[col].median(), inplace=True)

    for name in featurelist:
        if name in list(dfTest['VendorName'].unique()):
            dfTest[name] = dfTest['VendorName'].apply(vendor_apply, args=(name,))
        elif "Vendor " in name and name not in list(data['VendorName'].unique()):
            dfTest[name] = dfTest['VendorName'].apply(vendor_apply, args=(name,))

    dfTest=dfTest.drop('VendorName', axis=1)
    dfTest=dfTest.drop('VendorCountry', axis=1)
    dfTest['InvoiceAmount'].fillna(dfTest['InvoiceAmount'].mean(), inplace=True)
    # for i in dfTest.columns.values:
    #print(dfTest['InvoiceAmount'].unique())


    # dfTest['InvoiceAmount'] = pd.to_numeric(dfTest['InvoiceAmount'], downcast="integer", errors="raise")
    #dfTest['duration'] = pd.to_numeric(dfTest['duration'], downcast="integer")

    # print(dfTest)
    # print(dfTest.columns)
    #print(dfTest.dtypes)
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
