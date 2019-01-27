import pandas as pd

def clean_data(data):
    '''
    temporary data processor for loan.csv file
    erase unrelated columns and imputation is done.
    prints some debugging messages.

    :param: DataFrame
    :return: DataFrame
    '''
    print("Cleaning_data running...")

    dfTrain = data
    #copied data to refrain from warnings
    dfTrain= dfTrain.copy()

    #erase unrelated columns
    dfTrain= dfTrain[['loan_amnt', 'funded_amnt',
           'term', 'int_rate', 'installment', 'sub_grade',
           'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths'
           ,'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec'
           ,'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt'
           ,'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'
           ,'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt'
        ]]

    # TODO: Feature transformation can be done beforehand or after
    # when the data is normalized to numerical data, these steps should be omitted.
    dfTrain['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
    dfTrain['term'] = dfTrain['term'].astype(int)

    #print('Transform: sub_grade...')
    dfTrain['sub_grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
    dfTrain['sub_grade'].replace(to_replace='B', value='1', regex=True, inplace=True)
    dfTrain['sub_grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
    dfTrain['sub_grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
    dfTrain['sub_grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
    dfTrain['sub_grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
    dfTrain['sub_grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
    dfTrain['sub_grade'] = pd.to_numeric(dfTrain['sub_grade'], errors='coerce')

    #print('Transform: emp_length...')
    dfTrain['emp_length'].replace('n/a', '0', inplace=True)
    dfTrain['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
    dfTrain['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
    dfTrain['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
    dfTrain['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
    dfTrain['emp_length'] = pd.to_numeric(dfTrain['emp_length'], errors='coerce')

    #print('Transform: annual_inc...')
    dfTrain['annual_inc']= pd.to_numeric(dfTrain['annual_inc'], errors='coerce')


    '''
    #data imputation
    '''
    cols = ['loan_amnt', 'funded_amnt',
           'term', 'int_rate', 'installment', 'sub_grade',
           'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths'
           ,'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec'
           ,'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt'
           ,'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'
           ,'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt'
        ]

    for col in cols:
        #print('Imputation with Median: %s' % (col))
        if col in dfTrain.columns:
            dfTrain[col].fillna(dfTrain[col].median(), inplace=True)

    print("cleaning finished")
    return dfTrain


def get_specific_label(data, labelnum):
    '''
    Receive data with labels with predicted

    :parameter: DataFrame: data, integer: labelnum
    :return : dataset with labels with specific numbering
    '''

    dfTest = data
    dfTest = dfTest.drop(dfTest[dfTest.loan_status < labelnum].index)
    dfTest = dfTest.drop(dfTest[dfTest.loan_status > labelnum].index)

    return dfTest
