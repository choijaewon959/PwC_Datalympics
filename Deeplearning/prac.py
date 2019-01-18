"""
following lines of code was run in a
jupyter notebook

for practice matter

The focus was on binary:logistic training model
for xgboost sklearn model


"""

import pandas as pd
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


#print('Reading data...')
dfTrain = pd.read_csv('./loan.csv', low_memory=False)

#dfTrain.columns

dfTrain= dfTrain[['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
       'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'issue_d', 'loan_status']]

#orgdata = dfTrain.copy()

'''
Data transformation/ data selection
'''

#term into numeric terms
dfTrain['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
dfTrain['term']= pd.to_numeric(dfTrain['term'], errors='coerce')

#erase unrelated columns
dfTrain= dfTrain[['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
       'term', 'int_rate', 'installment', 'sub_grade',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status']]


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

dfTrain['annual_inc']= pd.to_numeric(dfTrain['annual_inc'], errors='coerce')

#print('Transform: loan_status...')
# for loan status just gave random 0 / 1 of binary representation of good or bad loan
dfTrain['loan_status'].replace('n/a', '0', inplace=True)
dfTrain['loan_status'].replace(to_replace='Fully Paid', value='0', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Current', value='0', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Charged Off', value='1', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='In Grace Period', value='1', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Late (31-120 days)', value='1', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Late (16-30 days)', value='1', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Issued', value='0', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Default', value='0', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Does not meet the credit policy. Status:Fully Paid Off', value='0', regex=True, inplace=True)
dfTrain['loan_status'].replace(to_replace='Does not meet the credit policy. Status:Charged Off', value='1', regex=True, inplace=True)
dfTrain['loan_status'] = pd.to_numeric(dfTrain['loan_status'], errors='coerce')

#dropped few more column to make it simpler
dfTrain= dfTrain.drop('verification_status', axis=1)
dfTrain=dfTrain.drop('home_ownership', axis=1)

#data imputation
cols = ['term', 'loan_amnt', 'funded_amnt', 'int_rate', 'sub_grade', 'annual_inc', 'emp_length', 'installment']
for col in cols:
    print('Imputation with Median: %s' % (col))
    dfTrain[col].fillna(dfTrain[col].median(), inplace=True)

cols=['member_id', 'loan_status']
for col in cols:
    print('Imputation with Zero: %s' % (col))
    dfTrain[col].fillna(0, inplace=True)
print('Missing value imputation done.')

#dropped funded_amnt_inv because I don't know what that term means
dfTrain = dfTrain.drop('funded_amnt_inv', axis=1)

#made member_id only identification key
dfTrain = dfTrain.drop('id', axis=1)

#separated test and train set by 2 : 8
dfTrain, dfTest = train_test_split(dfTrain,test_size=0.2)


# Separating the member_id column of test dataframe
test_member_id = pd.DataFrame(dfTest['member_id'])

train_target = pd.DataFrame(dfTrain['loan_status'])


##some feature engineering a this point has to be done.
# the data can be redundent, or dependent, insufficient in some ways raw data
#also the numerics has to be normalized???? into same ratio

selected_cols=[
    'member_id', 'emp_length', 'loan_amnt', 'funded_amnt', 'sub_grade', 'int_rate', 'annual_inc', 'term'
]

finalTrain=dfTrain[selected_cols]
finalTest=dfTest[selected_cols]


X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), test_size=0.20)
eval_set=[(X_test, y_test)]

from datetime import datetime

st = datetime.now()

clf = xgboost.sklearn.XGBClassifier(
    objective="binary:logistic",
    learning_rate=0.05,
    seed=9616, #seed that is not random
    max_depth=20,
    gamma=10,
    n_estimators=500)

clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=True)
#stops training if 20 rounds of estimation is same

print(datetime.now()-st)
#calculate time of training

y_pred = clf.predict(X_test)

accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
print("Accuracy: %.10f%%" % (accuracy * 100.0))

accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))

'''
final_pred = pd.DataFrame(clf.predict_proba(np.array(finalTest)))
dfSub = pd.concat([test_member_id, final_pred.ix[:, 1:2]], axis=1)
dfSub.rename(columns={1:'loan_status'}, inplace=True)
dfSub.to_csv((('%s.csv') % (submission_file_name)), index=False)
'''

import matplotlib.pyplot as plt
print(clf.feature_importances_)
idx = 0
for x in list(finalTrain):
    print('%d %s' % (idx, x))
    idx = idx + 1
plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.show()
