'''
Model object that contains all the possible classification models
'''
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class Models:
    def __init__(self, dataFrame):
        self.__algorithms = set() # list containing all the algorithms (str)
        self.__data = dataFrame


    def binary_logistic_regression(self, targetY):
        #targetY is the column name @param string
        # TODO: code by taemin

        dfTrain, dfTest = train_test_split(self.__data ,test_size=0.2)

        #test_member_id = pd.DataFrame(dfTest['member_id'])
        train_target = pd.DataFrame(dfTrain[targetY])

        X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), test_size=0.20)
        eval_set=[(X_test, y_test)]

        clf = xgboost.sklearn.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.05,
            seed=9616, #seed that is not random
            max_depth=20,
            gamma=10,
            n_estimators=500)

        clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True)
        #stops training if 10 rounds of estimation is same
        """
        print(datetime.now()-st)
        #calculate time of training
        """

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))

        accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
        print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))
