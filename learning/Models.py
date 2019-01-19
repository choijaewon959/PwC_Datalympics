'''
Model object that contains all the possible classification models
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm import SVC
import sklearn.metrics import clasification_report, confusion_matrix
from data.Preprocessor import Preprocessor

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class Models:
    def __init__(self, dataFrame):
        self.__algorithms = set() # list containing all the algorithms (str)
        self.__data = dataFrame
        self.__processor = Preprocessor(dataFrame) # processor managing data


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

        
    def linear_SVM(self):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: linear

        :param: None
        :return None
        '''
        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        #train svm model
        svclassifier = SVC(kernel = 'linear')
        svclassifier.fit(trainAttributes, trainLabels)

        #make prediction
        label_prediction = svclassifier.predict(testAttributes)

        #evaluation
        print(confusion_matrix(testLabels, label_prediction))
        print(clasification_report(testLabels, label_prediction))

    def gaussian_SVM(self):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: gaussian

        :param: None
        :return: None
        '''
        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        #train svm model
        svclassifier = SVC(kernel = 'rbf')
        svclassifier.fit(trainAttributes, trainLabels)

        #make prediction
        label_prediction = svclassifier.predict(testAttributes)

        #evaluation
        print(confusion_matrix(testLabels, label_prediction))
        print(clasification_report(testLabels, label_prediction))
