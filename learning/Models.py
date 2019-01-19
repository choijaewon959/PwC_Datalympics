'''
Model object that contains all the possible classification models
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from data.Preprocessor import Preprocessor

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class Models:
    def __init__(self):
        self.__algorithms = set() # list containing all the algorithms (str)
        self.__processor = Preprocessor() # processor managing data


    def binary_logistic_regression(self):
        """

        """
        # TODO: code by taemin
        print("Hello I'm binary logistic regression")

        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        print(trainAttributes.shape)
        print(trainLabels.shape)
        print(testAttributes.shape)
        print(testLabels.shape)

        eval_set=[(testAttributes, testLabels)]

        clf = xgboost.sklearn.XGBClassifier(
            objective="multi:softprob",
            learning_rate=0.05,
            seed=9616, #seed that is not random
            max_depth=20,
            gamma=10,
            n_estimators=500)

        clf.fit(trainAttributes, trainLabels, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True)

        y_pred = clf.predict(testAttributes)

        accuracy = accuracy_score(np.array(testLabels).flatten(), y_pred)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))

        accuracy_per_roc_auc = roc_auc_score(np.array(testLabels).flatten(), y_pred)
        print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))

        print("Hello I'm binary logistic regression")


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
