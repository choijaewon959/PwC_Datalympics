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

from keras.models import Sequential
from keras.layers import Dense
from num_node import * 

class Models:
    def __init__(self):
        self.__algorithms = set() # list containing all the algorithms (str)
        self.__processor = Preprocessor() # processor managing data


    def binary_logistic_regression(self):
        """

        """
        # TODO: code by taemin

        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()


        eval_set=[(testAttributes, testLabels)]

        clf = xgboost.sklearn.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.05,
            seed=9616, #seed that is not random
            max_depth=20,
            gamma=10,
            n_estimators=500)

        clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=True)

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
    def ff_network(self, n): 
        '''
        fowrad feeding neural network with one hidden layer 
        Kernel: 
        '''
        if(n == 1):
            X = self.__processor.get_train_attributes() 
            Y = self.__processor.get_train_labels()
            in_len = len(self.__processor.get_train_attributes())
            out_len = len(self.__processor.get_train_labels())
            model = Sequential()
            model.add(Dense(num_hidden_layer1(in_len,out_len,len(Y)), input_dim=in_len, activation='relu'))
            model.add(Dense(out_len, activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
            scores = model.evaluate(self.__processor.get_test_attributes(),self.__processor.get_test_labels())        
            print('Test Data Accuracy',scores[1])        
        elif(n==2):
            X = self.__processor.get_train_attributes() 
            Y = self.__processor.get_train_labels()
            in_len = len(self.__processor.get_train_attributes(self))
            out_len = len(self.__processor.get_train_labels(self))
            model = Sequential()
            model.add(Dense(Dense(num_hidden_layer2(in_len,out_len), input_dim=in_len, activation='relu'))) 
            model.add(Dense(out_len, activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            scores = model.evaluate(self.__processor.get_test_attributes(),self.__processor.get_test_labels())        scores = model.evaluate(X, Y)
            print('Test Data Accuracy',scores[1])
        elif(n==3):
            X = self.__processor.get_train_attributes() 
            Y = self.__processor.get_train_labels()
            in_len = len(self.__processor.get_train_attributes(self))
            out_len = len(self.__processor.get_train_labels(self))
            model = Sequential()
            model.add(Dense(num_hidden_layer3(in_len,out_len,len(Y))[0], input_dim=in_len, activation='relu')) 
            model.add(Dense(num_hidden_layer3(in_len,out_len,len(Y))[1], activation='relu'))
            model.add(Dense(out_len, activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
            scores = model.evaluate(self.__processor.get_test_attributes(),self.__processor.get_test_labels())
            print('Test Data Accuracy',scores[1])              
    