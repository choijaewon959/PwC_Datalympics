'''
Model object that contains all the possible classification models
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix
from data.Preprocessor import Preprocessor

import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# from keras.models import Sequential
# from keras.layers import Dense
# from num_node import *

class Models:
    def __init__(self):
        self.__algorithms = set() # list containing all the algorithms (str)
        self.__processor = Preprocessor() # processor managing data
        print("model made")

    def k_neighbor(self):
        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        knn = KNeighborsClassifier(n_neighbors=10)

        knn.fit(trainAttributes,trainLabels)

        y_pred = knn.predict(testAttributes)

        print("Accuracy:",accuracy_score(testLabels, y_pred))

    def decision_tree(self):
        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        clf = DecisionTreeClassifier()

        clf.fit(trainAttributes,trainLabels)

        y_pred = clf.predict(testAttributes)
        print("Accuracy:",accuracy_score(testLabels, y_pred))


    def random_forest(self):

        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()
        print(trainLabels)
        rfc = RandomForestClassifier(n_estimators=30)
        rfc.fit(trainAttributes, trainLabels)
        preds = rfc.predict(testAttributes)
        acc_rfc = (preds == testLabels).sum().astype(float) / len(preds)*100
        print("Scikit-Learn's Random Forest Classifier's prediction accuracy is: %3.2f" % (acc_rfc))

    def XGBClassifier(self):

        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        print(trainAttributes.head())
        print(trainLabels.head())
        print(testAttributes.shape)
        print(testLabels.shape)

        eval_set=[(testAttributes, testLabels)]

        clf = xgboost.sklearn.XGBClassifier(
            #objective="multi:softprob",
            learning_rate=0.05,
            seed=0, #seed that is not random
            max_depth=4,
            min_child_weight=1,
            reg_alpha=0.005,
            gamma=0,
            n_estimators=200, subsample=0.8, colsample_bytree=0.8)

        clf.fit(trainAttributes, trainLabels, early_stopping_rounds=20,  eval_set=eval_set, verbose=True)

        y_pred = clf.predict(testAttributes)

        acc_xgb = (y_pred == testLabels).sum().astype(float) / len(y_pred)*100
        print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))

        accuracy = accuracy_score(np.array(testLabels).flatten(), y_pred)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))

        #accuracy_per_roc_auc = roc_auc_score(np.array(testLabels).flatten(), y_pred)
        #print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))

        #print("Hello I'm binary logistic regression")

    def linear_SVM(self):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: linear

        :param: None
        :return None
        '''

        print("training...")

        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        #train svm model
        print("Learning...")
        svclassifier = SVC(kernel = 'linear')
        svclassifier.fit(trainAttributes, trainLabels)

        #make prediction
        label_prediction = svclassifier.predict(testAttributes)

        #evaluation
        print("Accuracy: ", accuracy_score(testLabels, label_prediction))

    def gaussian_SVM(self, trainData):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: gaussian

        :param: None
        :return: None
        '''
        trainAttributes = trainData
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        #train svm model
        print("Learning...")
        svclassifier = SVC(
            C=1.0,
            cache_size=700,
            class_weight=None,
            coef0=0.0,
            decision_function_shape='ovo',
            degree=3,
            gamma='scale',
            kernel='rbf',
            max_iter=-1,
            probability=False,
            random_state=None,
            shrinking=True,
            tol=0.001,
            verbose=False
        )
        svclassifier.fit(trainAttributes, trainLabels)

        #make prediction
        label_prediction = svclassifier.predict(testAttributes)

        #evaluation
        print("Accuracy: ", accuracy_score(testLabels, label_prediction))

    def logistic_regression(self):
        '''
        Logistic regression model

        :param: None
        :return: None
        '''

        trainAttributes = self.__processor.get_train_attributes()
        trainLabels = self.__processor.get_train_labels()
        testAttributes = self.__processor.get_test_attributes()
        testLabels = self.__processor.get_test_labels()

        lg = LogisticRegression(
            solver = 'newton-cg',
            multi_class = 'multinomial'
        )
        lg.fit(trainAttributes, trainLabels)
        print("Accuracy - score: ", lg.score(testAttributes, testLabels))

        lg_pred = lg.predict(testAttributes)
        print("Accuracy - predict: ", accuracy_score(testLabels, lg_pred))

    def ff_network(self, n):
        '''
        Fowrad feeding neural network with one hidden layer.

        '''
        X = self.__processor.get_train_attributes()
        Y = self.__processor.get_train_labels()
        in_len = 9
        out_len = 10
        YY = self.__processor.convert_label(Y)
        print("Train label converted into vector label")
        model = Sequential()
        if(n == 1):
            model.add(Dense(int(num_hidden_layer1(in_len,out_len,len(Y))), input_dim=in_len, activation='relu'))
        elif(n==2):
            model.add(Dense(int(num_hidden_layer2(in_len,out_len)), input_dim=in_len, activation='relu'))
        elif(n==3):
            model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(Y))[0]), input_dim=in_len, activation='relu'))
            model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(Y))[1]), activation='relu'))
        model.add(Dense(out_len, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, YY, epochs=150, batch_size=10, verbose=0)

        Y = self.__processor.get_test_labels()
        YY = self.__processor.convert_label(Y)
        print("Test label converted into vector label")
        scores = model.evaluate(self.__processor.get_test_attributes(),YY)
        print('Test Data Accuracy',scores[1])
