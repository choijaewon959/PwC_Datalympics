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
from sklearn import preprocessing
from learning.Hyperparameter import *

import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

<<<<<<< HEAD

=======
>>>>>>> 45cefc032b4e5089d9ce7fd1cc0db0f3be3e1ba5
from keras.models import Sequential
from keras.layers import Dense
import sys
sys.path.append('./learning')
from num_node import *
<<<<<<< HEAD
=======

import time
import itertools


# from keras.models import Sequential
# from keras.layers import Dense
# from num_node import *
>>>>>>> 45cefc032b4e5089d9ce7fd1cc0db0f3be3e1ba5

class Models:
    def __init__(self):
        self.__algorithms = set() # list containing all the algorithms (str)


    def k_neighbor(self, X_train, y_train, X_test, y_test):
        #Accuracy: 0.7485575514435755 using 800k dataset

        # labels=self.__dataprocessor.get_labels()
        # data= self.__dataprocessor.get_data()


        """if scaling is necessary for running this algorithm"""
        # scaling = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        # X_train = scaling.transform(X_train)
        # X_test = scaling.transform(X_test)

        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print("--- %s seconds ---" % (time.time() - start_time))

        print("Accuracy:",accuracy_score(y_test, y_pred))

        # scores = cross_val_score(knn, data.drop['loan_status'], data['loan_status'], cv=5)
        # print("cross_val_score is : ", scores)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        # features = y_train.unique()
        #
        # def plot_confusion_matrix(cm, classes,
        #                   normalize=False,
        #                   title='Confusion matrix',
        #                   cmap=plt.cm.Blues):
        #     """
        #     This function prints and plots the confusion matrix.
        #     Normalization can be applied by setting `normalize=True`.
        #     """
        #     if normalize:
        #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #         print("Normalized confusion matrix")
        #     else:
        #         print('Confusion matrix, without normalization')
        #
        #     print(cm)
        #
        #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #     plt.title(title)
        #     plt.colorbar()
        #     tick_marks = np.arange(len(classes))
        #     plt.xticks(tick_marks, classes, rotation=45)
        #     plt.yticks(tick_marks, classes)
        #
        #     fmt = '.2f' if normalize else 'd'
        #     thresh = cm.max() / 2.
        #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #         plt.text(j, i, format(cm[i, j], fmt),
        #                  horizontalalignment="center",
        #                  color="white" if cm[i, j] > thresh else "black")
        #
        #     plt.ylabel('True label')
        #     plt.xlabel('Predicted label')
        #     plt.tight_layout()
        #
        #
        # # Compute confusion matrix
        # cnf_matrix = confusion_matrix(y_test, y_pred)
        # np.set_printoptions(precision=2)
        #
        # # Plot non-normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=features,
        #                       title='Confusion matrix, without normalization')
        #
        # # Plot normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=features, normalize=True,
        #                       title='Normalized confusion matrix')
        #
        # plt.show()
        #
        # """
        # classification report is done when target_names is
        # in string type
        # (error when float64 was given to labels)
        # """
        #
        # # print(features)
        #
        # print(classification_report(y_test, y_pred,target_names=features))

    def decision_tree(self, X_train, y_train, X_test, y_test):

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:",accuracy_score(y_test, y_pred))

    def random_forest(self, X_train, y_train, X_test, y_test):

        rfc = RandomForestClassifier(n_estimators=30)
        rfc.fit(X_train, y_train)
        preds = rfc.predict(X_test)
        acc_rfc = (preds == y_test).sum().astype(float) / len(preds)*100
        print("Scikit-Learn's Random Forest Classifier's prediction accuracy is: %3.2f" % (acc_rfc))

    def XGBClassifier(self, X_train, y_train, X_test, y_test):

        # print(X_train.head())
        # print(y_train.head())
        # print(X_test.shape)
        # print(y_test.shape)

        eval_set=[(X_test, y_test)]

        clf = xgboost.sklearn.XGBClassifier(
            #objective="multi:softprob",
            learning_rate=0.05,
            seed=0, #seed that is not random
            max_depth=4,
            min_child_weight=1,
            reg_alpha=0.005,
            gamma=0,
            n_estimators=200, subsample=0.8, colsample_bytree=0.8)

        clf.fit(X_train, y_train, early_stopping_rounds=20,  eval_set=eval_set, verbose=True)

        y_pred = clf.predict(X_test)

        acc_xgb = (y_pred == y_test).sum().astype(float) / len(y_pred)*100
        print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))

        accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))

        #accuracy_per_roc_auc = roc_auc_score(np.array(testLabels).flatten(), y_pred)
        #print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))

        #print("Hello I'm binary logistic regression")

    def linear_SVM(self, X_train, y_train, X_test, y_test):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: linear

        :param: None
        :return None
        '''

        # scaling = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        # X_train = scaling.transform(X_train)
        # X_test = scaling.transform(X_test)

        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
        y_pred = svm_model_linear.predict(X_test)

        #train svm model
        # print("Learning...")
        # svclassifier = SVC(kernel = 'linear')
        # svclassifier.fit(X_train, y_train)

        print("Learning...")
        svclassifier = SVC(
            C = 10000,
            kernel = 'linear'
        )
        svclassifier.fit(X_train, y_train)

        #make prediction
        #label_prediction = svclassifier.predict(X_test)

        #evaluation
        print("Accuracy: ", accuracy_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        """
        https://en.wikipedia.org/wiki/F1_score
        https://en.wikipedia.org/wiki/Precision_and_recall

        The F1 score is the harmonic average of the precision and recall,
        where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
        """

    def SVM(self, X_train, y_train, X_test, y_test):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: gaussian

        :param: None
        :return: None
        '''
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
        svclassifier.fit(X_train, y_train)

        #make prediction
        label_prediction = svclassifier.predict(X_test)

        #evaluation
        accuracy = accuracy_score(y_test, label_prediction)
        print("Accuracy: ", accuracy)
        return accuracy

    def logistic_regression(self, paramDic, X_train, y_train, X_test, y_test):
        '''
        Logistic regression model

        :param: None
        :return: None
        '''
        lg = LogisticRegression(
            penalty=paramDic['penalty'],
            dual=paramDic['dual'],
            tol=paramDic['tol'],
            C=paramDic['C'],
            fit_intercept=paramDic['fit_intercept'],
            intercept_scaling=paramDic['intercept_scaling'],
            class_weight=paramDic['class_weight'],
            random_state=paramDic['random_state'],
            solver=paramDic['solver'],
            max_iter=paramDic['max_iter'],
            multi_class=paramDic['multi_class'],
            verbose=paramDic['verbose'],
            warm_start=paramDic['warm_start'],
            n_jobs=paramDic['n_jobs']
        )
        lg.fit(X_train, y_train)
        lg_pred = lg.predict(X_test)
        accuracy = accuracy_score(y_test, lg_pred)
        print("Accuracy - predict: ", accuracy)
        return accuracy

    def ff_network(self, n, X_train, y_train, X_test, y_test):
        '''
        Forward feeding neural network with one/two hidden layer.

        '''
        X = X_train
        Y = y_train
        in_len = 9 # number of input feature
        out_len = 10 # number of output label
        YY = self.__processor.convert_label(Y)
        print("Train label converted into vector label")
        model = Sequential()
        if(n==1):
            model.add(Dense(int(num_hidden_layer1(in_len,out_len,len(Y))), input_dim=in_len, activation='relu'))
        elif(n==2):
            model.add(Dense(int(num_hidden_layer2(in_len,out_len)), input_dim=in_len, activation='relu'))
        elif(n==3):
            model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(Y))[0]), input_dim=in_len, activation='relu'))
            model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(Y))[1]), activation='relu'))
        model.add(Dense(out_len, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, YY, epochs=150, batch_size=10, verbose=0)

        Y = y_test
        YY = self.__processor.convert_label(Y)
        print("Test label converted into vector label")
        scores = model.evaluate(X_test, YY)
        print('Test Data Accuracy',scores[1])
