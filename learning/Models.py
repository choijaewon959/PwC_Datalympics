'''
Model object that contains all the possible classification models
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.Preprocessor import Preprocessor
from learning.Hyperparameter import *

import xgboost

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import sys
sys.path.append('./learning')
from num_node import *

import time
import itertools
from evaluation.Visualization import *

# from keras.models import Sequential
# from keras.layers import Dense
# from num_node import *

class Models:
    def __init__(self):
        self.__algorithms = set() # list containing all the algorithms (str)


    def k_neighbor(self, paramDic, X_train, y_train, X_test, y_test):
        #Accuracy: 0.7485575514435755 using 800k dataset
        start_time = time.time()

        """if scaling is necessary for running this algorithm"""
        # scaling = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        # X_train = scaling.transform(X_train)
        # X_test = scaling.transform(X_test)

        knn = KNeighborsClassifier(
            n_neighbors=paramDic['n_neighbors'],
            weights=paramDic['weights'], algorithm=paramDic['algorithm'], leaf_size=paramDic['leaf_size'],
            p=paramDic['p'],metric=paramDic['metric'], metric_params=paramDic['metric_params'], n_jobs=paramDic['n_jobs']
        )

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        learningtime= time.time() - start_time
        print("---k_neighbor took : %.2f seconds---"  % learningtime)

        accuracy = accuracy_score(y_test, y_pred)
        print("k_neighbor Accuracy: " , accuracy)

        # scores = cross_val_score(knn, data.drop['loan_status'], data['loan_status'], cv=5)
        # print("cross_val_score is : ", scores)

        visual = Visualization(y_pred)
        visual.plot_confusion_matrix(X_train, y_train, X_test, y_test)
        visual.classification_report(X_train, y_train, X_test, y_test)

        return accuracy

    def decision_tree(self, paramDic, X_train, y_train, X_test, y_test):
        start_time = time.time()

        clf = DecisionTreeClassifier(
                 criterion=paramDic['criterion'],
                 splitter=paramDic['splitter'],
                 max_depth=paramDic['max_depth'],
                 min_samples_split=paramDic['min_samples_split'],
                 min_samples_leaf=paramDic['min_samples_leaf'],
                 min_weight_fraction_leaf=paramDic['min_weight_fraction_leaf'],
                 max_features=paramDic['max_features'],
                 random_state=paramDic['random_state'],
                 max_leaf_nodes=paramDic['max_leaf_nodes'],
                 min_impurity_decrease=paramDic['min_impurity_decrease'],
                 min_impurity_split=paramDic['min_impurity_split'],
                 presort=paramDic['presort'])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        learningtime= time.time() - start_time
        print("---decision_tree took : %.2f seconds---"  % learningtime)

        accuracy = accuracy_score(y_test, y_pred)
        print("[decision_tree Accuracy: %.4f ]" % (accuracy*100) )

        return accuracy

    def random_forest(self, paramDic, X_train, y_train, X_test, y_test):
        start_time = time.time()
        rfc = RandomForestClassifier(
                 n_estimators=paramDic['n_estimators'],
                 criterion=paramDic['criterion'],
                 max_depth=paramDic['max_depth'],
                 min_samples_split=paramDic['min_samples_split'],
                 min_samples_leaf=paramDic['min_samples_leaf'],
                 min_weight_fraction_leaf=paramDic['min_weight_fraction_leaf'],
                 max_features=paramDic['max_features'],
                 max_leaf_nodes=paramDic['max_leaf_nodes'],
                 min_impurity_decrease=paramDic['min_impurity_decrease'],
                 min_impurity_split=paramDic['min_impurity_split'],
                 bootstrap=paramDic['bootstrap'],
                 oob_score=paramDic['oob_score'],
                 n_jobs=paramDic['n_jobs'],
                 random_state=paramDic['random_state'],
                 verbose=paramDic['verbose'],
                 warm_start=paramDic['warm_start'],
                 class_weight=paramDic['class_weight'])

        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        learningtime= time.time() - start_time
        print("---random_forest took : %.2f seconds---"  % learningtime)
        #
        # acc_rfc = (y_pred == y_test).sum().astype(float) / len(y_pred)*100
        # print("Scikit-Learn's Random Forest Classifier's prediction accuracy is: %3.2f" % (acc_rfc))

        accuracy = accuracy_score(y_test, y_pred)
        print("[Random Forest Classifier Accuracy: %.4f %%]" % (accuracy *100) )

        return accuracy

    def XGBClassifier(self, paramDic, X_train, y_train, X_test, y_test):

        # print(X_train.head())
        # print(y_train.head())
        # print(X_test.shape)
        # print(y_test.shape)

        eval_set=[(X_test, y_test)]

        clf = xgboost.sklearn.XGBClassifier(
            max_depth=paramDic['max_depth'], learning_rate=paramDic['learning_rate'], n_estimators=paramDic['n_estimators'],
            silent=paramDic['silent'], objective=paramDic['objective'],
            booster=paramDic['booster'], n_jobs=paramDic['n_jobs'], nthread=paramDic['nthread'], gamma=paramDic['gamma'],
             min_child_weight=paramDic['min_child_weight'], max_delta_step=paramDic['max_delta_step'],
            subsample=paramDic['subsample'], colsample_bytree=paramDic['colsample_bytree'], colsample_bylevel=paramDic['colsample_bylevel'],
             reg_alpha=paramDic['reg_alpha'], reg_lambda=paramDic['reg_lambda'], scale_pos_weight=paramDic['scale_pos_weight'],
             base_score=paramDic['base_score'],
             random_state=paramDic['random_state'], seed=paramDic['seed'], missing=paramDic['missing'], importance_type=paramDic['importance_type'])

        clf.fit(X_train, y_train, early_stopping_rounds=20,  eval_set=eval_set, verbose=True)

        y_pred = clf.predict(X_test)

        acc_xgb = (y_pred == y_test).sum().astype(float) / len(y_pred)*100
        print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))

        accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))

        visual = Visualization(y_pred)
        visual.plot_confusion_matrix(X_train, y_train, X_test, y_test)
        visual.classification_report(X_train, y_train, X_test, y_test)

        return accuracy
        #accuracy_per_roc_auc = roc_auc_score(np.array(testLabels).flatten(), y_pred)
        #print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))


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


    def SVM(self, paramDic, X_train, y_train, X_test, y_test):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: gaussian

        :param: None
        :return: None
        '''
        #train svm model
        print("Learning...")
        svclassifier = SVC(
            C=paramDic['C'],
            cache_size=paramDic['cache_size'],
            class_weight=paramDic['class_weight'],
            coef0=paramDic['coef0'],
            decision_function_shape=paramDic['decision_function_shape'],
            degree=paramDic['degree'],
            gamma=paramDic['gamma'],
            kernel=paramDic['kernel'],
            max_iter=paramDic['max_iter'],
            probability=paramDic['probability'],
            random_state=paramDic['random_state'],
            shrinking=paramDic['shrinking'],
            tol=paramDic['tol'],
            verbose=paramDic['verbose']
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
        num_of_folds = 10

        accuracy = accuracy_score(y_test, lg_pred)

        cross_valid_accuracy = cross_val_score(lg, X_train, y_train, scoring='accuracy', cv = num_of_folds).mean()/num_of_folds
        print("Accuracy: ", accuracy)
        print("cross validation accuracy ", cross_valid_accuracy)
        return accuracy

    def ff_network(self, n, X_train, y_train, X_test, y_test, p):
        '''
        Forward feeding neural network with one/two hidden layer.

        '''
        X = X_train
        Y = y_train
        in_len = 4 # number of input feature
        out_len = 3 # number of output label
        print(y_train)
        YY = to_categorical(Y)
        print("Train label converted into vector label")
        model = Sequential()
        hidden_act = ['sigmoid','tanh', 'relu']
        epoch = [5,20, 50]
        Y_test = to_categorical(y_test)
        out = ""
        for act in hidden_act:
            for ep in epoch:
                if(n==1):
                    model.add(Dense(int(num_hidden_layer1(in_len,out_len,len(Y))), input_dim=in_len, activation=act))
                elif(n==2):
                    model.add(Dense(int(num_hidden_layer2(in_len,out_len)), input_dim=in_len, activation=act))
                elif(n==3):
                    model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(Y))[0]), input_dim=in_len, activation=act))
                    model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(Y))[1]), activation=act))
                model.add(Dense(out_len, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X, YY, epochs=ep, batch_size=20, verbose=0)
                scores = model.evaluate(X_test, Y_test)
                out = out +"Accuracy : "+ str(scores[1]) + ", Hidden layer activation : " + act +" Epoch:"+str(ep) +" |"
                print(out)
        print(out)
