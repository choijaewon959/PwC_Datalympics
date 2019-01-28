'''
Model object that contains all the possible classification models
'''
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from data.Preprocessor import Preprocessor
from learning.Hyperparameter import *
from util.Math import *
import xgboost
import sys

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

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical

sys.path.append('./learning')
from num_node import *
# from livelossplot.keras import PlotLossesCallback

import time
import itertools
from evaluation.Visualization import *
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
from num_node import *

class Models:
    def __init__(self):
        self.__algorithms = set() # list containing all the algorithms (str)

    def k_neighbor(self, paramDic, X_train, y_train, X_test, y_test):
        #Accuracy: 0.7485575514435755 using 800k dataset
        start_time = time.time()

        knn = KNeighborsClassifier(
            n_neighbors=paramDic['n_neighbors'],
            weights=paramDic['weights'], algorithm=paramDic['algorithm'], leaf_size=paramDic['leaf_size'],
            p=paramDic['p'],metric=paramDic['metric'], metric_params=paramDic['metric_params'], n_jobs=paramDic['n_jobs']
        )

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        learningtime= time.time() - start_time
        print("---k_neighbor took : %.2f seconds---"  % learningtime)

        print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("k_neighbor Accuracy: " , accuracy)

        # scores = cross_val_score(knn, data.drop['loan_status'], data['loan_status'], cv=5)
        # print("cross_val_score is : ", scores)

        visual = Visualization(y_pred)
        visual.plot_confusion_matrix(y_train, y_test)
        visual.classification_report(y_train, y_test)

        return accuracy

    def decision_tree(self, paramDic, X_train, y_train, X_test, y_test):

        modelName = "decision_tree"

        dt = DecisionTreeClassifier(
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

        dt.fit(X_train, y_train)

        return (modelName, dt)

    def random_forest(self, paramDic, X_train, y_train, X_test, y_test):
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

        rf=rfc.fit(X_train, y_train)
        #fit.summary()
        return rf

    def XGBClassifier(self, paramDic, X_train, y_train, X_test, y_test):

        modelName = "XGBClassifier"

        eval_set=[(X_train, y_train), (X_test, y_test)]

        xgb = xgboost.sklearn.XGBClassifier(
            max_depth=paramDic['max_depth'],
            learning_rate=paramDic['learning_rate'],
            n_estimators=paramDic['n_estimators'],
            silent=paramDic['silent'],
            objective=paramDic['objective'],
            booster=paramDic['booster'],
            n_jobs=paramDic['n_jobs'],
            nthread=paramDic['nthread'],
            gamma=paramDic['gamma'],
            min_child_weight=paramDic['min_child_weight'],
            max_delta_step=paramDic['max_delta_step'],
            subsample=paramDic['subsample'],
            colsample_bytree=paramDic['colsample_bytree'],
            colsample_bylevel=paramDic['colsample_bylevel'],
            reg_alpha=paramDic['reg_alpha'],
            reg_lambda=paramDic['reg_lambda'],
            scale_pos_weight=paramDic['scale_pos_weight'],
            base_score=paramDic['base_score'],
            random_state=paramDic['random_state'],
            seed=paramDic['seed'], 
            missing=paramDic['missing'],
            importance_type=paramDic['importance_type']
        )

        xgb.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True)

        return (modelName, xgb)

    def SVM(self, paramDic, X_train, y_train, X_test, y_test):
        '''
        Support Vector Machine algorithm for categorical classification.
        Kernel: gaussian

        :param: None
        :return: None
        '''

        modelName = "SVC"

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

        return (modelName, svclassifier)

    def logistic_regression(self, paramDic, X_train, y_train, X_test, y_test):
        '''
        Logistic regression model

        :param: None
        :return: None
        '''

        modelName = "logistic_regression"

        print("Training logistic regression...")
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

        return (modelName, lg)

    def ff_network(self, n, X_train, y_train, X_test, y_test):
        '''
        Forward feeding neural network with one/two hidden layer.

        :param: None
        :return: None
        '''
        in_len = X_train.shape[1] # number of input feature
        out_len = 10 # number of output label
        hidden_layer_l = [25, 15]
        weight_mu = [0.1]
        hidden_act = 'tanh'
        ep = 150
        #plot_losses = PlotLossesCallback()

        print("Train label converted into vector label")
        Y_test = to_categorical(y_test)
        Y_train = to_categorical(y_train)

        model = Sequential()
        out = ""
        print(Y_train)
        for hidden_layer in hidden_layer_l:
            for weight in weight_mu:
                class_weight = create_class_weight(y_test,weight)
                print(class_weight)

                if(n==1):
                    model.add(Dense(hidden_layer,input_dim=in_len,activation=hidden_act))
                elif(n==2):
                    model.add(Dense(int(num_hidden_layer2(in_len,out_len)), input_dim=in_len, activation=hidden_act))
                elif(n==3):
                    model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(y_train))[0]), input_dim=in_len, activation=hidden_act))
                    model.add(Dense(int(num_hidden_layer3(in_len,out_len,len(y_train))[1]), activation=hidden_act))
            model.add(Dense(out_len, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(X_train, Y_train, epochs=ep, batch_size=20, verbose=1, class_weight=class_weight, validation_data=(X_test, Y_test),callbacks=[plot_losses])
            scores_test = model.evaluate(X_test, Y_test)
            scores_train = model.evaluate(X_train, Y_train)

            #back to array
            y_pred = np.argmax(model.predict(X_test),axis=1)
            visual = Visualization(y_pred)
            visual.plot_confusion_matrix(y_train, y_test)
            print("y_pred == ", y_pred)
            print("confusion matrix printed")
            visual.classification_report(y_train, y_test)
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # out = "Accuracy : "+ str(scores_test[1]) + ", Hidden layer activation : " + hidden_act +" Epoch:"+str(ep) +'\n'
            # f = open(str(weight)+"_"+str(n)+"_"+str(hidden_layer)+".txt", "a")
            # f.write(out)
            # f.write("Test file Accurcy= "+str(scores_test))
            # f.write("Train file Accurcy= "+str(scores_train))
            # f.write(visual.plot_confusion_matrix(y_train, y_test))

            # #save model
            # model_json = model.to_json()
            # with open("model.json", "w") as json_file:
            #     json_file.write(model_json)
            # model.save_weights("model.h5")
            # print("Saved model to disk")

        return scores[1]

    def ffnn_eval(X_test):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        y_pred = np.argmax(model.predict(X_test),axis=1)
        data = X_test.join(pd.DataFrame(y_pred))
        print(data)
