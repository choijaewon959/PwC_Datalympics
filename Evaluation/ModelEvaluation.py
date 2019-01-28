'''
Class for evaluating the trained model.
'''
from sklearn.metrics import accuracy_score, auc, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

from evaluation.Visualization import Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelEvaluation:
    def __init__(self, X_test, y_test=None, X_train=None, y_train=None):
        '''
        Constructor

        :param: X_test: attributes for test
                y_test: labels for test
        :return: None
        '''

        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test

        self.__predicted_label = None

    def evaluate_model(self, model):
        '''
        Function that evaluates the trained model.

        :param: model object (str, trained model)
        :return: accuracy score
        '''
        modelName = model[0]
        model = model[1]

        X_train = self.__X_train
        y_train = self.__y_train
        X_test = self.__X_test
        y_test = self.__y_test

        y_pred = model.predict(X_test)
        self.__predicted_label = y_pred

        acc_model = (y_pred == y_test).sum().astype(float) / len(y_pred)*100
        print(modelName,"'s prediction accuracy is: %3.2f" % (acc_model))

        accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))

        #visualization
        visual = Visualization(y_pred)

        #confusion matrix
        visual.plot_confusion_matrix(y_train, y_test)
        visual.classification_report(y_train, y_test)

        if modelName == "XGBClassifier":
            results = model.evals_result()
            visual.draw_log_loss(results)   #log loss

            visual.draw_classification_error(results)   #classification error
        
        return accuracy
        #accuracy_per_roc_auc = roc_auc_score(np.array(testLabels).flatten(), y_pred)
        #print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))

    def run_model(self, model):
        '''
        Runs the trained model.
        Saves the predicted label into private variable predicted_label.
        Only used in test.py

        :param: None
        :return: None
        '''
        modelName = model[0]
        model = model[1]

        X_train = self.__X_train
        y_train = self.__y_train
        X_test = self.__X_test
        y_test = self.__y_test

        y_pred = model.predict(X_test)
        self.__predicted_label = y_pred

    def get_predicted_label(self):
        '''
        Return the predicted label  (Series)

        :param: None
        :return: predicted numpy array
        '''
        return pd.Series(self.__predicted_label)
