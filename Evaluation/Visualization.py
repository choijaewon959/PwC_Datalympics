import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

"""
https://en.wikipedia.org/wiki/F1_score
https://en.wikipedia.org/wiki/Precision_and_recall

The F1 score is the harmonic average of the precision and recall,
where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
"""


class Visualization:
    def __init__(self, learningResult):
        self.__learningResult=learningResult


    def plot_confusion_matrix(self,X_train, y_train, X_test, y_test):
        '''
        This function prints and plots the confusion matrix.

        :param: X_train, y_train, X_test, y_test (Pandas DataFrame (segments))
        :return: None
        '''

        #features concatenate  ; bc/ dataProcessor is not directly accessible
        features = y_train.unique()
        testfeatures= y_test.unique()
        result = np.unique(np.concatenate((features,testfeatures),0))
        #print (result)

        def ploting(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
            """
            This inner function plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.

            :param: X_train, y_train, X_test, y_test ()
            :return: None
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()


        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, self.__learningResult)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        ploting(cnf_matrix, classes=result,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        ploting(cnf_matrix, classes=result, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()



    def classification_report(self, X_train, y_train, X_test, y_test):
        """
        This function prints a classification_report.

        :param: X_train, y_train, X_test, y_test (Pandas DataFrame (segments))
        :return: None
        """
        features = y_train.unique()
        testfeatures= y_test.unique()
        result = np.unique(np.concatenate((features,testfeatures),0))
        a = np.array(result.tolist())
        print (a)
        list=[]
        for i in a:
            st=str(i)
            list.append(st)
        #result = np.array2string(result, precision=2)

        print(classification_report(y_test, self.__learningResult, target_names=list))
