'''
class to keep the history of the log of the models with different parameters
'''
import os
import datetime
import csv

class ResultLog:
    def __init__(self):
        self.__best_model = None
        self.__best_param = None
        self.__best_result = None # return the best algorithm with hyperparam

    def log_result(self, modelName, accuracy, hyperparameter):
        '''
        Log a new result so as to keep track of the best model and corresponding hyperparameter so far.

        :param:
            modelName: name of the model to be logged. (str)
            hyperparameter: hyperparameters. (dict)
            accuray: accuracy of the model. (float)

        :return: None
        '''
        currentTime = datetime.datetime.now()
        row = ['Model', 'Accuracy', 'Hyperparameter', 'Time']
        paramString = ""

        #change dictionary into string values
        for key in hyperparameter.keys():
            paramString += str(key) + ":" + str(hyperparameter[key])+ ","

        paramString = paramString[:len(paramString)-1] # erase comma at the end

        if not os.path.isfile('evaluation/result.csv'): # file does not exist
            with open('evaluation/result.csv', mode = 'w', newline = '') as csv_file:
                print("test")
                writer = csv.DictWriter(csv_file, fieldnames=row)
                writer.writeheader()
                writer.writerow({"Model": modelName, "Accuracy": accuracy, "Hyperparameter": paramString, "Time": currentTime})
        
        else:   #file exist
            with open('evaluation/result.csv', mode = 'a', newline = '') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=row)
                writer.writerow({"Model": modelName, "Accuracy": accuracy, "Hyperparameter": paramString, "Time": currentTime})

    def get_best_model(self):
        '''
        Return the model with the highest score.

        :param: None
        :return: name of the model (str)
        '''
        return self.__best_model

    def get_best_result(self):
        '''
        Return the the algorithm and best corresponding hyperparameter.

        :param: None
        :return: None
        '''
        return self.__best_result