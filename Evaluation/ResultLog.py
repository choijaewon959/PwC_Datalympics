'''
class to keep the history of the log of the models with different parameters
'''
import os
import datetime

class ResultLog:
    def __init__(self):
        self.__best_model = None
        self.__best_param = None
        self.__best_result = None # return the best algorithm with hyperparam

    def log_result(self, modelName, hyperparameter, accuracy):
        '''
        Log a new result so as to keep track of the best model and corresponding hyperparameter so far.

        :param:
            modelName: name of the model to be logged. (str)
            hyperparameter: hyperparameters. (dict)
            accuray: accuracy of the model. (float)

        :return: None
        '''
        resultFile = open("result.txt" , "w")
        log = ""
        currentTime = datetime.datetime.now()

        #update log string
        log = modelName + ', '
        log += "Accuracy: " + accuracy + ', '
        log += "Hyperparameters: "
        for value in hyperparameter.values():
            log += value + ": " + hyperparameter[value] + " "
        log += ", " + currentTime

        #update the txt file
        resultFile.write(log)

        resultFile.close()
    
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