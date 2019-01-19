'''
Model object that contains all the possible classification models
'''
class Models:
    def __init__(self, dataFrame):
        self.__algorithms = set() # list containing all the algorithms (str)
        self.__data = dataFrame

    def binary_logistic_regression(self):
        # TODO: code by taemin
