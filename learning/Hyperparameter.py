'''
Configuration class that contains all the hyperparameters used in each model.
'''

class Hyperparameter:
    def __init__(self):
        self.__k_neighor_dict = {}
        self.__decision_tree_dict = {}
        self.__random_forest_dict = {}
        self.__XGBClassifier_dict = {}
        self.__linear_SVM_dict = {}
        self.__gaussian_SVM_dict = {}
        self.__logistic_regression_dict = {}
        self.__ff_network_dict = {}

        self.__set_k_neighbor_hyperparams()
        self.__set_decision_tree_hyperparams()
        self.__set_random_forest_hyperparams()
        self.__set_logistic_regression_hyperparams()

    def __set_k_neighbor_hyperparams(self):
        '''
        Set hyperparameters for k_neighbor model here.
        '''
        pass

    def __set_decision_tree_hyperparams(self):
        '''
        Set hyperparameters for decision_tree model here.
        '''
        pass

    def __set_random_forest_hyperparams(self):
        '''
        Set hyperparameters for random_forest model here.
        '''
        pass

    def __set_XGBClassifier_hyperparams(self):
        '''
        Set hyperparameters for XGBClassifier model here.
        '''
        pass

    def __set_linear_SVM_hyperparams(self):
        '''
        Set hyperparameters for linear_SVM model here.
        '''
        pass

    def __set_gaussian_SVM_hyperparams(self):
        '''
        Set hyperparameters for gaussian_SVM model here.
        '''
        self.__gaussian_SVM_dict['C'] = 1.0
        self.__gaussian_SVM_dict['cache_size'] = 700
        self.__gaussian_SVM_dict['class_weight'] = None
        self.__gaussian_SVM_dict['cdef'] = 0.0
        self.__gaussian_SVM_dict['decision_function_shape'] = 'ovo'
        self.__gaussian_SVM_dict['degree'] = 3
        self.__gaussian_SVM_dict['gamma'] = 'scale'
        self.__gaussian_SVM_dict['kernel'] = 'rbf'
        self.__gaussian_SVM_dict['max_iter'] = -1
        self.__gaussian_SVM_dict['probability'] = False
        self.__gaussian_SVM_dict['random_state'] = None
        self.__gaussian_SVM_dict['shrinking'] = True
        self.__gaussian_SVM_dict['tol'] = 0.001
        self.__gaussian_SVM_dict['verbose'] = False

    def __set_logistic_regression_hyperparams(self):
        '''
        Set hyperparameters for logistic_regression model here.
        '''
        self.__logistic_regression_dict['penalty'] = 'l2'
        self.__logistic_regression_dict['dual'] = False
        self.__logistic_regression_dict['tol'] = 0.0001
        self.__logistic_regression_dict['C'] = 10.0 
        self.__logistic_regression_dict['fit_intercept'] = True 
        self.__logistic_regression_dict['intercept_scaling'] = 1
        self.__logistic_regression_dict['class_weight'] = None
        self.__logistic_regression_dict['random_state'] = None
        self.__logistic_regression_dict['solver'] = 'newton-cg'
        self.__logistic_regression_dict['max_iter'] = 500
        self.__logistic_regression_dict['multi_class'] = 'multinomial'
        self.__logistic_regression_dict['verbose'] = 0 
        self.__logistic_regression_dict['warm_start'] = False  
        self.__logistic_regression_dict['n_jobs'] = None


    def get_k_neighbor_hyperparams(self):
        '''
        return the dictionary that contains all the hyperparameters

        :param: None
        :return: Dictionary (key: hyperparameter name, value: value of parameter)
        '''
        return self.__k_neighor_dict

    def get_decision_tree_hyperparams(self):
        '''
        return the dictionary that contains all the hyperparameters

        :param: None
        :return: Dictionary (key: hyperparameter name, value: value of parameter)
        '''
        return self.__decision_tree_dict

    def get_gaussian_SVM_hyperparams(self):
        '''
        return the dictionary that contains all the hyperparameters

        :param: None
        :return: Dictionary (key: hyperparameter name, value: value of parameter)
        '''
        return self.__gaussian_SVM_dict

    def get_logistic_regression_hyperparams(self):
        '''
        return the dictionary that contains all the hyperparameters

        :param: None
        :return: Dictionary (key: hyperparameter name, value: value of parameter)
        '''
        return self.__logistic_regression_dict