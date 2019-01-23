'''
Configuration containing all the hyperparameters used in each model.
'''
k_neighor_dict = {}
decision_tree_dict = {}
random_forest_dict = {}
XGBClassifier_dict = {}
linear_SVM_dict = {}

SVM_dict = {
    'C' : 1.0,
    'cache_size' : 700,
    'class_weight' : None,
    'cdef' : 0.0,
    'decision_function_shape' : 'ovo',
    'degree' : 3,
    'gamma' : 'scale',
    'kernel' : 'rbf',
    'max_iter' : -1,
    'probability' : False,
    'random_state' : None,
    'shrinking' : True,
    'tol' : 0.001,
    'verbose' : False
}

logistic_regression_dict = {
    'penalty' : 'l2',
    'dual' : False,
    'tol' : 0.0001,
    'C' : 5.0,
    'fit_intercept' : True,
    'intercept_scaling' : 1,
    'class_weight' : None,
    'random_state' : None,
    'solver' : 'liblinear',
    'max_iter' : 100,
    'multi_class' : 'ovr',
    'verbose' : 0,
    'warm_start' : False,
    'n_jobs' : None
}

ff_network_dict = {
 
}
