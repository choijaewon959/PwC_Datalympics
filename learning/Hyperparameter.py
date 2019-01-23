'''
Configuration containing all the hyperparameters used in each model.
'''
k_neighor_dict = {
'n_neighbors' : 5 ,
 'weights' : 'uniform' ,
  'algorithm' : 'auto',
  'leaf_size' : 30,
  'p': 2 ,
  'metric' : 'minkowski',
  'metric_params' : None,
  'n_jobs' : None
}

decision_tree_dict = {
                'criterion' : "gini",
                 'splitter' : "best",
                 'max_depth' : None,
                 'min_samples_split' : 2,
                 'min_samples_leaf' : 1,
                 'min_weight_fraction_leaf'=0.,
                 'max_features' : None,
                 'random_state' : None,
                 'max_leaf_nodes' : None,
                 'min_impurity_decrease' : 0.,
                 'min_impurity_split' : None,
                 'class_weight' : None,
                 'presort' : False
}
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
