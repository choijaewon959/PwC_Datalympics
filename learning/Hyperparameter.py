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
                 'min_weight_fraction_leaf': 0,
                 'max_features' : None,
                 'random_state' : None,
                 'max_leaf_nodes' : None,
                 'min_impurity_decrease' : 0.,
                 'min_impurity_split' : None,
                 'class_weight' : None,
                 'presort' : False
}

random_forest_dict = {
                 'n_estimators':'warn',
                 'criterion':"gini",
                 'max_depth':None,
                 'min_samples_split':2,
                 'min_samples_leaf':1,
                 'min_weight_fraction_leaf':0.,
                 'max_features':"auto",
                 'max_leaf_nodes':None,
                 'min_impurity_decrease':0.,
                 'min_impurity_split':None,
                 'bootstrap':True,
                 'oob_score':False,
                 'n_jobs':None,
                 'random_state':None,
                 'verbose':0,
                 'warm_start':False,
                 'class_weight':None

}

XGBClassifier_dict = {
     'max_depth':5, #default: 3
     'learning_rate':0.01,
     'n_estimators':200,
     'silent':True,
     'objective':'multi:softprob',
     'booster':'gbtree',
     'n_jobs':3,
     'nthread':None,
     'gamma':1, #0 : no regularization, 1: medium regularization, 5: high regularization
     'min_child_weight':1,
     'max_delta_step':0,
     'subsample':0.8,
     'colsample_bytree':0.8,
     'colsample_bylevel':1,
     'reg_alpha':0,
     'reg_lambda':1,
     'scale_pos_weight':1,
     'base_score':0.5,
     'random_state':0,
     'seed':None,
     'missing':None,
     'importance_type':'gain'
}

XGBClassifier_dict2 = {
     'max_depth':4, #default: 3
     'learning_rate':0.0001,
     'n_estimators':50,
     'silent':True,
     'objective':'binary:logistic',
     'booster':'gbtree',
     'n_jobs':3,
     'nthread':None,
     'gamma':10, #0 : no regularization, 1: medium regularization, 5: high regularization
     'min_child_weight':1,
     'max_delta_step':0,
     'subsample':0.8,
     'colsample_bytree':0.8,
     'colsample_bylevel':1,
     'reg_alpha':3,
     'reg_lambda':3,
     'scale_pos_weight':1,
     'base_score':0.5,
     'random_state':0,
     'seed':None,
     'missing':None,
     'importance_type':'gain'
}

# XGBClassifier_dict2 = {
#      'max_depth':4, #default: 3
#      'learning_rate':0.0075,
#      'n_estimators':100,
#      'silent':True,
#      'objective':'multi:softprob',
#      'booster':'gbtree',
#      'n_jobs':3,
#      'nthread':None,
#      'gamma':10, #0 : no regularization, 1: medium regularization, 5: high regularization
#      'min_child_weight':1,
#      'max_delta_step':0,
#      'subsample':0.8,
#      'colsample_bytree':0.8,
#      'colsample_bylevel':1,
#      'reg_alpha':3,
#      'reg_lambda':3,
#      'scale_pos_weight':1,
#      'base_score':0.5,
#      'random_state':0,
#      'seed':None,
#      'missing':None,
#      'importance_type':'gain'
# }

XGBClassifier_dict3 = {
     'max_depth':4, #default: 3
     'learning_rate':0.0075,
     'n_estimators':100,
     'silent':True,
     'objective':'multi:softprob',
     'booster':'gbtree',
     'n_jobs':3,
     'nthread':None,
     'gamma':10, #0 : no regularization, 1: medium regularization, 5: high regularization
     'min_child_weight':1,
     'max_delta_step':0,
     'subsample':0.8,
     'colsample_bytree':0.8,
     'colsample_bylevel':1,
     'reg_alpha':3,
     'reg_lambda':3,
     'scale_pos_weight':1,
     'base_score':0.5,
     'random_state':0,
     'seed':None,
     'missing':None,
     'importance_type':'gain'
}

SVM_dict = {
    'C' : 1.0,
    'cache_size' : 700,
    'class_weight' : None,
    'coef0' : 0.0,
    'decision_function_shape' : 'ovo',
    'degree' : 3,
    'gamma' : 'scale',
    'kernel' : 'rbf',
    'max_iter' : -1,
    'probability' : False,
    'random_state' : 1,
    'shrinking' : True,
    'tol' : 0.001,
    'verbose' : False
}

logistic_regression_dict = {
    'penalty' : 'l2',
    'dual' : False,
    'tol' : 0.0001,
    'C' : 0.001,
    'fit_intercept' : True,
    'intercept_scaling' : 1,
    'class_weight' : None,
    'random_state' : 1,
    'solver' : 'newton-cg',
    'max_iter' : 100,
    'multi_class' : 'multinomial',
    'verbose' : 0,
    'warm_start' : False,
    'n_jobs' : None
}

linear_regression_dict = {
    'fit_intercept' : True,
    'normalize' : False,
    'copy_X' : True,
    'n_jobs' : None
}

polynomial_regression_dict = {
    'fit_intercept' : True,
    'normalize' : False,
    'copy_X' : True,
    'n_jobs' : None
}

polynomial_regression_dict2 = {
    'fit_intercept' : True,
    'normalize' : True,
    'copy_X' : True,
    'n_jobs' : None
}

ridge_regression_dict = {
    'alpha' : 1.0,
    'fit_intercept' : True,
    'normalize' : False,
    'copy_X' : True,
    'max_iter' : None,
    'tol' : 0.001,
    'solver' : 'auto',
    'random_state' : None
}

ff_network_dict = {
    'activation': 'tanh',
    'class_weight': '{ 0:1.0, 1:1.0, 2:3.0,3: 10,4:20,5:20,6:20,7:10}',
    'n' : 2,
    'output_activation':'softmax',
    'loss':'categorical_crossentropy',
    'optimizer':'adam',
    'batch_size':20,
    'epochs':20,
    'hidden_layer': 80,
    'weight_mu' : 0.8
}
