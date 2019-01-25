import numpy as np
from util.Stat import *
from util.Distribution import Distribution
from data.Preprocessor import Preprocessor
from data.FeatureFilter import FeatureFilter
from learning.Hyperparameter import *
from learning.Models import Models
from evaluation.ResultLog import ResultLog

print('test began')

#objects
dataProcessor = Preprocessor()
filtering = FeatureFilter()
algorithm = Models()
result = ResultLog()

#data for training
X_train = dataProcessor.get_train_attributes()
y_train = dataProcessor.get_train_labels()

#data for test
X_test = dataProcessor.get_test_attributes()
y_test = dataProcessor.get_test_labels()

# #Statified data for training
st_X_train = dataProcessor.get_stratified_train_attributes()
st_y_train = dataProcessor.get_stratified_train_labels()

#Statified data for testing
st_X_test = dataProcessor.get_stratified_test_attributes()
st_y_test = dataProcessor.get_stratified_test_labels()

#Scale data
normalized_X_train = filtering.scale_data(st_X_train)
normalized_X_test = filtering.scale_data(st_X_test)

#filtered X_train attributes
# filtered_data = filtering.PCA(X_train, 4)

# accuracy = algorithm.logistic_regression(logistic_regression_dict, normalized_X_train, normalized_y_train, st_X_test, st_y_test)
# result.log_result('logistic_regression', accuracy, logistic_regression_dict)

# ff_accuracy = algorithm.ff_network(3, X_train, y_train, X_test, y_test, dataProcessor)
# # accuracy = algorithm.SVM(SVM_dict, X_train, y_train, X_test, y_test)
# result.log_result('ff_network', ff_accuracy, ff_network_dict)

# accuracy = algorithm.k_neighbor(k_neighor_dict, X_train, y_train, X_test, y_test)
# result.log_result('k_neighbor', accuracy, k_neighor_dict)

accuracy = algorithm.XGBClassifier(XGBClassifier_dict, normalized_X_train, st_y_train, normalized_X_test, st_y_test)
result.log_result('XGBClassifier', accuracy, XGBClassifier_dict)

# accuracy = algorithm.decision_tree(decision_tree_dict, X_train, y_train, X_test, y_test)
# result.log_result('decision_tree', accuracy, decision_tree_dict)

# accuracy = algorithm.random_forest(random_forest_dict, normalized_X_train, st_y_train, st_X_test, st_y_test)
# result.log_result('random_forest', accuracy, random_forest_dict)

# accuracy = algorithm.ff_network(2, st_X_train, st_y_train, st_X_test, st_y_test)
# result.log_result('ff_network', accuracy, ff_network_dict)
