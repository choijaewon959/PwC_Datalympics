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

#filtered X_train attributes
filtered_data = filtering.PCA(X_train, 4)

accuracy = algorithm.logistic_regression(logistic_regression_dict, X_train, y_train, X_test, y_test)
result.log_result('logistic_regression', accuracy, logistic_regression_dict)
print(result.get_best())

