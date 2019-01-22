import numpy as np
from util.Stat import *
from util.Distribution import Distribution
from data.Preprocessor import Preprocessor
from data.FeatureFilter import FeatureFilter
from learning.Models import Models

print('test began')

algorithm = Models()
dataProcessor = Preprocessor()
filtering = FeatureFilter()

#data for training
X_train = dataProcessor.get_train_attributes()
y_train = dataProcessor.get_train_labels()

X_test = dataProcessor.get_test_attributes()
y_test = dataProcessor.get_test_labels()

filtered_data = filtering.PCA(X_train)

algorithm.linear_SVM(X_train, y_train, X_test, y_test)
