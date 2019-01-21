import numpy as np
from util.Stat import *
from util.Distribution import Distribution
from data.Preprocessor import Preprocessor
from data.FeatureFilter import FeatureFilter
from learning.Models import Models

print('test began')

algorithm = Models()
dataprocess = Preprocessor()
filtering = FeatureFilter()


raw_data = dataprocess.get_train_attributes()
filtered_data = filtering.PCA(raw_data)

algorithm.gaussian_SVM(filtered_data)

