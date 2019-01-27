import numpy as np
import pandas as pd
import sys
import pickle

from data.Preprocessor import Preprocessor
from data.MiniProcessor import MiniProcessor
from data.FeatureFilter import FeatureFilter
from learning.Hyperparameter import *
from learning.Models import Models
from evaluation.ResultLog import ResultLog
from evaluation.ModelEvaluation import ModelEvaluation
from data.MiniProcessor import MiniProcessor

from config import *

'''
Test script for scoring.
1. Gets the raw data to be tested.
2.
3.
4.d
'''

'''
Data to be tested.
Assumed data would be without label value.
'''
data = pd.read_csv(sys.argv[1]) #argument must be the file path of raw csv file.

#Objects to be used.
algorithm = Models()
miniProcessor = MiniProcessor(data)
evaluation = ModelEvaluation(data)

'''
First classification.
Giving the virtual labels to the data.
'''
#load the trained data.
trainedModel1 = pickle.load(open(MODELFILE1, 'rb')) #trained model for first classification.
evaluation.run_model(trainedModel1)




