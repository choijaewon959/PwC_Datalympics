import numpy as np
import pandas as pd
# from util.Stat import *
# from util.Distribution import Distribution
# from data.Preprocessor import Preprocessor
# from data.MiniProcessor import MiniProcessor
# from data.FeatureFilter import FeatureFilter
# from learning.Hyperparameter import *
# from learning.Models import Models
# from evaluation.ResultLog import ResultLog
from evaluation.ModelEvaluation import ModelEvaluation
from evaluation.Visualization import Visualization
from data.Cleaningdata import *

import pickle
from config import *
import sys

"""
test.py

Testing file

1. Parse input parameter from commandline
1a. Load the testing file

2. Load Trained models
3. Evaluate the test file with loaded model
4. Save the evaluated data onto the original testing file
5. With the result of 3, load specific labels to each dataframe
6. Run second model for each labels
7. Write results while joining with first classification

8. Export result(Prediction) to csv file


"""
#Commandline Parse input
if len(sys.argv) < 2 :
    print("Usage: Python test.py [~.csv] ")
    exit()

print (sys.argv[1])

#Retrieve and Save test data
print("Reading Test case data...")
data = pd.read_csv(sys.argv[1])
"""
The data given is supposed to have attributes of Training dataset
Labels are erased or omitted.

:format: pandas Datafrmae
"""
origin= data.copy()

data= clean_data(data)

file = open("Featurelist.txt", "r")

cols= file.read()
print(cols)

#load the trained data.
trainedModel1 = pickle.load(open(MODELFILE1, 'rb'))
trainedModel2 = pickle.load(open(MODELFILE2, 'rb'))

#New object to evaluate the given dataset
eval = ModelEvaluation(data)
eval.run_model(trainedModel1)
evaluated_result = eval.get_predicted_label()

print( evaluated_result)

#Add new column with predicted data
data['label']=evaluated_result
origin['First_classification']= evaluated_result

#Gather data with specific label
data = get_specific_label(data,3)

#drop predicted column because model cannot use the column
data= data.drop('loan_status', axis=1 )
print(data)

#Second classification model with grouped data
eval2 = ModelEvaluation(data)
eval2.run_model(trainedModel2)
evaluated_result2 = eval2.get_predicted_label()
data['Second_classification']=evaluated_result2
origin['Second_classification']=evaluated_result2
#combine both
print(origin)
exit()


origin.to_csv("Prediction.csv", sep=',')
