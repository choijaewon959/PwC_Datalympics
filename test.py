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
from data.datehandler import *

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
# if len(sys.argv) < 2 :
#     print("Usage: Python test.py [~.csv] ")
#     exit()
#
# print (sys.argv[1])

#Retrieve and Save test data
print("Reading Test case data...")
#data = pd.read_csv(sys.argv[1])

data = pd.read_csv("../data/InvoicePayment-evaluation.csv")

"""
The data given is supposed to have attributes of Training dataset
Labels are erased or omitted.

:format: pandas Datafrmae
"""
origin= data.copy()
data= datetime_data(data)

file = open("Featurelist.txt", "r")

cols= pd.read_csv("Featurelist.csv")
li= list(cols.columns)


cleaned_data, answer= clean_data(data,li)
#load the trained data.
trainedModel1 = pickle.load(open(MODELFILE1, 'rb'))
trainedModel2 = pickle.load(open(MODELFILE2, 'rb'))
trainedModel3 = pickle.load(open(MODELFILE3, 'rb'))

#New object to evaluate the given dataset

eval = ModelEvaluation(cleaned_data)
eval.run_model(trainedModel1)

evaluated_result = eval.get_predicted_label()

#print("1st classification:", evaluated_result)

#Add new column with predicted data
cleaned_data['label']=evaluated_result

#print(data.dtypes)
#origin['First_classification']= evaluated_result

data_labeled =cleaned_data

#Gather data with specific label
early_rows = get_specific_label(data_labeled,0)
#drop predicted column because model cannot use the column
early_rows= early_rows.drop('label', axis=1 )
#############################################################

#Second classification model with grouped data
eval2 = ModelEvaluation(early_rows)
eval2.run_model(trainedModel2)

evaluated_result2 = eval2.get_predicted_label()
#print("1st evaluated_result2:", evaluated_result2)

early_rows['Second_classification']=evaluated_result2
#print(early_rows)

#print("data labeled:: " , data_labeled)
#Gather data with specific label
late_rows = get_specific_label(data_labeled,2)

#drop predicted column because model cannot use the column
late_rows= late_rows.drop('label', axis=1 )

#Second classification model with grouped data
eval3 = ModelEvaluation(late_rows)
eval3.run_model(trainedModel3)
evaluated_result3 = eval3.get_predicted_label()
late_rows['Third_classification']=evaluated_result3

cleaned_data['Third_classification']=evaluated_result3
finalLabel = finalize_label(evaluated_result,evaluated_result2,evaluated_result3)


early_nodes, late_nodes = generate_node(data)
print(early_nodes,late_nodes)
final_result(data,finalLabel,early_nodes,late_nodes)
print(data['PaymentDueDate'])
result2csv(data)
print(csv2score('HKU_KD_result.csv',origin))