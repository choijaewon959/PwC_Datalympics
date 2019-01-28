import numpy as np
import pickle
from pandas import DataFrame
from data.Preprocessor import Preprocessor
from data.MiniProcessor import MiniProcessor
from data.FeatureFilter import FeatureFilter
from learning.Hyperparameter import *
from learning.Models import Models
from evaluation.ResultLog import ResultLog
from evaluation.ModelEvaluation import ModelEvaluation
from Submission import Submission

from sklearn.metrics import accuracy_score, auc, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from config import *

print('test began')

#objects
dataProcessor = Preprocessor()
transactionData = dataProcessor.get_data() #original given data
algorithm = Models()
result = ResultLog()
miniProcessor = MiniProcessor(transactionData)
submission = Submission()

#data for training
X_train = dataProcessor.get_train_attributes()
y_train = dataProcessor.get_train_labels()

#data for test
X_test = dataProcessor.get_test_attributes()
y_test = dataProcessor.get_test_labels()

print(X_train)
print(X_test)
# accuracy = algorithm.logistic_regression(logistic_regression_dict, X_train, y_train, X_test, y_test)
# result.log_result('logistic_regression', accuracy, logistic_regression_dict)

# ff_accuracy = algorithm.ff_network(3, X_train, y_train, X_test, y_test)
# # accuracy = algorithm.SVM(SVM_dict, X_train, y_train, X_test, y_test)
# result.log_result('ff_network', ff_accuracy, ff_network_dict)
#
# accuracy = algorithm.k_neighbor(k_neighor_dict, X_train, y_train, X_test, y_test)
# result.log_result('k_neighbor', accuracy, k_neighor_dict)

'''
First Learning to classify the rows into early, ontime, late
'''

trainedModel = algorithm.decision_tree(decision_tree_dict,X_train, y_train, X_test, y_test)

#save trained model
pickle.dump(trainedModel, open(MODELFILE1, 'wb'))

evaluation = ModelEvaluation(X_test, y_test, X_train, y_train)
accuracy_first = evaluation.evaluate_model(trainedModel)
#result.log_result(trainedModel1[0], accuracy_first, XGBClassifier_dict)

y_predicted = evaluation.get_predicted_label()

#TODO: convery y value to string
#log the submission result.
# submission.update_paymentTiming(y_predicted)    # update the timing value to csv. (early, ontime, late)
# submission.update_PwC_RowID(y_predicted)    # update the row ID.

'''
Learning for data with early paid label.
Specifically predict the date the user with 'early paid' would pay.
'''
#TODO:  change the n value to retrieve the data.
#Retrieve the data to be used for early paid training.
early_paid_Data = miniProcessor.get_second_data(0)

#Retrieve the trained model.
early_paid_trainedModel = algorithm.decision_tree(decision_tree_dict, early_paid_Data[2], early_paid_Data[3], early_paid_Data[0], early_paid_Data[1])

#save early paid trained model.
pickle.dump(early_paid_trainedModel, open(MODELFILE2, 'wb'))

#Evaluate early-paid model. (Second classification)
early_paid_evaluation = ModelEvaluation(early_paid_Data[0],early_paid_Data[1],early_paid_Data[2],early_paid_Data[3])
accuracy_early = early_paid_evaluation.evaluate_model(early_paid_trainedModel)

early_paid_y_predicted = early_paid_evaluation.get_predicted_label()

'''
Learning for data with late paid label.
Specifically predict the date the user with 'late paid' would pay.
'''
#TODO:  change the n value to retrieve the data.
#Retrieve the data to be used for early paid training.
late_paid_Data = miniProcessor.get_second_data(2)

#Retrieve the trained model.
late_paid_trainedModel = algorithm.decision_tree(decision_tree_dict, late_paid_Data[2], late_paid_Data[3], late_paid_Data[0], late_paid_Data[1])

#save early paid trained model.
pickle.dump(late_paid_trainedModel, open(MODELFILE3, 'wb'))

#Evaluate early-paid model. (Second classification)
late_paid_evaluation = ModelEvaluation(late_paid_Data[0],late_paid_Data[1],late_paid_Data[2],late_paid_Data[3])
accuracy_late = late_paid_evaluation.evaluate_model(late_paid_trainedModel)

late_paid_y_predicted = late_paid_evaluation.get_predicted_label()

#log the payment timing (early, on time, late) result.
submission.update_paymentTiming(y_predicted)

# '''
# Change the virtual label into more specific label.
# '''
# finalEval = miniProcessor.finalize_label(y_predicted, early_paid_y_predicted, late_paid_y_predicted)
# print(finalEval.unique())

# accuracy = accuracy_score(np.array(y_test).flatten(), finalEval)
# print(accuracy)

# accuracy = algorithm.XGBClassifier(XGBClassifier_dict, X_train, y_train, X_test, y_test)
# result.log_result('XGBClassifier', accuracy, XGBClassifier_dict)
#
# accuracy = algorithm.decision_tree(decision_tree_dict, X_train, y_train, X_test, y_test)
# result.log_result('decision_tree', accuracy, decision_tree_dict)
#
# accuracy = algorithm.random_forest(random_forest_dict, X_train, y_train, X_test, y_test)
# result.log_result('random_forest', accuracy, random_forest_dict)

# for i in range(1,4):
#     accuracy = algorithm.ff_network(i, X_train, y_train, X_test, y_test)
#     result.log_result('ff_network', accuracy, ff_network_dict)
