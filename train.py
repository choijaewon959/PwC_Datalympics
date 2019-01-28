import numpy as np
from data.Preprocessor import Preprocessor
from data.MiniProcessor import MiniProcessor
from data.FeatureFilter import FeatureFilter
from learning.Hyperparameter import *
from learning.Models import Models
from evaluation.ResultLog import ResultLog
from evaluation.ModelEvaluation import ModelEvaluation
from sklearn.metrics import accuracy_score, auc, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from config import *
import pickle

print('test began')

#objects
dataProcessor = Preprocessor()
loanData = dataProcessor.get_data() #original given data
dataProcessor.classify_label()
algorithm = Models()
result = ResultLog()
miniProcessor = MiniProcessor(loanData)
#data for training
X_train = dataProcessor.get_train_attributes()
y_train = dataProcessor.get_train_labels()

# loanData = dataProcessor.get_data() #original given data
# algorithm = Models()
# result = ResultLog()
# miniProcessor = MiniProcessor(loanData)
# #data for training
# X_train = dataProcessor.get_train_attributes()
# y_train = dataProcessor.get_train_labels()
#
# # #data for training
# # X_train = dataProcessor.get_train_attributes()
# # y_train = dataProcessor.get_train_labels()
#
# #data for test
# X_test = dataProcessor.get_test_attributes()
# y_test = dataProcessor.get_test_labels()

# accuracy = algorithm.logistic_regression(logistic_regression_dict, X_train, y_train, X_test, y_test)
# result.log_result('logistic_regression', accuracy, logistic_regression_dict)

# ff_accuracy = algorithm.ff_network(3, X_train, y_train, X_test, y_test)
# # accuracy = algorithm.SVM(SVM_dict, X_train, y_train, X_test, y_test)
# result.log_result('ff_network', ff_accuracy, ff_network_dict)

# accuracy = algorithm.k_neighbor(k_neighor_dict, X_train, y_train, X_test, y_test)
# result.log_result('k_neighbor', accuracy, k_neighor_dict)

# trainedModel1 = algorithm.XGBClassifier(XGBClassifier_dict, X_train, y_train, X_test, y_test)
#
# #save trained model
# pickle.dump(trainedModel1, open(MODELFILE1, 'wb'))
#
# evaluation = ModelEvaluation(X_test, y_test, X_train, y_train)
# accuracy_first = evaluation.evaluate_model(trainedModel1)
# #result.log_result(trainedModel1[0], accuracy_first, XGBClassifier_dict)
#
# y_predicted = evaluation.get_predicted_label()
#
# #Second learning for more specific labels.
# newData = miniProcessor.get_second_data(3)
#
# trainedModel2 = algorithm.XGBClassifier(XGBClassifier_dict2, newData[2], newData[3], newData[0], newData[1])
#
# #save trained model
# pickle.dump(trainedModel2, open(MODELFILE2, 'wb'))
#
# #   evaluate second model
# evaluation2 = ModelEvaluation(newData[0],newData[1],newData[2],newData[3])
# accuracy_second = evaluation2.evaluate_model(trainedModel2)
#
# y_predicted2 = evaluation2.get_predicted_label()
#
# finalEval = miniProcessor.finalize_label(y_predicted, y_predicted2)
# print(finalEval.unique())
#
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
