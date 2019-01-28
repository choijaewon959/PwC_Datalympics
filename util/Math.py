import numpy as np
import pandas as pd
import math
def mse(predictionData, actualData):
    '''
    Mean squared error 
    
    Calcuate mean square error from predicted data array and actual data array.
    Two arrays must have same length of data.
    :param: float arr, float arr
    :return: float
    '''
    dataLength = len(predictionData)
    sqrdError = 0

    for i in range(dataLength):
        sqrdError += (predictionData[i] - actualData[i]) ** 2

    return sqrdError / dataLength
    
def create_class_weight(transactionData,mu=0.9):
    total = transactionData.shape[0]
    labels_dict = pd.value_counts(transactionData)
    class_weight = dict()
    for num ,tmp in zip(labels_dict, [0,2,3,4,5,6,7,8,9]):
        sco = math.log(mu*total/float(num))
        class_weight[tmp] = sco if sco > 1.0 else 1.0
        tmp = tmp + 1
    # for i in range(0,3):
    #     if i ==4 :
    #         class_weight[i] += 20
    #         continue
    #     if i ==8: 
    #         class_weight[i] += 30
    #         continue
    #     if i ==9: 
    #         class_weight[i] += 30
    #         continue
    #     class_weight[i] += 50
    return class_weight
