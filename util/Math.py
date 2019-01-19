import numpy as np

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
    
    