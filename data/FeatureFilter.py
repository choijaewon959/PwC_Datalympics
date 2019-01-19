'''
Feature filtering object for reducing the dimensionality of the feature vectors
'''
import pandas as pd
from util.Distribution import Distribution
from data.FileManager import FileManager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

class featureFilter:
    def __init__(self, dataFrame):
        self.__dataFrame = dataFrame # data frame with features and values
        self.__meaingfulFeatures = set() # features to be used after filtering. (str)
        self.__reducedDimension = 0 # number of meaningful feature vectors

    def PCA(self):


    def get_reduced_dimension(self):
        '''
        Return the reduced dimension of reduced feature vectors

        :param: None
        :return: dimensionality of feature vector (int)
        '''
        return self.__reducedDimension
