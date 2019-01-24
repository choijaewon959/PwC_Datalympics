'''
Feature filtering object for reducing the dimensionality of the feature vectors
'''
import pandas as pd
from util.Distribution import Distribution
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

class FeatureFilter:
    def __init__(self):
        self.__dataFrame = None # data frame with features and values
        self.__meaingfulFeatures = set() # features to be used after filtering. (str)
        self.__reducedDimension = 0 # number of meaningful feature vectors

    def PCA(self, raw_attributes, numOfComponents):
        '''
        Principal component analysis.
        Filter algorithm to retrieve meaningful feature attributes.
        
        :param: 
            raw_attributes: unfiltered data
            numOfComponents: number of components to be used.
        
        :return: filtered feature vector (data frame)
        '''
        pca = PCA(n_components = numOfComponents)
        keyFeaturesVector = pca.fit_transform(raw_attributes)
        return keyFeaturesVector

    def get_reduced_dimension(self):
        '''
        Return the reduced dimension of reduced feature vectors

        :param: None
        :return: dimensionality of feature vector (int)
        '''
        return self.__reducedDimension

    def scale_data(self, X_train, y_train):
        '''
        Normalize data.

        :param: data to be normalized. (Data frame)
        :return: nomalized data. (Data frame)
        '''
        scaling = preprocessing.MinMaxScaler(feature_range=(-1,1))
        X_train_normalized = pd.DataFrame(scaling.fit_transform(X_train))
        y_train_normalized = pd.Series(scaling.fit_transform(y_train.reshape(-1,1)))

        return X_train_normalized, y_train_normalized