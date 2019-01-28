'''
Feature filtering object for reducing the dimensionality of the feature vectors
'''
import pandas as pd
from util.Distribution import Distribution
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


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

    def feature_score(self, data):
        # for column in data.columns:
        #     s= data[column]
        #     print(any(s<0))
        # print(data[(data[0,27] < 0).all(1)])
        dfTrain = data
        dfTrain['InvoiceAmount'] = dfTrain['InvoiceAmount'].abs()
        dfTrain['duration'] = dfTrain['duration'].abs()

        X = dfTrain.drop(['label','difference'], axis = 1)
        y = dfTrain['label']

        #apply SelectKBest
        bestfeatures= SelectKBest(score_func=chi2, k=20)
        fit = bestfeatures.fit(X,y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns= pd.DataFrame(X.columns)

        #concatenate two datafrmae for Visualization
        feature_score = pd.concat([dfcolumns, dfscores], axis=1)
        feature_score.columns = ['Features', 'Score']
        #print(feature_score.nlargest(20, 'Score'))
        print("20 features are selected and saved to Featurelist.txt")
        #plt.show()
        #print(feature_score.nlargest(20,'Score')['Features'].tolist())
        feature_score.nlargest(20,'Score')['Features'].to_csv("Featurelist.csv",sep=',',index=False,
        header="feature")

        #text_file.write("feature_score by SELECTKBEST :\n")
        # text_file.write(str(feature_score.nlargest(20,'Score')['Features'].tolist()))
        #
        # text_file.close()


        return feature_score.nlargest(20,'Score')['Features'].tolist()


    def feature_importance(self, data):
        X = data.drop('loan_status', axis = 1)
        y = data['loan_status']

        ETC = ExtraTreesClassifier()
        ETC.fit(X,y)
        print (ETC.feature_importances_)
        # dfscores=ETC.feature_importances_
        # dfcolumns= pd.DataFrame(X.columns)
        #
        # feature_importance = pd.concat([dfcolumns, dfscores], axis=1)
        # feature_importance.columns = ['Features', 'importance']
        # feature_importance.nlargest(20).plot(kind='barh')
        # plt.show()
        #
        # return feature_score.nlargest(20,'Score')['Features'].tolist()

        feature_importance= pd.Series(ETC.feature_importances_, index = X.columns )

        text_file = open("Featurelist.txt", "w")

        #text_file.write("feature_score by SELECTKBEST :\n")
        text_file.write(str(feature_importance.nlargest(20).index.tolist()))

        text_file.close()
        print("20 features are selected and saved to Featurelist.txt")

        # feature_importance.nlargest(20).plot(kind='barh')
        # print(feature_importance.nlargest(20).index.tolist())
        # plt.show()
        # #print(feature_importance.nlargest(20).index.tolist())
        return feature_importance.nlargest(20).index.tolist()

    def dominant_feature_filter(self, data):
        '''
        Filter out the data rows with the dominant label.

        :param: None
        :return: filtered data (data frame)
        '''

        df = data.loc[data['loan_status'] != 'Current']
        return df
