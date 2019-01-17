# In a nutshell, this is what PCA is all about: Finding the directions of maximum variance in high-dimensional data 
# and project it onto a smaller dimensional subspace while retaining most of the information.import pandas as pd

df = pd.read_csv(file, sep=',')
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end
df.tail()
X = df.ix[:,0:4].values
y = df.ix[:,4].values

#normalisation
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)
