import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def create_cluster(train_scaled, X, k, cluster_name):
    ''' Takes in df, X (dataframe with variables you want to cluster on), k number of clusters,
    and the name you want to name the column (enter column as string)
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe
    train_scaled enter the scaled train dataframe
    for X enter the dataframe of the two features for your cluster
    for k enter number of features
    for cluster_name enter name of the cluster column name you want as a string
    '''
    scaler = MinMaxScaler(copy=True).fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X)
    kmeans.predict(X)
    train_scaled[cluster_name] = kmeans.predict(X)
    # train_scaled[cluster_name] = 'cluster_' + train_scaled[cluster_name].astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
    return train_scaled, X, scaler, kmeans, centroids


def create_cluster_scatter_plot(x, y, train_scaled, X, kmeans, scaler, cluster_name):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = train_scaled, hue = cluster_name)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')