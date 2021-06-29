import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def create_cluster(train_scaled, X, k, cluster_name):
    ''' Takes in df, X (dataframe with variables you want to cluster on), k number of clusters,
    and the name you want to name the column (enter column as string)
    It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    the scaler and kmeans object and unscaled centroids as a dataframe
    note: train_scaled enter the scaled train dataframe
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


def rmse(algo, X_train, X_validate, y_train, y_validate, target, model_name):
    '''
    This function takes in an algorithm name, X_train, X_validate, y_train, y_validate, target and a model name
    and returns the RMSE score for train and validate dataframes.
    '''

    # enter target and model_name as a string
    # algo is algorithm name, enter with capitals for print statement
    
    # fit the model using the algorithm
    algo.fit(X_train, y_train[target])

    # predict train
    y_train[model_name] = algo.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train[target], y_train[model_name])**(1/2)

    # predict validate
    y_validate[model_name] = algo.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate[target], y_validate[model_name])**(1/2)

    print("RMSE for", model_name, "using", algo, "\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    print()
    
    return rmse_train, rmse_validate