import pandas as pd
import numpy as np

import env
import os

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler



def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the sql database.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'



def get_zillow():
    '''
    This function pulls in the zillow dataframe from my sql query. I specified
    columns from sql to bring in.
    '''

    sql_query = '''
    SELECT *
    FROM properties_2017
    JOIN predictions_2017 USING (parcelid)
    INNER JOIN (SELECT parcelid, MAX(transactiondate) AS transactiondate
                FROM predictions_2017
                GROUP BY parcelid) 
                AS t USING (parcelid, transactiondate)
    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE properties_2017.latitude IS NOT NULL
    OR properties_2017.longitude IS NOT NULL
    AND propertylandusetypeid IN (260,261,262,263,264,266)
    AND transactiondate LIKE '2017%';
    '''
    if os.path.isfile('zillow_data.csv'):
            
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = pd.read_sql(sql_query, get_connection('zillow'))
        
        # Cache data
        df.to_csv('zillow_data.csv')

    return df



def get_info(df):
    '''
    This function takes in a dataframe and prints out information about the dataframe.
    '''

    print(df.info())
    print()
    print('------------------------')
    print()
    print('This dataframe has', df.shape[0], 'rows and', df.shape[1], 'columns.')
    print()
    print('------------------------')
    print()
    print('Null count in dataframe:')
    print('------------------------')
    print(df.isnull().sum())
    print()
    print('------------------------')
    print(' Dataframe sample:')
    print()
    return df.sample(3)


def value_counts(df, column):
    '''
    This function takes in a dataframe and list of columns and prints value counts for each column.
    '''
    for col in column:
        print(col)
        print(df[col].value_counts())
        print('-------------')


def visualize_numerals(df, column):
    '''
    This function takes in a dataframe and columns and creates a histogram with each column
    '''
    for col in column:
        plt.hist(df[col])
        plt.title(f"{col} distribution")
        plt.show()


def nulls_by_column(df):
    '''
    take in a dataframe 
    return a dataframe with each cloumn name as a row 
    each row will show the number and percent of nulls in the column
    
    '''
    
    # get columns paired with the number of nulls in that column
    num_missing = df.isnull().sum()
    
    # get percent of nulls in each column
    pct_missing = df.isnull().sum()/df.shape[0]
    
    # create/return dataframe
    return pd.DataFrame({'num_rows_missing': num_missing, 'pct_rows_missing': pct_missing})


def nulls_by_row(df):
    '''take in a dataframe 
       get count of missing columns per row
       percent of missing columns per row 
       and number of rows missing the same number of columns
       in a dataframe'''
    
    num_cols_missing = df.isnull().sum(axis=1) # number of columns that are missing in each row
    
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100  # percent of columns missing in each row 
    
    # create a dataframe for the series and reset the index creating an index column
    # group by count of both columns, turns index column into a count of matching rows
    # change the index name and reset the index
    
    return (pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index()
            .groupby(['num_cols_missing','pct_cols_missing']).count()
            .rename(index=str, columns={'index': 'num_rows'}).reset_index())



def show_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers and displays them
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
    



def remove_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers in variables and then removes them.
    recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
    
    
        df = df[(df[i] <= upper_bound) & (df[i] >= lower_bound)]
        print('-----------------')
    return print('Dataframe now has ', df.shape[0], 'rows and ', df.shape[1], 'columns')


def drop_nulls(df, prop_required_column = .5, prop_required_row = .5):
    ''' 
    take in a dataframe and a proportion for columns and rows
    return dataframe with columns and rows not meeting proportions dropped
    equate prop_required_column/prop_required_row to a percentage
    For Example:
    .5 for 50% of filled in data.
    Ex:
    handle_missing_values(df, .5, .5)
    '''
    col_thresh = int(round(prop_required_column*df.shape[0],0)) # calc column threshold
    
    df.dropna(axis=1, thresh=col_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    row_thresh = int(round(prop_required_row*df.shape[1],0))  # calc row threshhold
    
    df.dropna(axis=0, thresh=row_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    return df


def impute(df, strategy_method, column_list):
    ''' take in a df, strategy, and cloumn list
        return df with listed columns imputed using input stratagy
    '''
        
    imputer = SimpleImputer(strategy=strategy_method)  # build imputer

    df[column_list] = imputer.fit_transform(df[column_list]) # fit/transform selected columns

    return df


def clean_zillow(df):
    ''' Prepare Zillow Data
    '''
    
    # dropping the columns with 17K missing values too much to fill/impute/drop rows
    df = df.drop(columns=['heatingorsystemtypeid', 'buildingqualitytypeid', 'propertyzoningdesc', 'unitcnt', 'heatingorsystemdesc'])
    
    # imputing discrete columns with most frequent value
    df = impute(df, 'most_frequent', ['calculatedbathnbr', 'fullbathcnt', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock'])
    
    # imputing continuous columns with median value
    df = impute(df, 'median', ['finishedsquarefeet12', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount'])
    
    return df



def split_data(df):
    '''
    This function takes in a dataframe and splits it into train, test, and 
    validate dataframes for my model
    Do this only after you split to avoid data leakage
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('train--->', train.shape)
    print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, validate, test

