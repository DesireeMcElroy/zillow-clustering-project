## Project Goals
1. Import data from zillow database to wrangle and clean.
2. Utilize, data visualizations, statistical testing and clustering methods to find drivers of log error.
3. Create muotiple models that predict log error and compare to using the mean average of log error as prediction.

## Replicate my Project
    1. python
    2. pandas
    3. scipy
    4. sci-kit learn
    5. numpy
    6. matplotlib.pyplot
    7. seaborn
* Steps to recreate
    1. Clone this repository
    - https://github.com/DesireeMcElroy/zillow-clustering-project
    2. Import dataframe from SQL

## Key Findings
1. Home age and tax variable columns such as structure tax value were the highest drivers of log error.
2. County surprisingly did not have a major correlation to log error.
3. Bedrooms and bathrooms had no correlation with each other and were able to be separated and binned.
4. Models did not perform largely better than the baseline using average mean log error.

## Drawing Board
View my trello board [here](https://trello.com/b/zGrZv1t8/zillow-clustering-project).

------------

I want to examine these possibilities:
1. Does the log error increase as the number of bedrooms/bathrooms increase?
2. Does the log error increase as home size increases?
3. Does the log error have an association to the home's age?

-------

I will verify my hypotheses using statistical testing and where I can move forward with the alternate hypothesis, I will use those features in exploration. By the end of exploration, I will have identified which features are the best for my model.

During the modeling phase I will establish two baseline models and then use my selected features to generate a regression model. I will evaluate each model with the highest correlated features to minimize error and compare each model's performance to the baseline. Once I have selected the best modeling method, I will subject it to the training sample and evaluate the results.


## Data Dictionary

#### Target
Name | Description | Type
:---: | :---: | :---:
log_error | The variance amount the estimate was off from the actual price | float
#### Features
Name | Description | Type
:---: | :---: | :---:
num_bathrooms | The number of bathrooms a property has | float
num_bedrooms | The number of bedrooms a property has | float
total_lot_sqft | The square footage of the entire property | float
finished_sqft | The square footage of the footprint of the home | float
county | The county the property is located in | int
fips | The FIPS county code of the property location | float
tax_rate | Observation tax amount divided by the tax value  | float
latitude | The latitude location of the home | float
longitude | The longitude location of the home | float
price_per_sqft | The price of each square foot of the home | float
structure_tax_value | The tax value of the build of the home | float
home_age | The current year minus the build year | float
quadrimester | Calendar year divided into three | int
abs_logerr | The absolute value of log error | float

## Results
Model 3 using PolynomialRegression narrowly outperformed the baseline model:
1. Baseline model achieved root mean square error (RMSE) of .16 and .17 on train and validate test sets respectively.
2. Model 3 achieved RMSE of .15 and .17 on train and validate test sets respectively.
3. Model 3 resulted in RMSE of .14 on the unseen test data set. Arguably a good performance.



## Recommendations
1. I would conduct a lot more statistical testing to be sure I binned my data correctly.
2. Adjust the amount of my features used in my model to see which helps my model perform better.
4. Take a deeper look at additional features such as garages and pools to see if they help my model perform better.
5. Utilize absolute log error to use TweedieRegressor to asses whether that model may perform better.
6. Create multivariate project to predict how to better bin columns including the target log error to see if that may help the regression model perform better.


Resources:
https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt