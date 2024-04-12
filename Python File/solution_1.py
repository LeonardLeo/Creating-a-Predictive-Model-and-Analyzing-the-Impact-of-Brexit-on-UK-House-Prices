# -*- coding: utf-8 -*-


"""
You are required to carry out the following tasks:
a. Identify and address any issues in the dataset. Then conduct exploratory data
analysis on the dataset (10 marks)

b. Analyse the relationships between price and the other features. What are your
conclusions? (10 marks)

c. Analyse and visualise the impact of Brexit on house prices. What are your
conclusions? (10 marks)

d. Build a predictive model to estimate house prices. You need to show how to:
        • Use appropriate methods to select relevant features.
        • Split the dataset into training and testing sets.
        • Train at least 3 different prediction models.
        • Evaluate the models with the performance, and report in terms of R2 and
        RMSE.
        • Visualise the actual testing data and predicted values.
        • State your conclusions on the models.
        • Save your best model. 
"""

# Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from solution1_functions import (eda,
                                 build_multiple_regressors,
                                 save_model_from_cross_validation)
import joblib
from category_encoders import TargetEncoder

# Get the dataset
dataset = pd.read_csv("../../Datasets/Coursework Datasets/UK_Housing_Data.csv")



# QUESTION 1: Identify and address any issues in the dataset. Then conduct exploratory data
# analysis on the dataset

# SOLUTION:
# Exploratory data analysis
initial_eda = eda(dataset, graphs = True)

# Data cleaning and Transformation
    # --- Issues:
"""
1) We have 1828855 missing values initially in our data. After doing some initial 
data preprocessing, we are left with 23741 missing values in our data to handle.
--->
One possible fix is using the SimpleImputer from sklearn.impute or KNNImputer from sklearn.impute.
Another fix could be to drop missing rows.

2) TID (Transaction Identifier) has curly braces around each value.
--->
Remove them using .replace()

3) We have some irrelevant columns for the house prices analysis. Some of the irrelevant 
columns we have include:
        a) Unnamed: 0
        b) TID
        c) SAON
        d) PAON
        e) Record Status
The features SAON and PAON are considered irrelevant as the house number doesn't help us in 
predicting the price of a house, neither does the flat number. The SAON feature also has over 
a million rows missing in the dataset.

The feature TID (Transaction ID) is irrelevant for this analysis given it is just an 
identifier for the transactions.

The feature Record Status has one unique value (A), therefore, this won't be useful for 
gaining any insights given the absolute value is always (A) meaning variance is zero.
--->
Drop irrelevant columns using the .drop() pandas command.

4) TDate (Transaction Date) is given of type object.
--->
This is false and needs to be replaced with type datetime64.

5) Drop the LOCALITY column. More than half of the data is missing in that column.

6) Remove duplicate columns created after the above preprocessing steps. 2409 duplicated values 
were created that need to be dropped.
--->
Using the .drop_duplicates command solves this problem.

7) The TDate column needs to be processed to extract date features such as year, month, and day, 
for our model creation.

8) Drop categorical columns that aren't correlated with Price to avoid create a complex model and 
introducing noise into our model, further reducing our models predictive power.

9) Handling the categorical features for prediction. This process has a huge influence on the 
predictive power of our model, hence it is very crucial.
--->
One possible solution will be to use LabelEncoder from sci-kit learn or TargetEncoder from category_encoders.
"""

    # --- Solution:
# TID has curly braces --- Remove Them
dataset["TID"] = dataset["TID"].replace({"{": "",
                                          "}": ""})
# Drop irrelevant columns
dataset = dataset.drop(["TID",
                        "Unnamed: 0",
                        "SAON",
                        "PAON",
                        "RecordStatus"], axis = 1)

# Drop locality columns due to more than half of the data in the column missing
dataset = dataset.drop(["Locality"], axis = 1)

# Storing clean dataset for visualization
data = dataset

# Transaction date to datetime64
dataset["TDate"] = pd.to_datetime(dataset["TDate"])

# Drop duplicates
dataset = dataset.drop_duplicates()

# Fix missing values
dataset = dataset.dropna()

# Sort Price column
dataset = dataset.sort_values("TDate", ascending = True)

# Extract time features
dataset["Year"] = dataset["TDate"].dt.year
dataset["Month"] = dataset["TDate"].dt.month
dataset["Day"] = dataset["TDate"].dt.day

# Selecting dependent and independent variables
X = dataset.drop(["Price"], axis=1)
y = dataset.Price

# Encoding categorical features
encoder = TargetEncoder(smoothing = 50)
#encoder = ce.CatBoostEncoder(a = 3, drop_invariant = True)
X = encoder.fit_transform(X, y)



# QUESTION b: Analyse the relationships between price and the other features. What are your
# conclusions?
correlation_between_features = pd.concat([X, y], axis = 1).corr()

"""
CONCLUSIONS

From our analysis of the correlation matrix, insights show a positive linear relationship 
between postcode and price, as well as street and price. All other features don't have either 
no correlation with price or a relatively low correlation with price.

Including these uncorrelated columns into our model can introduce noise and lead to a complex 
model. The features postcode and street can be used to implement linear models with high 
accuracy given the approximate 0.97 and 0.67 positive relationship with price respectively.

These two features can succesfully capture the linear variations in price measured by r-squared.
"""

# Exploratory data analysis - After data cleaning and transformation
data_eda = eda(X)



# QUESTION c: Analyse and visualise the impact of Brexit on house prices. What are your
# conclusions?
# ---> PLEASE REFER TO THE JUPYTER NOTEBOOK FILE
# ---> PLEASE REFER TO THE FILE TITLED VISUALIZING_IMPACT_OF_BREXIT.PY
# ---> This is done to avoid congesting the entire workload in a single python file.

# Dropping the date column
X = X.drop("TDate", axis = 1)



# QUESTION d: Build a predictive model to estimate house prices. You need to show how to:
                        # • Use appropriate methods to select relevant features.
                        # • Split the dataset into training and testing sets.
                        # • Train at least 3 different prediction models.
                        # • Evaluate the models with the performance, and report in terms of R2 and
                        # RMSE.
                        # • Visualise the actual testing data and predicted values.
                        # • State your conclusions on the models.
                        # • Save your best model. 
                        
# Feature selection - To select best features
selector = SelectKBest(f_regression, k = 2)
X = selector.fit_transform(X, y)
    # Get feature importance
feature_importance = {feature: score for feature, score in zip(selector.feature_names_in_, selector.scores_)}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the X Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building
regressors = [LinearRegression(), 
              KNeighborsRegressor(), 
              SGDRegressor(loss='squared_error', 
                            penalty='l2',
                            alpha=0.0001,
                            fit_intercept=True,
                            max_iter=1000,
                            tol=1e-3,
                            learning_rate='optimal', 
                            early_stopping=False, 
                            validation_fraction=0.1,  
                            n_iter_no_change=5),
              Ridge(alpha = 8.0, random_state = 0),
              Lasso(alpha = 8.0, warm_start = True, precompute = True)]

# THE FUNCTION ---> build_multiple_regressors visualises the actual testing data and predicted values for each algorithm. 
algorithm_metrics, cross_validation_metrics, model_info = build_multiple_regressors(regressors, X_train, y_train, X_test, y_test)

average_fit_time = cross_validation_metrics["Fit time"].mean()     
average_score_time = cross_validation_metrics["Score time"].mean()  
              
joblib.dump(model_info, "model_info")

"""
CONCLUSIONS

Using the following algorithms, we create a predictive model for house prices in the United Kingdom.
    - Linear Regression
    - KNN Regression
    - Stochastic Gradient Descent Regression
    - Ridge Regression
    - Lasso Regression
We evaluate these algorithms with following metrics
    - Root Mean Squared Error (RMSE)
    - R-Squared (Coefficient of Determination)
    - Cross Validation Mean
    - Cross Validation Standard Deviation
    - Fit Time
    - Score Time
    
From our model building phase, Linear, Lasso, and Ridge regression perform best in training, testing, and have 
the best cross validation mean score of 92.49. In our analysis, we note the huge variation in the predictions of
the Stochastic Gradient Descent Regression as in it produces a test r-squared of -86.003 while having a cross validation
mean of 89.84. Therefore, it is not a reliable model. The KNN Regressor has the lowest R-Squared and Cross Validation mean
of 0.76 and 86.11 respectively. 

The Linear, Lasso, and Ridge Regression models have a Training R-Squared of 0.93 and a test R-Squared of 0.95 indcating 
how well it can generalize. When analysing their cross validation statistics, we see the best algorithm is the Lasso
regression as it records the lowest fit time and score time indicating how fast it is, while recording the highest
cross validation mean for test evaluation of 0.967182. This is slightly higher than the Ridge and Linear regression which
had similar cross validation test score of 0.967181. Regardless, Lasso regression remains our best model because it remains
the model with the best fit time and score time as well.

The model with the worst fit to our data with a test cross validation score of -124.536 is the Stochastic Gradient Descent
algorithm.
"""

# Saving the best model
save_model = save_model_from_cross_validation(models_info = model_info, 
                                              algorithm = "Lasso", 
                                              index = 9)

