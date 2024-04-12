# -*- coding: utf-8 -*-
"""
Creating all functions which will be used in our machine learning workflow to 
train our model, perform exploratory data analysis, save our model, and perform
visualizations on the bean dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from typing import Union
import joblib
import time

# Creating functions
def eda(dataset: pd.DataFrame, graphs: bool = False) -> dict:
    """
    Perform exploratory data analysis on the dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset to perform EDA.
    graphs : bool, optional
        Choose to display exploratory data analysis visuals. The default is False.

    Returns
    -------
    dict
        A dictionary containing different evaluation metrics for exploring the 
        columns and understanding how values in the dataset are distributed.

    """
    data_unique = {}
    data_category_count = {}
    dataset.info()
    data_head = dataset.head()
    data_tail = dataset.tail()
    data_mode = dataset.mode().iloc[0]
    data_descriptive_stats = dataset.describe()
    data_more_descriptive_stats = dataset.describe(include = "all", 
                                                   datetime_is_numeric=True)
    data_correlation_matrix = dataset.corr(numeric_only = True)
    data_distinct_count = dataset.nunique()
    data_count_duplicates = dataset.duplicated().sum()
    data_count_null = dataset.isnull().sum()
    data_total_null = dataset.isnull().sum().sum()
    for each_column in dataset.columns: # Loop through each column and get the unique values
        data_unique[each_column] = dataset[each_column].unique()
    for each_column in dataset.select_dtypes(object).columns: 
        # Loop through the categorical columns and count how many values are in each category
        data_category_count[each_column] = dataset[each_column].value_counts()
        
    if graphs == True:
        # Visuals
        dataset.hist(figsize = (25, 20), bins = 10)
        plt.figure(figsize = (15, 10))
        sns.heatmap(data_correlation_matrix, annot = True, cmap = 'coolwarm')
        plt.show()
        plt.figure(figsize = (50, 30))
        sns.pairplot(dataset) # Graph of correlation across each numerical feature
        plt.show()
    
    result = {"data_head": data_head,
              "data_tail": data_tail,
              "data_mode": data_mode,
              "data_descriptive_stats": data_descriptive_stats,
              "data_more_descriptive_stats": data_more_descriptive_stats,
              "data_correlation_matrix": data_correlation_matrix,
              "data_distinct_count": data_distinct_count,
              "data_count_duplicates": data_count_duplicates,
              "data_count_null": data_count_null,
              "data_total_null": data_total_null,
              "data_unique": data_unique,
              "data_category_count": data_category_count,
              }
    return result


def build_regressor_model(regressor, 
                          x_train: pd.DataFrame, 
                          y_train: pd.DataFrame, 
                          x_test: pd.DataFrame, 
                          y_test: pd.DataFrame, 
                          kfold: int = 10):
    # Model Training
    model = regressor.fit(x_train, y_train)
    
    # Model Prediction
    y_pred = model.predict(x_train) # Training Predictions: Check OverFitting
    y_pred1 = model.predict(x_test) # Test Predictions: Check Model Predictive Capacity
    
    # Model Evaluation
    # Training Evaluation: Check OverFitting
    training_rsquared = r2_score(y_train, y_pred)
    training_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    
    # Test Evaluations: Check Model Predictive Capacity
    test_rsquared = r2_score(y_test, y_pred1)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred1))
    
    # Validation of Predictions
    cross_val = cross_val_score(model, x_train, y_train, cv = kfold)  
    cross_validation = cross_validate(model, 
                                      x_train, 
                                      y_train, 
                                      cv = kfold, 
                                      return_estimator = True,
                                      return_train_score = True)   
    score_mean = round((cross_val.mean() * 100), 2)
    score_std_dev = round((cross_val.std() * 100), 2)
    
    # Visualization
    # Visualising the actual testing data and predicted values
    plt.figure(figsize=(15, 10))
    plt.grid(True)
    plt.scatter(y_test, y_pred1, color='blue', alpha=0.5, label = {"Test RMSE": test_rmse})
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--c", label = {"Test R-Squared": test_rsquared})
    plt.title(f'Analyzing the Actual values against the Predicted Values - {regressor.__class__.__name__}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()
    
    return {
        "Model": model, 
        "Predictions": {
            "Actual Training Y": y_train, 
            "Actual Test Y": y_test, 
            "Predicted Training Y": y_pred, 
            "Predicted Test Y": y_pred1
            }, 
        "Training Evaluation": {
            "Training R2": training_rsquared, 
            "Training RMSE": training_rmse
            }, 
        "Test Evaluation": {
            "Test R2": test_rsquared, 
            "Test RMSE": test_rmse
            }, 
        "Cross Validation": {
            "Cross Validation Mean": score_mean, 
            "Cross Validation Standard Deviation": score_std_dev,
            "Validation Models": cross_validation
            }
        }


def build_multiple_regressors(regressors: Union[list or tuple], 
                              x_train: pd.DataFrame, 
                              y_train: pd.DataFrame, 
                              x_test: pd.DataFrame, 
                              y_test: pd.DataFrame, 
                              kfold: int = 10):
    multiple_regressor_models = {} # General store for all metrics from each algorithm
    store_algorithm_metrics = [] # Store all metrics gotten from the algorithm at each iteration in the loop below
    dataframe = pd.DataFrame(columns = ["Algorithm",
                                        "Fit time",
                                        "Score time",
                                        "Test score",
                                        "Train score"]) # Store cross validation metrics
    
    # Creating a dataframe for all classifiers
    # ---> Loop through each classifier ain classifiers and do the following
    for algorithms in regressors:
        store_cross_val_models = {}
        
        # Call the function build_classifier_model to get classifier metrics
        print(f"Building regressor model and metrics for {algorithms.__class__.__name__} model.")
        multiple_regressor_models[f"{algorithms.__class__.__name__}"] = build_regressor_model(regressor = algorithms, 
                                                                                              x_train = x_train, 
                                                                                              y_train = y_train, 
                                                                                              x_test = x_test, 
                                                                                              y_test = y_test, 
                                                                                              kfold = kfold)
        
        # Collecting individual metric to build algorithm dataframe
        training_r2 = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training R2"]
        training_rmse = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Training RMSE"]
        test_r2 = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test R2"]
        test_rmse = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Test RMSE"]
        cross_val_mean = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"]
        cross_val_std = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"]
        
        # Collecting indiviual metric to build cross validation dataframe
        cross_val_fit_time = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["fit_time"]
        cross_val_score_time = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["score_time"]
        cross_val_test_score = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["test_score"]
        cross_val_train_score = multiple_regressor_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["train_score"]
        
        # Storing all individual algorithm metrics from each iteration 
        store_algorithm_metrics.append([algorithms.__class__.__name__,
                                        training_r2,
                                        training_rmse,
                                        test_r2,
                                        test_rmse,
                                        cross_val_mean,
                                        cross_val_std])
        # Storing all individual cross validation metrics from each iteration 
        store_cross_val_models["Algorithm"] = algorithms.__class__.__name__
        store_cross_val_models["Fit time"] = cross_val_fit_time
        store_cross_val_models["Score time"] = cross_val_score_time
        store_cross_val_models["Test score"] = cross_val_test_score
        store_cross_val_models["Train score"] = cross_val_train_score
        # Creating dataframe for cross validation metric
        data_frame = pd.DataFrame(store_cross_val_models)
        dataframe = pd.concat([dataframe, data_frame])
        print("Model building completed.\n")
        
    # Creating dataframe for algorithm metric  
    df = pd.DataFrame(store_algorithm_metrics, columns = ["Algorithm",
                                                          "Training R2",
                                                          "Training RMSE",
                                                          "Test R2",
                                                          "Test RMSE",
                                                          "CV Mean",
                                                          "CV Standard Deviation"])
    # Save datasets in folder for analysis
    save_dataframe(dataset = dataframe, name = "Cross_Validation_Evaluation")
    save_dataframe(dataset = df, name = "Algorithm_Evaluation")
            
    return (df, dataframe, multiple_regressor_models)


def save_model_from_cross_validation(models_info: dict, algorithm: str, index: None):
    model_to_save = models_info[algorithm]["Cross Validation"]["Validation Models"]["estimator"][index]
    
    # Using Joblib to save the model in our folder
    joblib.dump(model_to_save, f"models/{algorithm}_Model_{index}.pkl")
    print(f"\nThis model is gotten from cross validating with the {algorithm} algorithm at iteration {index + 1}.")
    return models_info[algorithm]["Cross Validation"]["Validation Models"]["estimator"][index]


def save_dataframe(dataset: pd.DataFrame, name: str):
    """
    Save the data to the generated_data folder.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing the information we want to save. For this project,
        it could be a dataframe of algorithm metrics or cross validation metrics.
    name: str
        A string indicating the name of the dataset and how it should be saved.

    Returns
    -------
    None.

    """
    try:
        data_name = name 
        date = time.strftime("%Y-%m-%d")
        dataset.to_csv(f"../../Datasets/generated_data/section1/{data_name}_{date}.csv", index = True)
        print("\nSuccessfully saved file to the specified folder ---> generated_data folder.")
    except FileNotFoundError:
        print("\nFailed to save file to the specified folder ---> generated_data folder.")
                