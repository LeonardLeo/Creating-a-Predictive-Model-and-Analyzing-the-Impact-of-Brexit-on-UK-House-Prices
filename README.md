# UK House Prices Predictive Model and Brexit Impact Analysis

This project involves creating a predictive model to estimate UK house prices and analyzing the impact of Brexit on these prices.

# Project Summary:

The project aimed to understand the impact of Brexit on UK house prices through data analysis and predictive modeling. It involved data preprocessing to handle missing values and irrelevant columns, followed by exploratory data analysis to gain insights into the dataset. The relationships between house prices and other features were analyzed, and the impact of Brexit was visualized using both yearly and monthly average house prices. Additionally, predictive models were built using various algorithms, and their performance was evaluated based on R2 and RMSE. The best performing model, Lasso Regression, was saved for future use.

## Files Included:

- `UK_Housing_Data.csv`: Dataset containing UK house prices.
- `visualizing_impact_of_brexit.py`: Python script for visualizing the impact of Brexit on house prices.
- `model_building_and_evaluation.py`: Python script for building and evaluating predictive models.

## Project Structure:

- `Data Preprocessing`: Handling missing values, removing irrelevant columns, and encoding categorical features.
- `Exploratory Data Analysis (EDA)`: Analyzing relationships between house prices and other features.
- `Brexit Impact Analysis`: Visualizing the impact of Brexit on UK house prices.
- `Model Building`: Building predictive models using various algorithms.
- `Model Evaluation`: Evaluating model performance using R2 and RMSE.

## Dependencies:

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- category_encoders

## Instructions:

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the scripts to perform data analysis and model building.

## Recommendations for Future Work:

While this project provides a deep analysis of the impack of Brexit on Uk house prices, we could always improve our model and analysis by:
- Further analyzing the impact of specific Brexit events on house prices.
- Exploring additional features that might influence house prices.
- Experimenting with different machine learning algorithms and hyperparameters.
- Implementing time series analysis techniques for forecasting house prices.
- Considering incorporating external datasets for a more comprehensive analysis.
