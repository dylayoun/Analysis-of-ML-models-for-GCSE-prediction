# Analysis-of-ML-models-for-GCSE-prediction

## Project Overview

This project develops and evaluates machine learning models for predicting student GCSE outcomes in Maths and English using anonymised data from a secondary school trust along with 2 other free schools. Four regression models are implemented and compared:
- Linear Regression
- Decision Tree
- Random Forest
- Support Vector Regression

This project also includes a preprocessing pipeline with a systematic imputation strategy for handling missing data, and an evaluation framework using MAE, RMSE and R^2 metric

## Libraries

**Python Version:** 3.11.5

The following external libraries are required to run this project. All dependencies are imported at the top of each file.

- pandas
    - openpyx1
- numpy
- scikit-learn: v1.6.1
- matplotlib


There are 3 python files which are used:

**preprocessing.py** 
This takes in the raw data from csv file and completes all necessary preprocessing outputting training and test sets for both English and Maths. The subjects are split to allow for different target variable but identical otherwise

**optimisation.py**
This holds the testing for imputation strategies. It has 4 different strategies which were tested using the cleaned data and returns MAE scores for each. This guided preprocessing for the imputation of data for missing values. It is **NOT** required to reproduce the final results and is provided for reference only. Ultimately, the final imputation strategy chosen was *Cohort Median* 

**models.py** 
This is the final generation of models and evaluation. It has two main functions which the first completes validation using RandomCV() to find the optimised hyperparameters. From this, the best_model() function creates the best models with helper function creating graphs and charts from the results of evaluation. 

**Order of Execution**
The files should be run in order of *preprocessing.py* then optionally *optimisiation.py* and finally *models.py*. Without the outputs of preprocessing.py, models.py will fail to run.

# Findings

- **Best Model:** Random Forest performed best overall, with Maths predictions consistently more accuracte than English predictions for all models
- **Maths Performance:** The best model achieved an MAE of **0.514** and an R^2 of **0.889** which was an 18% improvement over the Linear Regression baseline.
- **Key Predictors:** Y11 Spring Mock Performance was the strongest predictor of GCSE Maths Results accounting for roughly **68%** of feature importance
- **Model Comparison:** Random Forest outperformed Linear Regression, Decision Tree and SVR with Maths predictions showing tighter residuals and fewer larger errors than English predictions
- **Interpretations:** The results should be taken with a grain of salt as predictions are dependent on school data and cannot contain every factor involved with student outcomes. 


