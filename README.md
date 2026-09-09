# Analysis of ML Models for GCSE Prediction

## Project Overview

This project develops and evaluates machine learning models for predicting student GCSE outcomes in Maths and English using anonymised data from 9 UK schools. Four regression models are implemented and compared:
- Linear Regression
- Decision Tree
- Random Forest
- Support Vector Regression

This project also includes a preprocessing pipeline with a systematic imputation strategy for handling missing data, and an evaluation framework using MAE, RMSE and R² metric

The aim was to compare different machine learning approaches and identify which factors were most useful for predicting GCSE outcomes.

## Findings

- **Best Model:** Random Forest performed best overall, with Maths predictions consistently more accurate than English predictions for all models
- **Maths Performance:** The best model achieved an MAE of **0.514** and an R^2 of **0.889** which was an 18% improvement over the Linear Regression baseline.
- **Key Predictors:** Y11 Spring Mock Performance was the strongest predictor of GCSE Maths Results accounting for roughly **68%** of feature importance
- **Model Comparison:** Random Forest outperformed Linear Regression, Decision Tree and SVR with Maths predictions showing tighter residuals and fewer larger errors than English predictions
- **Interpretation:** The results should be taken with a grain of salt as predictions are dependent on school data and cannot contain every factor involved with student outcomes. 

## Dataset
The original dataset is not included in this repository due to data privacy and data-sharing restrictions.

The dataset consisted of anonymised student-level data from 9 UK schools, covering the 2023 and 2024 cohorts, with approximately 2,700 student records used in the modelling.

The main categories of variables included:

| Category | Examples |
|---|---|
| GCSE outcomes | GCSE Maths and English results |
| Demographics | Age, Gender, Ethnicity |
| Student Characteristics | FSM, PP, EAL, SEND |
| Attendance and Behaviour | Attendance, Suspensions |
| Prior Attainment | KS2 Reading, KS2 Maths |
| School Assesments | GL assessments |
| GCSE Preparation | Y10 and Y11 Mock results |


## Methodology

The project followed a structured machine learning pipeline:

1. **Data Preprocessing**
    - Data cleaning and preparation
    - Categorical variable encoding
    - Missing Value imputation
    - Feature scaling

2. **Train/Test Split**
    - Data was split into **85%** training and **15%** testing.
    - Preprocessing parameters were fitted using only training data to prevent data leakage

3. **Model Development**
    - Linear Regression
    - Decision Tree
    - Random Forest
    - Support Vector Regression (SVR)

4. **Hyperparameter Optimisation**
    - RandomizedSearchCV was used for model hyperparameter optimisation
    - **5-fold cross-validation** was performed on training data

5. **Model Evaluation**
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R²
    - Residual analysis
    - Feature importance analysis


## Project Structure

```text
Analysis-of-ML-models-for-GCSE-prediction/
├── Data/
│   └── README.md
├── Figures/
├── Source_Code/
│   ├── preprocessing.py
│   ├── optimisation.py
│   └── models.py
├── .gitignore
├── README.md
└── requirements.txt
```

### `Source_Code/preprocessing.py`

Handles data cleaning and preprocessing, including categorical encoding, missing-value imputation and feature scaling. Produces the training and test datasets used by the modelling stage.

### `Source_Code/optimisation.py`

Contains experiments comparing different missing-value imputation strategies. This was used to determine the final preprocessing approach and is not required to reproduce the final model results.

### `Source_Code/models.py`

Trains and evaluates the machine learning models, performs hyperparameter optimisation and generates evaluation visualisations.

### `Figures/`

Contains visualisations generated during model evaluation, including model performance, residuals, actual vs predicted values and feature importance.

### `Data/`

Contains documentation describing the dataset. The original dataset is not included in the repository due to data privacy and data-sharing restrictions.

## Installation and Usage

### Requirements

- Python 3.11.5

Install the required dependencies using:
pip install -r requirements.txt

## Running the project

The scripts should be run in the following order:
1. `preprocessing.py`
2. `optimisation.py` **(optional)**
3. `models.py`

`optimisation.py` contains experiments used to determine the final imputation and so is not required to reproduce model results

The dataset is not included in this repo so the input data **must** be provided separately.


## Results & Visualisations

### Maths — Random Forest

The final Random Forest model achieved an MAE of **0.514** and an R² of **0.889** for GCSE Maths results.

#### Actual vs Predicted

The actual and predicted values show how closely the model's predictions matched the actual GCSE Maths results.

![Maths Random Forest Actual vs Predicted](Figures/Maths%20Data_Random%20Forest_actual_vs_predicted.png)

#### Feature Importance

Feature importance analysis showed that Year 11 Spring Mock performance was the strongest predictor of GCSE Maths results, accounting for approximately **68% of feature importance**.

![Maths Random Forest Feature Importance](Figures/Maths%20Data_Random_Forest_feature_importance.png)

#### Residual Analysis

The residual distribution was analysed to view the size and spread of the model's prediction errors.

![Maths Random Forest Residuals](Figures/Maths%20Data_Random%20Forest_residual_histogram.png)

### English — Random Forest

Random Forest also produced the strongest predictions for GCSE English, although performance was less accurate than for Maths.

#### Actual vs Predicted

The actual and predicted values show the relationship between the model's predictions and the observed GCSE English results.

![English Random Forest Actual vs Predicted](Figures/English%20Data_Random%20Forest_actual_vs_predicted.png)

## Limitations

The results depend on the variables available within the dataset and cannot capture every factor that may influence student outcomes.

The dataset was limited to two GCSE cohorts from 9 UK schools, meaning the findings may not generalise to other schools or student populations.

The strong influence of recent mock-exam performance also means that predictions may be less reliable when this information is unavailable or changes significantly.

The models are intended to provide predictive insights and could support professional judgement with further investigation.