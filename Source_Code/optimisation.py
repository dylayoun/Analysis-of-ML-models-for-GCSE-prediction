import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import sklearn.preprocessing as skpp
from sklearn.model_selection import train_test_split
"""
optimisation.py

This script evaluates four imputation strategies for handeling missing attendance and mock grade data in the 
2023 cohort. The strategies tested are:

    1. Historical Mean  - imputes missing values using the mean of each 
                         student's available historical attendance records
                         from prior years.
                         
    2. Cohort Median    - imputs missing values using the median attendance 
                         value within each cohort group
    
    3. Percentile-Based - imputes missing values using a specified percentile
                          of the 2024 cohort's attendance distribution, tested
                          across percentiles from the 10th to the 90th
    
    4. Recursive Range  - imputes missing values using random values drawn 
                          from a defined range, tested across multiple lower 
                          bound values
                          
Each strategy is evaluated by training a baseline Linear Regression model on the imputed dataset and 
calculating Mean Absolute Error (MAE) for both English and Maths predictions. The strategy producing the 
lowest MAE is selected as the imputation approach used in the final preprocessing pipeline implemented
in preprocessing.py
"""

if __name__ == "__main__":
    data = pd.read_csv("Data/CleanedData/Raw Data/cleaned_raw_data.csv")

    # Historical Mean

    data_his_mean = data.copy()

    hist_cols = ["attendance 22/23", "attendance 21/22"]
    data_his_mean["temp_his_mean"] = data_his_mean[hist_cols].replace(0.0, np.nan).mean(axis=1)

    m2023_zero = (data_his_mean["Cohort"] == 2023) & (data_his_mean["attendance 23/24"] == 0.0)
    data_his_mean.loc[m2023_zero, 'attendance 23/24'] = data_his_mean.loc[m2023_zero, 'temp_his_mean']

    data_his_mean = data_his_mean.drop(columns="temp_his_mean")

    #data_his_mean is the dataset with missing attendance values imputed using the historical mean method.

    # Cohort Median

    data_cohort_median = data.copy()

    for col in ["attendance 22/23", "attendance 23/24"]:
        data_cohort_median[col] = data_cohort_median.groupby("Cohort")[col].transform(lambda x: x.replace(0.0, np.nan).fillna(x.replace(0.0, np.nan).median()))
        
    #data_cohort_median is the dataset with missing attendance values imputed using the cohort median method.

    # Recursive Range Imputation
    def recursive_range(lower, upper):
        data_recursive_range = data.copy()

        range_lower = lower
        range_upper = upper

        m2023_zero = (data_recursive_range["Cohort"] == 2023) & (data_recursive_range["attendance 23/24"] == 0.0)
        count_missing = m2023_zero.sum()
        random_values = [random.uniform(range_lower, range_upper) for _ in range(count_missing)]

        data_recursive_range.loc[m2023_zero, 'attendance 23/24'] = random_values
        
        return data_recursive_range

    #data_recursive_range is the dataset with missing attendance values imputed using the recursive range imputation method.
    #Each time change the lower and upper bounds to see the effect on model performance and narrow it down to the optimal range.

    #Percentile Imputation
    def percentile_imp(target_p):
        data_percentile = data.copy()

        target_percentile = target_p

        ref_data_2024 = data_percentile.loc[(data_percentile['Cohort'] == 2024) & (data_percentile['attendance 23/24'] > 0), 'attendance 23/24']

        attendance_value = np.percentile(ref_data_2024, target_percentile)


        mask_att_zero = (data_percentile['Cohort'] == 2023) & (data_percentile['attendance 23/24'] == 0.0)


        data_percentile.loc[mask_att_zero, 'attendance 23/24'] = attendance_value
        
        return data_percentile

    #data_percentile is the dataset with missing attendance values imputed using the percentile imputation method.
    #Each time change the target_percentile to see the effect on model performance and narrow it down to the optimal percentile.

    #Scale the new attendance columns to keep consistency
    def scale_numeric(df, columns):
        
        for col in columns:
            scaler = skpp.StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            
        return df

    scale_numeric(data_his_mean, ["attendance 23/24"])
    scale_numeric(data_cohort_median, ["attendance 23/24"])




    #create a function to create model and evaluate performance with MAE

    def evaluate_model(data, target_col):
        
        if target_col == "GCSEEnglishResult":
            other_target = "GCSEMathsResult"
        elif target_col == "GCSEMathsResult":
            other_target = "GCSEEnglishResult"
        else:
            raise ValueError("Invalid target column. Must be 'GCSEEnglishResult' or 'GCSEMathsResult'.")
        
        drop_list = [
            "ExternalID", "Cohort", "GCSEEnglishResult", "GCSEMathsResult"
        ]

        X = data.drop(columns=drop_list, errors="ignore")
        y = data[target_col]
        
        X = pd.get_dummies(X, drop_first=True)
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.median(numeric_only=True))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        return mae

    #Evaluate the performance of Historical Mean and Cohort Median imputation method on both English and Maths targets
    test_configs = {
        "Historical Mean": data_his_mean,
        "Cohort Median": data_cohort_median
    }
    prediction_results = {}

    for name, df in test_configs.items():
        mae_eng = evaluate_model(df, "GCSEEnglishResult")
        mae_mat = evaluate_model(df, "GCSEMathsResult")
        
        prediction_results[name] = {"MAE_English": mae_eng, "MAE_Maths": mae_mat }
        
    #Output the results for the imputation methods

    print("Imputation Method Performance for Historical Mean and Cohort Median:")
    for method, results in prediction_results.items():
        print(f"{method}: MAE English = {results['MAE_English']:.6f}, MAE Maths = {results['MAE_Maths']:.6f}")

    percentile_imp_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    percentile_results = {}

    for p in percentile_imp_values:
        df_percentile = percentile_imp(p)
        scale_numeric(df_percentile, ["attendance 23/24"])
        
        mae_eng = evaluate_model(df_percentile, "GCSEEnglishResult")
        mae_mat = evaluate_model(df_percentile, "GCSEMathsResult")
        
        percentile_results[p] = {"MAE_English": mae_eng, "MAE_Maths": mae_mat }
        
    print("Imputation Method Performance:")
    for method, results in percentile_results.items():
        print(f"{method}: MAE English = {results['MAE_English']:.6f}, MAE Maths = {results['MAE_Maths']:.6f}")

    print("Imputation Method Performance for Recursive Range Imputation:")
    for lower in [0, 25, 50, 75]:
        df_recursive = recursive_range(lower, 100)
        df_recursive = scale_numeric(df_recursive, ["attendance 23/24"])
        
        mae_eng = evaluate_model(df_recursive, "GCSEEnglishResult")
        mae_mat = evaluate_model(df_recursive, "GCSEMathsResult")
        
        print(f"Range {lower} to 100: MAE English = {mae_eng:.6f}, MAE Maths = {mae_mat:.6f}")