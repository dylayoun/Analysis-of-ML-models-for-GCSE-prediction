import pandas as pd
import numpy as np
import sklearn.preprocessing as skpp
from sklearn.model_selection import train_test_split



def load_and_clean():

    """ 
    Loads the 2023 and 2024 trust datasets from CSV, combining them into a single
    dataframe and performing intial cleaning. This includes dropping irrelevant 
    columns, stripping whitespace from column names, generating a Cohort column
    from the ExternalID field, replacing blank strings with NaN and converting the
    GCSE results columns to integers with U grades mapped to 0

    Returns:
        pd.DataFrame: Combined and cleaned dataset ready for preprocessing
    """
    
    data2023 = pd.read_csv("Data/RawData/Trust data 2023.csv")
    data2024 = pd.read_csv("Data/RawData/Trust data 2024.csv")
    data = pd.concat([data2023, data2024], ignore_index=True)
    
    
    data.columns = data.columns.str.strip()
    
    data = data.drop(columns=["Start Y7 GL Eng", "Start Y7 GL Mat", "ExternalSchID", "AcademicYear", "Unnamed: 18"], errors="ignore")
   
        
    # Create cohort column for imputation use
    data["Cohort"] = data["ExternalID"].str[:4].astype(int)

    #Replace blank values with NaN values
    data=data.replace(r'^\s*$', np.nan, regex=True)
                
    # Change grade columns to all numeric values (U -> 0)

    for col in ["GCSEEnglishResult", "GCSEMathsResult"]:
        data[col] = data[col].replace("U", 0).astype(int)


    return data

def prep_subject_data(df, subject_col):
    
    """
    Separates the dataset into features (X) and target (y) for a given subject.
    Drops the alternative subject's results column to prevent data leakage 
    between subject-specific models.

    Args:
        df (pd.DataFrame): Cleaned combined dataset
        subject_col (str): Target column name, either 'GCSEEnglishResult' or
                           'GCSEMathsResult

    Returns:
        tuple: (X, y) where X is the feature dataframe and y is the target column
    """
    
    data = df.copy()
    
    if subject_col == "GCSEEnglishResult":
        other = "GCSEMathsResult"
    elif subject_col == "GCSEMathsResult":
        other = "GCSEEnglishResult"
    else:
        raise ValueError("Invalid subject column")
    
    y = data[subject_col].copy()
    X = data.drop(columns=[subject_col, other, "ExternalID"], errors="ignore")
    
    return X, y


# Feature Encoding (Binary and One-Hot)
    
def fit_binary(data, column):
    le = skpp.LabelEncoder()
    le.fit(data[column].astype(str))
    return le

def apply_binary(data, le, column):
    data = data.copy()
    data[column] = le.transform(data[column].astype(str))
    return data
    

def one_hot_fit(df, columns):
    encoder = skpp.OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoder.fit(df[columns].astype(str))
    return encoder

def one_hot_apply(data, encoder, columns):
    data = data.copy()
    encoded_array = encoder.transform(data[columns].astype(str))
    
    col_names = encoder.get_feature_names_out(columns)
    encoder_df = pd.DataFrame(encoded_array, columns=col_names, index=data.index)
    
    data = pd.concat([data.drop(columns=columns), encoder_df], axis=1)
    
    return data

# Encoding for Ethnicity column (Threshold of 50)

def get_ethnicities(data, threshold=50):
    counts = data["Ethnicity"].value_counts()
    rare_ethnicity =  counts[counts < threshold].index.tolist()
    return rare_ethnicity

def apply_ethnicity_encoding(data, rare_ethnicity):
    data = data.copy()
    if "Ethnicity" in data.columns:
        mask = data["Ethnicity"].isin(rare_ethnicity)
        data.loc[mask, "Ethnicity"] = "OTHER"
    return data

# Scaling numeric features

def fit_scale_numeric(data, columns):
    scaler = skpp.StandardScaler()
    scaler.fit(data[columns])
    return scaler

def apply_scale_numeric(data, scaler, columns):
    data = data.copy()
    data[columns] = scaler.transform(data[columns])
    return data

def fit_preprocess(X_train):
    
    """
    Fits all preprocessing transformations on the training data only, preventing
    data leakage from the test set. This included cohort-based median imputation
    for missing attendance and mock grade values, binary and one-hot encoding of 
    categorical features and a standart scaling of numeric columns
    
    Args:
        X_train (pd.DataFrame): Training feature set prior to preprocessing
        

    Returns:
        dict: A dictionary of fitted preprocessing objects including the encoders,
        scalers, imputation medians and column lists, to be passed to apply_preprocessing().
    """
    
    
    X_train_copy = X_train.copy()
    
    not_int_cols = ["suspensions 23/24","suspensions 22/23","suspensions 21/22","End Y10 Mock Eng","End Y10 Mock Mat"]
    
    for col in not_int_cols:
        X_train_copy[col] = pd.to_numeric(X_train_copy[col], errors="coerce")
            
    
    numeric_cols = X_train_copy.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "Cohort"]
    
    
    ethnicities = get_ethnicities(X_train_copy)
    #Have to apply ethnicity encoding before fitting one hot
    X_train_copy = apply_ethnicity_encoding(X_train_copy, ethnicities)

    binary_cols = ["Gender", "FSM", "PP", "EAL"]
    one_hot_cols = ["Age", "SEND", "Ethnicity"]
    
    imp_cols = ["attendance 23/24", "End Y10 Mock Eng", "End Y10 Mock Mat"]
    imp_cols = [col for col in imp_cols if col in X_train_copy.columns]
    
    
    temp = X_train_copy.copy()
    for col in imp_cols:
        temp[col] = temp[col].replace(0.0, np.nan)
    
    cohort_median = temp.groupby("Cohort")[imp_cols].median()
    global_median = temp[imp_cols].median()
    
    for col in imp_cols:
        
        filler = cohort_median[col]
        
        X_train_copy[col] = X_train_copy[col].replace(0.0, np.nan)
        X_train_copy[col] = X_train_copy[col].fillna(X_train_copy["Cohort"].map(filler))
        X_train_copy[col] = X_train_copy[col].fillna(global_median[col])
    
    general_medians = X_train_copy[numeric_cols].median()
    for col in numeric_cols:
        X_train_copy[col] = X_train_copy[col].fillna(general_medians[col])
        
    binary_encoders = {}
    for col in binary_cols:
        binary_encoders[col] = fit_binary(X_train_copy, col)
    
    
    one_hot_encoder = one_hot_fit(X_train_copy, one_hot_cols)
    
    scaler = fit_scale_numeric(X_train_copy, numeric_cols)
    
    preprocessing_objects = {
        "binary_encoders": binary_encoders,
        "one_hot_encoder": one_hot_encoder,
        "ethnicities": ethnicities,
        "scaler": scaler,
        "cohort_median": cohort_median,
        "numeric_cols": numeric_cols,
        "imputation_cols": imp_cols,
        "binary_cols": binary_cols,
        "one_hot_cols": one_hot_cols,
        "global_median": global_median,
        "general_medians": general_medians
    }
    return preprocessing_objects


def apply_preprocessing(data, objects):
    
    """
    Applies the fitted preprocessing transformations from fit_preprocess onto
    a given dataset. Needs to be called on both training and test sets using
    the same preprocessing objects to ensure consistency and prevent data leakage
    
    Args:
        data (pd.DataFrame): Features to preprocess
        objects (dict): Dictionary of fitted preprocessing objects from 
                        fit_preprocess()

    Returns:
        pd.DataFrame: Fully preprocessed feature set ready for model training
                      or evaluation
    """
    
    data = data.copy()
    
    if "Ethnicity" in data.columns:
        data = apply_ethnicity_encoding(data, objects["ethnicities"])
    
    for col in objects["imputation_cols"]:
        filler = objects["cohort_median"][col]
        data[col] = data[col].replace(0.0, np.nan)
        data[col] = data[col].fillna(data["Cohort"].map(filler))
        data[col] = data[col].fillna(objects["global_median"][col])
        
    for col in objects["numeric_cols"]:
        if col in data.columns:
            data[col] = data[col].fillna(objects["general_medians"][col])
        
    for col, encoder in objects["binary_encoders"].items():
        data = apply_binary(data, encoder, col)
        
    
    data = one_hot_apply(data, objects["one_hot_encoder"], objects["one_hot_cols"])
    data = apply_scale_numeric(data, objects["scaler"], objects["numeric_cols"])
    
    data = data.drop(columns=["Cohort"], errors="ignore")
    
    
    
    return data


if __name__ == "__main__":
    raw_data = load_and_clean()
    raw_data.to_csv("Data/CleanedData/Raw Data/cleaned_raw_data.csv", index=False)

    X_eng, y_eng = prep_subject_data(raw_data, "GCSEEnglishResult")
    X_mat, y_mat = prep_subject_data(raw_data, "GCSEMathsResult")

    #Create test train split of the dataset 
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_eng, y_eng, test_size=0.15, random_state=42)
    X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(X_mat, y_mat, test_size=0.15, random_state=42)

    prep_info_eng = fit_preprocess(X_train_eng)
    prep_info_mat = fit_preprocess(X_train_mat)

    X_train_eng_processed = apply_preprocessing(X_train_eng, prep_info_eng)
    X_test__eng_processed = apply_preprocessing(X_test_eng, prep_info_eng)

    X_train_mat_processed = apply_preprocessing(X_train_mat, prep_info_mat)
    X_test_mat_processed = apply_preprocessing(X_test_mat, prep_info_mat)

    X_train_eng_processed.to_csv("Data/CleanedData/English Data/X_train.csv", index=False)
    X_test__eng_processed.to_csv("Data/CleanedData/English Data/X_test.csv", index=False)
    y_train_eng.to_csv("Data/CleanedData/English Data/y_train.csv", index=False)
    y_test_eng.to_csv("Data/CleanedData/English Data/y_test.csv", index=False)

    X_train_mat_processed.to_csv("Data/CleanedData/Maths Data/X_train.csv", index=False)
    X_test_mat_processed.to_csv("Data/CleanedData/Maths Data/X_test.csv", index=False)
    y_train_mat.to_csv("Data/CleanedData/Maths Data/y_train.csv", index=False)
    y_test_mat.to_csv("Data/CleanedData/Maths Data/y_test.csv", index=False)