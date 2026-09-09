import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def feature_importance(subject, model, X_train, top_n=15):
    """
    Generates and saves a horizontal bar chat of the top N most important features from a fitted 
    Random Forest model ranked by mean decrease in impurity. Used to identify which input features
    contribute most to predictions for either subject

    Args:
        subject (str): Subject lable used for filename and title
        model (RandomForestRegressor): Fitted RandomForestRegressor model 
        X_train (pd.DataFrame): Training feature set used to retrive feature names
        top_n (int, optional): Number of top features to display. Defaults to 15.

    Returns:
        None
    """
    
    importance = model.feature_importances_
    feature_names = X_train.columns
    
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })
    
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["Feature"], feature_importance["Importance"])
    plt.gca().invert_yaxis()
    
    plt.title(f"Random Forest Feature Importance - {subject}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(f"Figures/{subject}_Random_Forest_feature_importance.png", bbox_inches="tight")
    plt.close()
    
    return None

def residual_histogram(subject, model_name, y_test, predictions):
    """
    Generates and saves a histogram of residuals for a given model and subject. Used to assess
    the distribution of prediction errors and identify and systematic bias in the model

    Args:
        subject (str): Subject lable used in filename and title
        model_name (str): Model name used in filename and title
        y_test : Actual target values from the test set
        predictions : Predicted values generated from the model
        
    Returns:
        None
    """
    
    y_test = np.ravel(y_test)
    predictions = np.ravel(predictions)

    residuals = y_test - predictions

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(0, linestyle="--", color="black")
    plt.title(f"{model_name} - Residual Distribution ({subject})")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(f"Figures/{subject}_{model_name}_residual_histogram.png", bbox_inches="tight")
    plt.close()


def error_boxplot(subject, model_name, y_test, predictions):
    """
    Generates and saves a boxplot of residuals for a given model and subject. Used to visualise
    the spread and distribution of prediction errors, including identifying any outliers

    Args:
        subject (str): Subject lable used in filename and title
        model_name (str): Model name used in filename and title
        y_test : Actual target values from the test set
        predictions : Predicted values generated from the model

    Returns:
        None
    """
    
    y_test = np.ravel(y_test)
    predictions = np.ravel(predictions)
    
    residuals = y_test - predictions
    
    plt.figure(figsize=(6, 6))
    plt.boxplot(residuals, vert=True)
    plt.title(f"{model_name} Residuals Boxplot ({subject})")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(f"Figures/{subject}_{model_name}_residuals_boxplot.png")
    plt.close()
    
    return None

def actual_vs_predicted_scatter(subject, model_name, y_test, predictions):
    """
    Generates and saves a scatter plot comparing actual vs predicted GCSE grades for a given
    model and subject. A small amount of jitter is introduced to actual values given the 
    discrete nature of GCSE grades. A diagonal line representing perfect prediction is included

    Args:
        subject (str): Subject lable used in filename and title
        model_name (str): Model name used in filename and title
        y_test : Actual target values from the test set
        predictions : Predicted values generated from the model

    Returns:
        None
    """
    
    y_test = np.ravel(y_test)
    predictions = np.ravel(predictions)
    
    plt.figure (figsize=(8, 8))
    
    jitter = np.random.normal(0, 0.25, size=len(y_test))
    
    plt.scatter(y_test + jitter, predictions, alpha=0.35, s=30)
    
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    
    plt.title(f"{model_name} - Actual vs Predicted ({subject})")    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(f"Figures/{subject}_{model_name}_actual_vs_predicted.png")
    plt.close()
    
    return None


def plot_tuning_line(name, subject, cv_results):
    """
    Generates and saves a line plot of cross-validation MAE scores across all tested hyperparameter
    configurations, ranking them from highest to lowest. This visualised the optimisation process
    and confirms convergence. This is skipped for Linear Regressor as it contains no hyperparameters
    to optimise

    Args:
        name (str): Model name used in filename, title and to determine if tuning is required
        subject (str): Subject label used in filename and title
        cv_results (dict): Cross-validation results dictionary from a fitted RandomisedSearchCV 
    
    Returns:
        None
    """
    
    if name == "Linear Regression":
        return
    
    scores = sorted([abs(x) for x in cv_results["mean_test_score"]], reverse=True)
    iter = range(1, len(scores) + 1)
    
    plt.figure(figsize=(10,6))
    plt.plot(iter, scores, marker="o", linestyle="-", color="blue", label="Mean MAE for CV")
    
    plt.title(f"{name} Hyperparameter Tuning - {subject}")
    plt.xlabel("Ranked Configurations")
    plt.ylabel("Mean MAE")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.savefig(f"Figures/{subject}_{name}_tuning.png")
    plt.close()
    
    print(f"Line plot created for {name} - {subject}")
    
    return 

def model_experiment(subject):
    """
    Runs hyperparameter optimisation for all four models (Linear Regression, Decision Tree, Random Forest
    and Support Vector Regression) for a given subject using RandomisedSearchCV with 5-fold cross-validation.
    Generates tuning plots for each model and evaluates the best found configuration on the held test set

    Args:
        subject (str): Subject data folder name, either English or Maths Data.

    Returns:
        pd.DataFrame: Results dataframe containing the model name, best parameters, MAE, R^2 and RMSE for 
                      each model
    """
    
    X_train = pd.read_csv(f"Data/CleanedData/{subject}/X_train.csv")
    X_test = pd.read_csv(f"Data/CleanedData/{subject}/X_test.csv")
    y_train = pd.read_csv(f"Data/CleanedData/{subject}/y_train.csv")
    y_test = pd.read_csv(f"Data/CleanedData/{subject}/y_test.csv")
    
    models = {
        "Linear Regression": (LinearRegression(),{}),
        "Decision Tree": (DecisionTreeRegressor(random_state=42),
                          {"max_depth": [None, 5, 10, 15, 20, 25, 30],
                           "min_samples_split": [2, 5, 10, 20, 25, 30, 40, 50]
                           }),
    
        "Random Forest": (RandomForestRegressor(random_state=42),
                          {"n_estimators": [100, 200, 300],
                           "max_depth": [None, 5, 10, 20],
                           "min_samples_leaf": [1, 2, 4, 8]
                            }),
        "Support Vector Regressor": (SVR(), 
                                   {"kernel": ["rbf", "linear"],
                                    "C": [0.1, 1, 10],
                                    "epsilon": [0.01, 0.1, 0.5, 1]
                                   })
    }
    
    results = []
    
    for name, (model, params) in models.items():
        search = RandomizedSearchCV(model, params, n_iter=25, cv=5, 
                                    scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        
        search.fit(X_train, y_train.values.ravel())
        
        plot_tuning_line(name, subject, search.cv_results_)
        
        cv_results_df = pd.DataFrame(search.cv_results_)
        
        
        top_3 = cv_results_df.nsmallest(3, 'rank_test_score')
        
        print(f"\n--- Top 3 {name} Configurations ({subject}) ---")
        for _, row in top_3.iterrows():
            print(f"Rank {row['rank_test_score']}: MAE {abs(row['mean_test_score']):.4f} with {row['params']}")
            
        best_model = search.best_estimator_
        
        predictions = best_model.predict(X_test)
        
        
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({
            "Model": name,
            "Best Parameters": search.best_params_,
            "MAE": mae,
            "R^2": r2,
            "RMSE": rmse
        })
        

        
    return pd.DataFrame(results)

def best_model(subject):
    """
    Trains the final models using the optimised hyperparameters identified in model_experiment() and
    evaluates them on the held test set. Generates the evaluation figures including actual vs predicted
    scatter plots, histograms, error boxplots and feature importance graphs for the Random Forest model. 

    Args:
        subject (str): Subject data folder name, either English or Maths Data.

    Returns:
        pd.DataFrame: Results dataframe containing model name, MAE, R^2 and RMSE for each
                      final model
    """
    
    X_train = pd.read_csv(f"Data/CleanedData/{subject}/X_train.csv")
    X_test = pd.read_csv(f"Data/CleanedData/{subject}/X_test.csv")
    y_train = pd.read_csv(f"Data/CleanedData/{subject}/y_train.csv")
    y_test = pd.read_csv(f"Data/CleanedData/{subject}/y_test.csv")
    
    models = []
    
    if subject == "English Data":
        Linear_Regression =  LinearRegression()
        Decision_Tree_English = DecisionTreeRegressor(max_depth=5, min_samples_split=25, random_state=42)
        Random_Forest_English = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=4, random_state=42)
        SVM_English = SVR(kernel="linear", C=0.1, epsilon=0.01)
        models = [Linear_Regression, Decision_Tree_English, Random_Forest_English, SVM_English]
    else:
        Linear_Regression =  LinearRegression()
        Decision_Tree_Maths = DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42)
        Random_Forest_Maths = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)
        SVM_Maths = SVR(kernel="rbf", C=1, epsilon=0.01)
        models = [Linear_Regression, Decision_Tree_Maths, Random_Forest_Maths, SVM_Maths]
   
    
    results = []
    for model in models:
        model.fit(X_train, y_train.values.ravel())
        predictions = model.predict(X_test)
        
        model_name = type(model).__name__
        
        if "RandomForest" in model_name:
            actual_vs_predicted_scatter(subject, "Random Forest", y_test, predictions)
            error_boxplot(subject, "Random Forest", y_test, predictions)
            residual_histogram(subject, "Random Forest", y_test, predictions)
            feature_importance(subject, model, X_train)
            
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({
            "Model": type(model).__name__,
            "MAE": mae,
            "R^2": r2,
            "RMSE": rmse
        })
    
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    eng_results = model_experiment("English Data")
    mat_results = model_experiment("Maths Data")

    print("\n--- English Results ---")
    print(eng_results[['Model', 'MAE', 'R^2', 'RMSE', 'Best Parameters']])

    print("\n--- Maths Results ---")
    print(mat_results[['Model', 'MAE', 'R^2', 'RMSE', 'Best Parameters']])

    eng_results = best_model("English Data")
    mat_results = best_model("Maths Data")

    print("\n--- English Results ---")
    print(eng_results[['Model', 'MAE', 'R^2', 'RMSE']])

    print("\n--- Maths Results ---")
    print(mat_results[['Model', 'MAE', 'R^2', 'RMSE']])

    with pd.ExcelWriter("Data/Results/Model_Performance.xlsx") as writer:
        eng_results.to_excel(writer, sheet_name="English Results", index=False)
        mat_results.to_excel(writer, sheet_name="Maths Results", index=False)