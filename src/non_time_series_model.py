import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import shap
import os
from functools import partial
import joblib
from sklearn.preprocessing import OneHotEncoder
optuna.logging.set_verbosity(optuna.logging.WARNING)
import optuna



def load_mapping(path):
    files = os.listdir(path)
    df_original = pd.concat([pd.read_csv(os.path.join(path, file)) for file in files])
    df = df_original.drop(columns=['equinox', 'original_track_id']).reset_index(drop=True)
    return df


def load_and_preprocess_data(mapping_path, filepath, smooth_climate=False, window=7):
    """
    Loads the CSV file, filters for 'Adelie Penguin', maps the breed_stage from the mapping file,
    and (optionally) applies a rolling mean to climate variables.
    """
    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        print("Error loading CSV with index_col=0. Loading without it.")
        df = pd.read_csv(filepath)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Filter for 'Adelie Penguin'
    df_adelie = df[df['common_name'] == 'Adelie Penguin'].copy()

    # Map breed_stage using the mapping file
    mapping_frame = load_mapping(mapping_path)
    mapping_frame_unique = mapping_frame.drop_duplicates(subset='track_id')
    df_adelie['breed_stage'] = df_adelie['track_id'].map(
        mapping_frame_unique.set_index('track_id')['breed_stage']
    )

    # Optionally apply smoothing to climate variables
    climate_vars = ['tp', 't2m', 'sst', 'siconc', 'sd', 'rsn', 'avg_smr']
    if smooth_climate:
        for var in climate_vars:
            df_adelie[var] = df_adelie[var].rolling(window=window, min_periods=1).mean()

    return df_adelie


def get_features_and_target(df):
    """
    Constructs the feature matrix X and the target vector y.
    Numeric features: climate variables.
    Categorical features: colony_name and breed_stage.
    """
    climate_vars = ['tp', 't2m', 'sst', 'siconc', 'sd', 'rsn', 'avg_smr']
    categorical_features = ['colony_name', 'breed_stage']
    needed_vars = climate_vars + categorical_features
    target = 'km_to_colony_mean'
    
    X = df[needed_vars]
    y = df[target]
    return X, y

def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
    return r2, rmse

# ---------------- Hyperparameter Optimization Objectives ----------------

def objective_rf(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 6)
    max_features = trial.suggest_float('max_features', 0.5, 1.0)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    # Use 5-fold CV within the inner loop
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(r2_score(y_val, y_pred))
    return np.mean(scores)

def objective_xgb(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    gamma = trial.suggest_float('gamma', 0, 5)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 5)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 5)
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(r2_score(y_val, y_pred))
    return np.mean(scores)

# ---------------- Nested Cross Validation ----------------

def nested_cv_rf(X, y, outer_splits=5, inner_trials=50):
    """
    Performs nested cross-validation for Random Forest:
      - Outer loop: splits the data into training and test folds.
      - Inner loop: hyperparameter tuning via Optuna.
    """
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    outer_r2_scores = []
    outer_rmse_scores = []
    best_params_list = []
    all_fold_predictions = {}
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        study = optuna.create_study(direction="maximize")
        study.optimize(partial(objective_rf, X=X_train_fold, y=y_train_fold), n_trials=inner_trials)
        best_params = study.best_params
        best_params_list.append(best_params)
        
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        
        r2 = r2_score(y_test_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
        print(f"RF Fold {fold}: Best params: {best_params}, R²: {r2:.3f}, RMSE: {rmse:.3f}")
        outer_r2_scores.append(r2)
        outer_rmse_scores.append(rmse)

        all_fold_predictions[f'fold_{fold}'] = {
            'indices': X_test_fold.index,
            'y_true': y_test_fold.values,
            'y_pred': y_pred
        }

    # Create SHAP explainer
    explainer = shap.Explainer(model)
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_fold)
    # Summary plot
    shap.summary_plot(shap_values, X_test_fold)
        
    avg_r2 = np.mean(outer_r2_scores)
    avg_rmse = np.mean(outer_rmse_scores)
    print(f"\nNested CV Random Forest - Average R²: {avg_r2:.3f} - Average RMSE: {avg_rmse:.3f}")
    joblib.dump(all_fold_predictions, 'rf_nested_predictions.pkl', compress = 1)    
    return outer_r2_scores, best_params_list

def nested_cv_xgb(X, y, outer_splits=5, inner_trials=50):
    """
    Performs nested cross-validation for XGBoost:
      - Outer loop: splits the data into training and test folds.
      - Inner loop: hyperparameter tuning via Optuna.
    """
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    outer_r2_scores = []
    outer_rmse_scores = []
    best_params_list = []
    all_fold_predictions = {} 
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        study = optuna.create_study(direction="maximize")
        study.optimize(partial(objective_xgb, X=X_train_fold, y=y_train_fold), n_trials=inner_trials)
        best_params = study.best_params
        best_params_list.append(best_params)
        
        model = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        
        r2 = r2_score(y_test_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
        print(f"XGB Fold {fold}: Best params: {best_params}, R²: {r2:.3f}, RMSE: {rmse:.3f}")
        outer_r2_scores.append(r2)
        outer_rmse_scores.append(rmse)

        all_fold_predictions[f'fold_{fold}'] = {
            'indices': X_test_fold.index,
            'y_true': y_test_fold.values,
            'y_pred': y_pred
        }

    # Create SHAP explainer
    explainer = shap.Explainer(model)
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_fold)
    # Summary plot
    shap.summary_plot(shap_values, X_test_fold)
        
    avg_r2 = np.mean(outer_r2_scores)
    avg_rmse = np.mean(outer_rmse_scores)
    print(f"\nNested CV XGBoost - Average R²: {avg_r2:.3f} - Average RMSE: {avg_rmse:.3f}")
    joblib.dump(all_fold_predictions, 'xgb_nested_predictions.pkl', compress = 1)
    return outer_r2_scores, best_params_list


# ======= Plotting Functions =======

def plot_predictions(predictions):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (fold, data) in enumerate(predictions.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        indices = data['indices']
        
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.3)
        ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            color='red', lw=2, linestyle='--', label='Ideal')
        ax.set_title(f'{fold}')
        ax.set_xlabel('True Distance (km)')
        ax.set_ylabel('Predicted Distance (km)')
        ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_distance_to_colony(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['km_to_colony_mean'], kde=True)
    plt.title('Distribution of Distance to Colony')
    plt.xlabel('Distance to Colony (km)')
    plt.grid(False)
    plt.show()

def plot_residuals(predictions):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (fold, data) in enumerate(predictions.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        indices = data['indices']
        
        residuals = y_true - y_pred

        ax = axes[i]
        sns.histplot(residuals, bins=30, kde=True, color='orange', 
                    label='XGBoost', ax=ax, stat='count', alpha=0.5)
        ax.set_title("Residual Distribution for " + fold)
        ax.set_xlabel('"Residuals (km)"')
        ax.set_ylabel('Frequency')
        ax.grid(False)

    plt.tight_layout()
    plt.show()


# get one example fold of rf and xgb
def plot_random_sample(df, fold, predictions_rf, predictions_xgb):
    rf_fold = predictions_rf[fold]
    xgb_fold = predictions_xgb[fold]

    rf_best_preds = rf_fold['y_pred']
    xgb_best_preds = xgb_fold['y_pred']

    # only use 20% of df_adelie, to represent test set size
    df_adelie_sampled = df.sample(frac=0.2)

    # Create common bins for both histograms
    all_kms = np.concatenate([df_adelie_sampled['km_to_colony_mean'], xgb_best_preds, rf_best_preds])
    bins = np.linspace(all_kms.min(), all_kms.max(), 76)  # 75 bins

    plt.figure(figsize=(8, 6))

    # Plot histogram dataframe
    sns.histplot(df_adelie_sampled['km_to_colony_mean'], bins=bins, kde=True, color='black', 
                label='actual', stat='count', alpha=0.8)

    # Plot histogram for XGBoost
    sns.histplot(xgb_best_preds, bins=bins, kde=True, color='orange', 
                label='XGBoost', stat='count', alpha=0.4)

    # Plot histogram for Random Forest
    sns.histplot(rf_best_preds, bins=bins, kde=True, color='blue', 
                label='Random Forest', stat='count', alpha=0.4)

    plt.xlabel("Km to colony")
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.title("km to colony actual and predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()




