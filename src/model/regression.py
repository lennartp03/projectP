import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

def rolling_forecast(X_scaled, y, initial_training_size=30, n_bootstrap=100, alpha=1.0):
    """
    Perform an expanding-window Ridge regression forecast with bootstrapped uncertainty.
    
    Parameters:
        X_scaled (DataFrame): Standardized predictor variables.
        y (Series): Target variable.
        initial_training_size (int): Number of initial observations used for the first training.
        n_bootstrap (int): Number of bootstrap iterations for each forecast.
        alpha (float): Regularization strength for Ridge regression.
    
    Returns:
        dict: Contains lists of point predictions, lower/upper bounds of the 95% prediction interval,
              actual target values, forecast dates, model coefficients for each forecast, and performance metrics.
    """
    predictions = []
    prediction_lower = []
    prediction_upper = []
    actuals = []
    coefficients_list = []
    n_obs = len(X_scaled)
    
    for i in range(initial_training_size, n_obs):
        X_train = X_scaled.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X_scaled.iloc[[i]]  # Ensure test data remains 2D
        y_test = y.iloc[i]
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        
        # Bootstrap to estimate uncertainty.
        bootstrap_preds = []
        for b in range(n_bootstrap):
            bootstrap_indices = np.random.choice(X_train.index, size=len(X_train), replace=True)
            X_train_boot = X_train.loc[bootstrap_indices]
            y_train_boot = y_train.loc[bootstrap_indices]
            boot_model = Ridge(alpha=alpha)
            boot_model.fit(X_train_boot, y_train_boot)
            bootstrap_preds.append(boot_model.predict(X_test)[0])
        bootstrap_preds = np.array(bootstrap_preds)
        lower_bound = np.percentile(bootstrap_preds, 2.5)
        upper_bound = np.percentile(bootstrap_preds, 97.5)
        
        predictions.append(y_pred)
        prediction_lower.append(lower_bound)
        prediction_upper.append(upper_bound)
        actuals.append(y_test)
        coefficients_list.append(model.coef_)
    
    metrics = {"r2": r2_score(actuals, predictions),
               "rmse": np.sqrt(mean_squared_error(actuals, predictions))}
    
    return {"predictions": predictions,
            "prediction_lower": prediction_lower,
            "prediction_upper": prediction_upper,
            "actuals": actuals,
            "coefficients": coefficients_list,
            "metrics": metrics,
            "model": model}
