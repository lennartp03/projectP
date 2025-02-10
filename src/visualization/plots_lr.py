import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(daily_df, target, forecast_dates, predictions, lower_bounds, upper_bounds):
    """
    Plot the full time series along with forecast predictions and 95% prediction intervals.
    
    Parameters:
        daily_df (DataFrame): Original daily data containing the 'date' column.
        target (str): Name of the target variable.
        forecast_dates (array-like): Dates corresponding to the forecast period.
        predictions (list): List of point predictions.
        lower_bounds (list): Lower bounds of the prediction interval.
        upper_bounds (list): Upper bounds of the prediction interval.
    """
    plt.figure(figsize=(14,6))
    plt.plot(daily_df['date'], daily_df[target], label='Actual', color='gray', linewidth=2)
    plt.plot(pd.to_datetime(forecast_dates), predictions, label='Ridge Predictions', color='red', linewidth=2)
    plt.fill_between(pd.to_datetime(forecast_dates), lower_bounds, upper_bounds, color='red', alpha=0.2, label='95% Prediction Interval')
    plt.xlabel("Date")
    plt.ylabel(f"Daily Mean {target}")
    plt.title("Rolling Forecast with Ridge Regression and Bootstrapped Uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_coefficients(coefficients, features, forecast_dates):
    """
    Plot the evolution of Ridge regression coefficients over the forecast period.
    
    Parameters:
        coefficients (list): List of coefficient arrays (one per forecast).
        features (list): List of feature names.
        forecast_dates (array-like): Dates corresponding to the forecasts.
    """
    coef_df = pd.DataFrame(coefficients, columns=features, index=pd.to_datetime(forecast_dates))
    plt.figure(figsize=(14,8))
    for feature in features:
        plt.plot(coef_df.index, coef_df[feature], label=feature)
    plt.xlabel("Date")
    plt.ylabel("Coefficient Value (Standardized)")
    plt.title("Evolution of Ridge Regression Coefficients Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, 
                            xlabel="Actual km_to_colony_mean", 
                            ylabel="Predicted km_to_colony_mean", 
                            title="Actual vs. Predicted Distance to Colony",
                            figsize=(8, 6)):
    """
    Plot a scatter plot of actual vs. predicted values along with a 45-degree reference line.

    Parameters:
        y_test (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Plot title.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Plot a 45-degree reference line:
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_features(daily_df, feature_list):
    """
    Plots each feature in feature_list over time using the 'date' column from daily_df.
    
    Parameters:
        daily_df (pd.DataFrame): DataFrame containing a 'date' column and feature columns.
        feature_list (list): List of column names to plot.
    """
    n_features = len(feature_list)
    
    # Create subplots: one row per feature, shared x-axis.
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True)
    
    # When there's only one feature, axes is not a list; wrap it into a list.
    if n_features == 1:
        axes = [axes]
    
    # Plot each feature.
    for i, feat in enumerate(feature_list):
        ax = axes[i]
        ax.plot(daily_df['date'], daily_df[feat], marker='o', linestyle='-', label=feat)
        ax.set_title(f"{feat} over Time")
        ax.set_ylabel(feat)
        ax.legend()
    
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()
