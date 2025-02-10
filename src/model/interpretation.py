import shap

def compute_shap_explainer(model, X_scaled):
    """
    Compute SHAP values for a linear model.
    """
    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    return explainer, shap_values