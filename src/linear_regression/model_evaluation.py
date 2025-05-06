import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

def metrics_table(Y_test, y_pred_list, indexes):
    metrics = {
        'R2': [r2_score(Y_test, y_pred) for y_pred in y_pred_list],
        'RMSE': [root_mean_squared_error(Y_test, y_pred) for y_pred in y_pred_list],
        'MAE': [mean_absolute_error(Y_test, y_pred) for y_pred in y_pred_list]
    }

    df_metrics = pd.DataFrame(metrics, index=[idx for idx in indexes])

    df_metrics = df_metrics.round(6)

    return df_metrics


def weights_table(data, X, fitted_models, names):
    numerical_features = list(X.select_dtypes(['int64', 'float64']).columns)

    categorical_features = list(X.select_dtypes(['object', 'category']).columns)

    # creating feature names
    onehotencoded_features = []
    for feat in categorical_features:
        values = data[feat].dropna().unique()
        onehotencoded_features.extend([f"{feat} - {val}" for val in values])

    all_features = []
    all_features.extend(numerical_features)
    all_features.extend(onehotencoded_features)

    coefs = {}
    coefs['Feature'] = all_features
    for name, model in zip(names, fitted_models):
        coefs[name] = model.coef_

    df_coefficients = pd.DataFrame(coefs)

    intercepts = {}
    intercepts['Feature'] = ['Intercept']
    for name, model in zip(names, fitted_models):
        intercepts[name] = model.intercept_

    df_intercepts = pd.DataFrame(intercepts)

    df_results = pd.concat([df_coefficients, df_intercepts], ignore_index=True)

    df_results = df_results.round(6)

    styled_df = (
        df_results.style
        .set_properties(**{
            'text-align': 'left',
            'white-space': 'normal',
            'min-width': '200px'
        })
        .set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'left')]
        }])
    )

    return styled_df