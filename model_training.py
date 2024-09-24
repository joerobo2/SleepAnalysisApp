import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet

# For SVR and XGBoost
from sklearn.svm import SVR
#from xgboost import XGBRegressor

# For Neural Networks
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score

import pickle
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.stats.mstats import winsorize

# Load Dataset
df = pd.read_csv('cmu-sleep.csv')

# Data Preprocessing (Replace empty strings, convert relevant columns, Winsorize)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df['term_units'] = df['term_units'].astype(float)
df['Zterm_units_ZofZ'] = df['Zterm_units_ZofZ'].astype(float)
variables = ['bedtime_mssd', 'TotalSleepTime', 'midpoint_sleep', 'frac_nights_with_data', 'daytime_sleep', 'study',
             'term_units', 'Zterm_units_ZofZ']
for var in variables:
    df[var] = winsorize(df[var], limits=[0.05, 0.05])

# Impute missing values (KNN for numerical, most frequent for categorical)
categorical_cols = ['cohort', 'demo_race', 'demo_gender', 'demo_firstgen']
numerical_cols = [col for col in df.columns if col not in categorical_cols]
knn_imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])
most_frequent_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = most_frequent_imputer.fit_transform(df[categorical_cols])

# Verify no NaN values remaining
assert not df.isnull().values.any(), "There are still NaN values in the dataframe."

# Features and target variable
X = df[['TotalSleepTime', 'midpoint_sleep', 'daytime_sleep', 'term_gpa', 'term_units',
        'frac_nights_with_data', 'demo_firstgen', 'demo_race', 'demo_gender']]  # Include demo_firstgen as numerical
y = df['cum_gpa']

# Convert all columns in X to float
X = X.astype(float)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create polynomial features (degree can be adjusted)
poly_features = PolynomialFeatures(degree=2)  # Adjust the degree as needed
X_poly = poly_features.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Define models and parameter grids for GridSearchCV
models = {
    'HistGradientBoostingRegressor': (
        HistGradientBoostingRegressor(),
        {
            'learning_rate': [0.01, 0.03, 0.1],
            'max_depth': [3, 5, 7],
            'max_iter': [100, 200],
            'loss': ['quantile'],
            'quantile': [0.4, 0.5]
        }
    ),
    'RandomForestRegressor': (
        RandomForestRegressor(),
        {
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [2, 4],
            'min_samples_split': [2, 5],
            'n_estimators': [50, 100]
        }
    ),
    'SVR': (
        SVR(),
        {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    ),
    'MLPRegressor': (
        MLPRegressor(),
        {
            'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.001, 0.01, 0.1]
        }
    ),
    'ElasticNet': (
        ElasticNet(),
        {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
    )
}

# Function to train and evaluate models without early stopping
def train_evaluate_model(model_name, model, param_grid, X_train, X_test, y_train, y_test):
    # KFold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV with early stopping
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error',
                               error_score='raise')
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Predict on testing data
    y_pred = best_model.predict(X_test)

    # Evaluate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Print results
    print(f"\nBest {model_name} parameters: {grid_search.best_params_}")
    print(f"{model_name} - Mean Squared Error: {mse}")
    print(f"{model_name} - R-squared: {r2}")
    print(f"{model_name} - Mean Absolute Error: {mae}")

    # Visualize predictions vs. actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Cumulative GPA')
    plt.ylabel('Predicted Cumulative GPA')
    plt.title(f'Predicted vs. Actual Cumulative GPA ({model_name})')
    plt.show()

    return best_model


# Train and evaluate each model
for name, (model, param_grid) in models.items():
    best_model = train_evaluate_model(name, model, param_grid, X_train, X_test, y_train, y_test)

    # Save the best model
    with open(f'{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)