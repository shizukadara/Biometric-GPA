import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Load the data
file_path = 'finalfinal.csv'
df = pd.read_csv(file_path)

# List of irrelevant features to drop
irrelevant_features = [
    'TotalTimeOutsideNonOvernight_Weekend', 
    'TotalTimeOutsideOvernight_Weekend', 
    'TotalTimeOutsideOvernight_Weekday', 
    'GymnasiumVisitCount', 
]

# Drop irrelevant features
df = df.drop(columns=irrelevant_features, errors='ignore')

# Check for missing values and drop rows with missing target variable (CGPA)
df = df.dropna(subset=['CGPA'])

# Separate features (X) and target (y)
X = df.drop(columns=['CGPA'])  # Exclude the target column
y = df['CGPA']

# Log transform the target variable to stabilize variance
y_log = np.log1p(y)

# Feature Engineering: Use PowerTransformer to transform features to normal distribution
transformer = PowerTransformer()
X_transformed = pd.DataFrame(transformer.fit_transform(X), columns=X.columns)

# Add Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_transformed)
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

# Combine original and polynomial features
X_combined = pd.concat([X_transformed, X_poly], axis=1)

# Add Interaction Features
X_combined['Trips_Library_Interaction'] = X_transformed['TotalTrips_Weekday'] * X_transformed['TotalLibraryTime']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_log, test_size=0.2, random_state=42)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------ Gradient Boosting Model ------------------------

# Define and train Gradient Boosting Regressor
best_gb_model = GradientBoostingRegressor(
    learning_rate=0.01, 
    max_depth=3, 
    n_estimators=300, 
    subsample=0.8, 
    random_state=42
)
best_gb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_gb_log = best_gb_model.predict(X_test_scaled)
y_pred_gb = np.expm1(y_pred_gb_log)  # Reverse log transformation

# ------------------------ XGBoost Model ------------------------

# Define and train XGBoost Regressor
best_xgb_model = XGBRegressor(
    learning_rate=0.01, 
    max_depth=3, 
    n_estimators=300, 
    subsample=0.8, 
    random_state=42, 
    objective='reg:squarederror', 
    verbosity=0
)
best_xgb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_xgb_log = best_xgb_model.predict(X_test_scaled)
y_pred_xgb = np.expm1(y_pred_xgb_log)  # Reverse log transformation

# ------------------------ CatBoost Model ------------------------

# Define and train CatBoost Regressor
catboost_model = CatBoostRegressor(
    iterations=500, 
    learning_rate=0.01, 
    depth=6, 
    random_state=42, 
    verbose=0
)
catboost_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_cat_log = catboost_model.predict(X_test_scaled)
y_pred_cat = np.expm1(y_pred_cat_log)  # Reverse log transformation

# ------------------------ Evaluation Metrics ------------------------

def evaluate_regression(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n========== {model_name} Regression Metrics ==========")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='orange', edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error Line')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residual Plot for {model_name}')
    plt.legend()
    plt.show()

# Evaluate Gradient Boosting Model
evaluate_regression(np.expm1(y_test), y_pred_gb, 'Gradient Boosting')

# Evaluate XGBoost Model
evaluate_regression(np.expm1(y_test), y_pred_xgb, 'XGBoost')

# Evaluate CatBoost Model
evaluate_regression(np.expm1(y_test), y_pred_cat, 'CatBoost')
