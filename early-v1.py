## Early Engagement Signals as Predictors of Algorithmic Amplification


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv("dataset.csv")

# Basic inspection
print(df["algorithmic_amplification_index"].describe())

# Plot distribution
# plt.hist(df["algorithmic_amplification_index"], bins=50)
# plt.title("Distribution of Algorithmic Amplification Index")
# plt.xlabel("Algorithmic Amplification Index")
# plt.ylabel("Frequency")
# plt.show()

early_features = [
    "early_likes",
    "early_comments",
    "early_shares",
    "early_total_engagement",
    "early_engagement_velocity",
    "early_comment_share_ratio"
]

print(df[early_features].corrwith(df["algorithmic_amplification_index"]))

# Select features components
features = [
    "early_likes",
    "early_comments",
    "early_shares",
    "follower_count",
    "authority_log",
    "verified",
    "account_age_years"
]

## Features for Velocity Model
# features = [
#     "early_engagement_velocity",
#     "follower_count",
#     "authority_log",
#     "verified",
#     "account_age_years"
# ]

X = df[features]
y = df["algorithmic_amplification_index"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

##### MODEL BLOCK #####

# Train model
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluation
print("R² (Ridge):", r2_score(y_test, y_pred_ridge))
print("RMSE (Ridge):", root_mean_squared_error(y_test, y_pred_ridge))

# Coefficients (standardized)
ridge_coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": ridge_model.coef_
})

print("\nRidge Coefficients (Standardized):")
print(ridge_coefficients.sort_values(by="Coefficient", ascending=False))

print(X.corr())

##### GRADIENT BOOSTING MODEL #####

# IMPORTANT: Do NOT scale features for tree-based models
X_train_gb = X_train
X_test_gb = X_test

# Initialize model
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Train model
gb_model.fit(X_train_gb, y_train)

# Predictions
y_pred_gb = gb_model.predict(X_test_gb)

# Evaluation
print("\n--- Gradient Boosting Results ---")
print("R² (Gradient Boosting):", r2_score(y_test, y_pred_gb))
print("RMSE (Gradient Boosting):", root_mean_squared_error(y_test, y_pred_gb))

# Feature Importance
gb_importance = pd.DataFrame({
    "Feature": features,
    "Importance": gb_model.feature_importances_
})

print("\nGradient Boosting Feature Importance:")
print(gb_importance.sort_values(by="Importance", ascending=False))