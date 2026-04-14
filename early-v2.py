## Early Engagement Signals as Predictors of Algorithmic Amplification
## Graduate Research Project — Jose Diaz, CSUN, May 2026
## Improved version: tuned hyperparameters, leakage checks, VIF, log-transform,
##                   cross-validated GB, feature importance plot, reproducibility.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV          # auto-tunes alpha via CV
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
print(f"scikit-learn version : {sklearn.__version__}")
print(f"pandas version       : {pd.__version__}")
print(f"numpy version        : {np.__version__}")

# ── Load dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv("dataset.csv")

# ── Target variable inspection ───────────────────────────────────────────────
# Algorithmic amplification indices are typically right-skewed.
# Log-transforming a skewed target often improves linear model performance.
target_raw = df["algorithmic_amplification_index"]
print("\nTarget variable summary:")
print(target_raw.describe())
print(f"Skewness: {target_raw.skew():.3f}")

# Apply log1p transform if skewness > 1 (common threshold)
USE_LOG_TARGET = target_raw.skew() > 1.0
if USE_LOG_TARGET:
    print(">> Skewness > 1 detected. Applying log1p transform to target.")
    y = np.log1p(target_raw)
else:
    print(">> Target is approximately symmetric. Using raw values.")
    y = target_raw

# ── Correlation check (early signal features vs. target) ─────────────────────
# NOTE: early_total_engagement is excluded from modeling because it is a
# linear combination of early_likes + early_comments + early_shares.
# Including it alongside its components causes perfect multicollinearity.
early_features = [
    "early_likes",
    "early_comments",
    "early_shares",
    "early_total_engagement",     # inspected here only — NOT used in model
    "early_engagement_velocity",
    "early_comment_share_ratio"
]
print("\nPearson correlations with target (raw):")
print(df[early_features].corrwith(target_raw).sort_values(ascending=False))

# ── Feature selection ─────────────────────────────────────────────────────────
# Switch USE_VELOCITY_MODEL to True to swap in the velocity feature set.
USE_VELOCITY_MODEL = False

if USE_VELOCITY_MODEL:
    features = [
        "early_engagement_velocity",
        "follower_count",
        "authority_log",
        "verified",
        "account_age_years"
    ]
    print("\n>> Using VELOCITY feature set.")
else:
    features = [
        "early_likes",
        "early_comments",
        "early_shares",
        "follower_count",
        "authority_log",
        "verified",
        "account_age_years"
    ]
    print("\n>> Using COMPONENT feature set.")

X = df[features]

# ── Multicollinearity check (VIF) ─────────────────────────────────────────────
# VIF > 10 is a common threshold indicating problematic multicollinearity.
vif_data = pd.DataFrame({
    "Feature": features,
    "VIF": [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
})
print("\nVariance Inflation Factors (VIF):")
print(vif_data.sort_values("VIF", ascending=False))
high_vif = vif_data[vif_data["VIF"] > 10]
if not high_vif.empty:
    print(">> WARNING: High VIF detected for:", high_vif["Feature"].tolist())
    print("   Consider removing or combining correlated features.")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ── Ridge Regression (auto-tuned alpha via RidgeCV) ──────────────────────────
# RidgeCV performs internal LOO or K-Fold CV to select the best alpha.
# This replaces the arbitrary alpha=1.0 used previously.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

alphas = np.logspace(-3, 3, 100)    # search space: 0.001 → 1000
ridge_model = RidgeCV(alphas=alphas, cv=5)
ridge_model.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_model.predict(X_test_scaled)

print(f"\n── Ridge Regression Results ──────────────────────────")
print(f"Best alpha (auto-selected): {ridge_model.alpha_:.4f}")
print(f"R²   (Ridge): {r2_score(y_test, y_pred_ridge):.4f}")
print(f"RMSE (Ridge): {root_mean_squared_error(y_test, y_pred_ridge):.4f}")

ridge_coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": ridge_model.coef_
}).sort_values("Coefficient", ascending=False)
print("\nRidge Coefficients (standardized):")
print(ridge_coef_df.to_string(index=False))

# ── Ridge Cross-Validation (full dataset) ────────────────────────────────────
X_scaled_full = scaler.fit_transform(X)
ridge_cv_model = RidgeCV(alphas=alphas, cv=5)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores_ridge = cross_val_score(
    ridge_cv_model, X_scaled_full, y, cv=kf, scoring="r2"
)
print(f"\nRidge CV R² scores : {np.round(cv_scores_ridge, 4)}")
print(f"Mean CV R²         : {cv_scores_ridge.mean():.4f} "
      f"(±{cv_scores_ridge.std():.4f})")

# ── Gradient Boosting ─────────────────────────────────────────────────────────
# Tree-based models do NOT require feature scaling.
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=SEED
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print(f"\n── Gradient Boosting Results ─────────────────────────")
print(f"R²   (Gradient Boosting): {r2_score(y_test, y_pred_gb):.4f}")
print(f"RMSE (Gradient Boosting): {root_mean_squared_error(y_test, y_pred_gb):.4f}")

gb_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": gb_model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nGradient Boosting Feature Importances:")
print(gb_importance_df.to_string(index=False))

residuals = y_test - y_pred_gb
plt.scatter(y_pred_gb, residuals, alpha=0.3, s=10)
plt.axhline(0, color='red', linewidth=1)
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual plot — Gradient Boosting")
plt.savefig("residuals.png", dpi=150)

# ── Gradient Boosting Cross-Validation ───────────────────────────────────────
# Previously missing — single-split GB results are sensitive to the random seed.
cv_scores_gb = cross_val_score(
    GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, random_state=SEED
    ),
    X, y, cv=kf, scoring="r2"
)
print(f"\nGradient Boosting CV R² scores : {np.round(cv_scores_gb, 4)}")
print(f"Mean CV R²                     : {cv_scores_gb.mean():.4f} "
      f"(±{cv_scores_gb.std():.4f})")

# ── Model comparison summary ──────────────────────────────────────────────────
print("\n── Model Comparison Summary ──────────────────────────")
summary = pd.DataFrame({
    "Model": ["Ridge (held-out)", "Ridge (5-fold CV)", "Gradient Boosting (held-out)", "Gradient Boosting (5-fold CV)"],
    "R²":    [
        r2_score(y_test, y_pred_ridge),
        cv_scores_ridge.mean(),
        r2_score(y_test, y_pred_gb),
        cv_scores_gb.mean()
    ],
    "RMSE": [
        root_mean_squared_error(y_test, y_pred_ridge),
        np.nan,
        root_mean_squared_error(y_test, y_pred_gb),
        np.nan
    ]
})
print(summary.to_string(index=False))

# ── Feature Importance Plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ridge coefficients
colors_ridge = ["steelblue" if c >= 0 else "tomato" for c in ridge_coef_df["Coefficient"]]
axes[0].barh(ridge_coef_df["Feature"], ridge_coef_df["Coefficient"], color=colors_ridge)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_title("Ridge Coefficients (Standardized)")
axes[0].set_xlabel("Coefficient Value")
axes[0].invert_yaxis()

# Gradient Boosting importances
axes[1].barh(gb_importance_df["Feature"], gb_importance_df["Importance"], color="steelblue")
axes[1].set_title("Gradient Boosting Feature Importances")
axes[1].set_xlabel("Importance Score")
axes[1].invert_yaxis()

plt.suptitle("Early Engagement Signals — Feature Importance\nJose Diaz, CSUN 2026",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to feature_importance.png")