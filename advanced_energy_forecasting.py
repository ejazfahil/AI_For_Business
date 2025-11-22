"""
Advanced Energy Load Forecasting
Using state-of-the-art time series models

Based on latest research:
- LightGBM for gradient boosting (faster and more accurate than Random Forest)
- Feature engineering with lag features and rolling statistics
- Time-based cross-validation
- SHAP for feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('energydata_complete.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Feature Engineering
def create_time_features(df):
    """Create comprehensive time-based features"""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    return df

def create_lag_features(df, target_col='Appliances', lags=[1, 2, 3, 6, 12, 24]):
    """Create lag features for time series"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col='Appliances', windows=[3, 6, 12, 24]):
    """Create rolling statistics features"""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    return df

# Apply feature engineering
print("\nCreating features...")
df = create_time_features(df)
df = create_lag_features(df)
df = create_rolling_features(df)

# Drop rows with NaN values (from lag and rolling features)
df = df.dropna()

print(f"Dataset shape after feature engineering: {df.shape}")

# Prepare data for modeling
target = 'Appliances'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

# Train LightGBM model
print("\n" + "="*50)
print("Training LightGBM Model")
print("="*50)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'max_depth': 8,
    'min_child_samples': 20
}

# Cross-validation
cv_scores = []
feature_importance_list = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Predictions
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    cv_scores.append({'fold': fold + 1, 'rmse': rmse, 'mae': mae, 'r2': r2})
    
    # Store feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    })
    feature_importance_list.append(importance_df)

# Print average CV scores
cv_df = pd.DataFrame(cv_scores)
print("\n" + "="*50)
print("Cross-Validation Results")
print("="*50)
print(cv_df)
print(f"\nAverage RMSE: {cv_df['rmse'].mean():.2f} (+/- {cv_df['rmse'].std():.2f})")
print(f"Average MAE: {cv_df['mae'].mean():.2f} (+/- {cv_df['mae'].std():.2f})")
print(f"Average R²: {cv_df['r2'].mean():.4f} (+/- {cv_df['r2'].std():.4f})")

# Train final model on all data
print("\n" + "="*50)
print("Training Final Model on All Data")
print("="*50)

train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=model.best_iteration
)

# Feature importance analysis
print("\n" + "="*50)
print("Top 20 Most Important Features")
print("="*50)

# Average feature importance across folds
avg_importance = pd.concat(feature_importance_list).groupby('feature')['importance'].mean().sort_values(ascending=False)
print(avg_importance.head(20))

# Save feature importance plot
plt.figure(figsize=(10, 8))
avg_importance.head(20).plot(kind='barh')
plt.xlabel('Feature Importance (Gain)')
plt.title('Top 20 Most Important Features for Energy Load Forecasting')
plt.tight_layout()
plt.savefig('feature_importance_energy.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance_energy.png'")

# Save the model
final_model.save_model('advanced_energy_model.txt')
print("Model saved as 'advanced_energy_model.txt'")

# SHAP analysis (optional - can be slow)
"""
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X.sample(1000))

# Summary plot
shap.summary_plot(shap_values, X.sample(1000), show=False)
plt.savefig('shap_summary_energy.png', dpi=300, bbox_inches='tight')
print("SHAP summary plot saved as 'shap_summary_energy.png'")
"""

print("\n" + "="*50)
print("Advanced Energy Load Forecasting Complete!")
print("="*50)
