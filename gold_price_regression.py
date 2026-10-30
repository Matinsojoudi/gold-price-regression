"""
Urmia University - Deep Learning & Machine Learning
Assignment #1: Regression for Gold Price Prediction
Author: Matin Sojoudi
Student ID: 14014421025
Supervisor: Dr. Saleh Yousefi
"""

import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # 1. Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), 'GLD.csv')
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 2. Target Variable: Next Day's Closing Price
    df['Target_Next_Close'] = df['Close'].shift(-1)
    
    # Drop last row containing NaN target
    df_clean = df.dropna().copy().reset_index(drop=True)
    
    print(f"Dataset shape after preprocessing: {df_clean.shape}")
    print(f"Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")

    # 3. Features & Target
    # Note: Volume is scaled down to prevent float32 matmul overflow
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_clean['Volume'] = df_clean['Volume'] / 1e6  # Volume in Millions
    X = df_clean[feature_cols].astype(np.float64)
    y = df_clean['Target_Next_Close'].astype(np.float64)

    # 4. Train-Test Split (80% Train, 20% Test without shuffling for time-series)
    split_index = int(len(df_clean) * 0.8)
    
    X_train = np.ascontiguousarray(X.iloc[:split_index].values, dtype=np.float64)
    X_test = np.ascontiguousarray(X.iloc[split_index:].values, dtype=np.float64)
    y_train = np.ascontiguousarray(y.iloc[:split_index].values, dtype=np.float64)
    y_test = np.ascontiguousarray(y.iloc[split_index:].values, dtype=np.float64)
    dates_train = df_clean['Date'].iloc[:split_index]
    dates_test = df_clean['Date'].iloc[split_index:]

    print(f"Training samples: {len(X_train)} ({dates_train.min().date()} to {dates_train.max().date()})")
    print(f"Testing samples:  {len(X_test)} ({dates_test.min().date()} to {dates_test.max().date()})")

    # 5. Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # 6. Train Ridge Regression Model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)

    # 7. Model Evaluation
    def evaluate(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {'Model': name, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

    eval_lr = evaluate(y_test, y_pred_lr, "Linear Regression")
    eval_ridge = evaluate(y_test, y_pred_ridge, "Ridge Regression")

    eval_df = pd.DataFrame([eval_lr, eval_ridge])
    print("\n" + "="*50)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*50)
    print(eval_df.to_string(index=False))
    print("="*50 + "\n")

    # Sample Predictions
    sample_df = pd.DataFrame({
        'Date': dates_test.dt.strftime('%Y-%m-%d').values,
        'Actual Close': np.round(y_test, 2),
        'LR Predicted': np.round(y_pred_lr, 2),
        'Ridge Predicted': np.round(y_pred_ridge, 2)
    })
    print("Sample Next Day Predictions (First 10 test points):")
    print(sample_df.head(10).to_string(index=False))
    print("\nSample Next Day Predictions (Last 10 test points):")
    print(sample_df.tail(10).to_string(index=False))

    # 8. Visualization
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(14, 7), dpi=300)
    
    plt.plot(dates_test, y_test, label='Actual Next Day Close Price', color='#1f77b4', linewidth=2.0)
    plt.plot(dates_test, y_pred_lr, label=f"Linear Regression (R²={eval_lr['R2']})", color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.9)
    plt.plot(dates_test, y_pred_ridge, label=f"Ridge Regression (R²={eval_ridge['R2']})", color='#2ca02c', linestyle=':', linewidth=1.8, alpha=0.9)
    
    plt.title('Gold Price Prediction (SPDR Gold Shares - GLD): Actual vs. Predicted', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Gold Price (USD)', fontsize=12, labelpad=10)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, fontsize=11, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(__file__), 'actual_vs_predicted.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot successfully saved to: {plot_path}")

    # Residuals plot
    plt.figure(figsize=(14, 5), dpi=300)
    residuals_lr = y_test - y_pred_lr
    residuals_ridge = y_test - y_pred_ridge
    plt.plot(dates_test, residuals_lr, label='Residuals (Linear Regression)', color='#e74c3c', alpha=0.7, linewidth=1.2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Prediction Residuals (Actual - Predicted) Over Time', fontsize=13, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Error (USD)', fontsize=11)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    res_path = os.path.join(os.path.dirname(__file__), 'residuals.png')
    plt.savefig(res_path, dpi=300)
    print(f"Residuals plot saved to: {res_path}")

if __name__ == '__main__':
    main()
