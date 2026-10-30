# تمرین اول یادگیری عمیق - پیش‌بینی قیمت طلا با رگرسیون
# دانشجو: متین سجودی - شماره دانشجویی: 14014421025
# استاد: دکتر صالح یوسفی

import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # خواندن داده‌های تاریخی طلا
    csv_path = os.path.join(os.path.dirname(__file__), 'GLD.csv')
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # ایجاد متغیر هدف: قیمت روز بعد
    df['Target_Next_Close'] = df['Close'].shift(-1)
    df_clean = df.dropna().copy().reset_index(drop=True)

    # ویژگی‌ها
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_clean['Volume'] = df_clean['Volume'] / 1e6  # مقیاس حجم به میلیون برای پایداری عددی
    X = df_clean[feature_cols].astype(np.float64)
    y = df_clean['Target_Next_Close'].astype(np.float64)

    # تقسیم به ۸۰٪ آموزش و ۲۰٪ تست (بدون شافل برای رعایت توالی زمانی)
    split_index = int(len(df_clean) * 0.8)
    X_train = np.ascontiguousarray(X.iloc[:split_index].values, dtype=np.float64)
    X_test = np.ascontiguousarray(X.iloc[split_index:].values, dtype=np.float64)
    y_train = np.ascontiguousarray(y.iloc[:split_index].values, dtype=np.float64)
    y_test = np.ascontiguousarray(y.iloc[split_index:].values, dtype=np.float64)
    dates_test = df_clean['Date'].iloc[split_index:]

    # آموزش مدل رگرسیون خطی
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # آموزش مدل رگرسیون ریج
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)

    # ارزیابی نتایج
    def evaluate(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {'Model': name, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

    eval_lr = evaluate(y_test, y_pred_lr, "Linear Regression")
    eval_ridge = evaluate(y_test, y_pred_ridge, "Ridge Regression")

    eval_df = pd.DataFrame([eval_lr, eval_ridge])
    print(eval_df.to_string(index=False))

    # رسم نمودار مقایسه
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual Close Price', color='navy', linewidth=1.8)
    plt.plot(dates_test, y_pred_lr, label='Linear Regression', color='darkorange', linestyle='--', linewidth=1.4)
    plt.plot(dates_test, y_pred_ridge, label='Ridge Regression', color='green', linestyle=':', linewidth=1.4)
    plt.title('Gold Price Prediction: Actual vs. Predicted (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'actual_vs_predicted.png'), dpi=200)

    # نمودار مانده‌ها
    plt.figure(figsize=(12, 4))
    plt.plot(dates_test, y_test - y_pred_lr, color='crimson', linewidth=1)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('Date')
    plt.ylabel('Residual (USD)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'residuals.png'), dpi=200)

if __name__ == '__main__':
    main()
