import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 1. Đọc dữ liệu
df = pd.read_csv('camera_cleaned.csv', encoding='utf-8')

# 2. Làm sạch dữ liệu
# Drop rows where critical variables are NaN
df = df.dropna(subset=['price_cleaned', 'pixel_cleaned', 'weight', 'max_video_fps', 'battery_life', 'sensor_size'])

# 3. Lọc ngoại lai IQR cho giá
Q1 = df['price_cleaned'].quantile(0.25)
Q3 = df['price_cleaned'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_clean = df[(df['price_cleaned'] >= lower) & (df['price_cleaned'] <= upper)].copy()
print(f"Số dòng ban đầu: {len(df)}, sau khi lọc outlier: {len(df_clean)}")

# 4. Feature engineering (chỉ dùng các đặc trưng hợp lệ, KHÔNG rò rỉ target)
df_clean['weight_per_mp'] = df_clean['weight'] / df_clean['pixel_cleaned']

# 5. Phân chia features & target
features = ['pixel_cleaned', 'weight', 'max_video_fps', 'battery_life', 'sensor_size', 'weight_per_mp']
target = 'price_cleaned'

X = df_clean[features].copy()
y = df_clean[target].copy()

# 6. Mã hóa cảm biến (LabelEncoder)
le = LabelEncoder()
X['sensor_size'] = le.fit_transform(X['sensor_size'])
# Đổi tên cột để giống file cũ
X.rename(columns={'sensor_size': 'sensor_size_encoded'}, inplace=True)

# Đảm bảo thứ tự cột cố định
column_order = ['pixel_cleaned', 'weight', 'max_video_fps', 'battery_life', 'sensor_size_encoded', 'weight_per_mp']
X = X[column_order]

# 7. Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 9. So sánh Linear Regression vs Random Forest
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_y_pred)
lr_mae = mean_absolute_error(y_test, lr_y_pred)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)

print(f"\nLinear Regression Test R2: {lr_r2:.4f}, MAE: {lr_mae:,.0f} Rp")
print(f"Random Forest Test R2: {rf_r2:.4f}, MAE: {rf_mae:,.0f} Rp")

# Chọn mô hình tốt hơn
if rf_r2 > lr_r2:
    best_model = rf
    print("\nChọn mô hình: Random Forest Regressor")
else:
    best_model = lr
    print("\nChọn mô hình: Linear Regression")

# Train lại mô hình trên toàn bộ tập dữ liệu để có độ chính xác cao nhất
best_model.fit(X_scaled, y)

# 10. Đóng gói mô hình và các encoder
joblib.dump(best_model, 'camera_price_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'sensor_encoder.joblib')

with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(X.columns))

print("\nĐóng gói mô hình thành công!")

# 11. Test nhanh chức năng predict thực tế
def test_predict(pixel, weight, max_fps, battery, sensor_size):
    try:
        encoded_sensor = le.transform([sensor_size])[0]
        # input với đúng thứ tự cột
        input_data = pd.DataFrame([[
            pixel, weight, max_fps, battery, encoded_sensor, weight / pixel
        ]], columns=column_order)
        
        scaled_input = scaler.transform(input_data)
        price_pred = best_model.predict(scaled_input)[0]
        return price_pred
    except Exception as e:
        print(f"Lỗi dự đoán: {e}")
        return None

# Test dự đoán máy ảnh: 24MP, nặng 500g, 30 fps, pin 400 mAh, cảm biến APS-C
print(f"Dự đoán máy 24MP, 500g, APS-C: {test_predict(24, 500, 30, 400, 'APS-C'):,.0f} Rp")
