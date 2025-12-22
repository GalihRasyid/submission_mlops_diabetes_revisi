import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Load Data
try:
    df = pd.read_csv('diabetes.csv')
    print("Data dimuat.")
except:
    print("File tidak ditemukan.")
    exit()

# 2. Preprocessing (LOGIKA INI HARUS SAMA DENGAN NOTEBOOK)
# Masalah di dataset Diabetes: nilai 0 di kolom tertentu adalah missing value
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='mean')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

# Scaling
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_clean = pd.DataFrame(X_scaled, columns=X.columns)
df_clean['Outcome'] = y

# 3. Simpan Hasil (PENTING: Save ke CSV baru)
df_clean.to_csv('diabetes_clean.csv', index=False)
print("Preprocessing Selesai! Disimpan ke diabetes_clean.csv")