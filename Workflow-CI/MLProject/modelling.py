import os
import joblib
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dagshub.auth import add_app_token

# --- 1. SETUP AUTHENTICATION ---

TOKEN_ASLI = "d1f669853cea910190197feb84d64f7cb5691026" 

os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN_ASLI
os.environ["DAGSHUB_USER_TOKEN"] = TOKEN_ASLI

try:
    add_app_token(TOKEN_ASLI)
except Exception:
    pass

dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")

# --- 2. LOAD DATA ---
# Perhatikan path baru sesuai struktur folder Reviewer
try:
    df = pd.read_csv('diabetes_preprocessing/diabetes_clean.csv')
except FileNotFoundError:
    # Backup jika dijalankan dari root
    df = pd.read_csv('../diabetes_preprocessing/diabetes_clean.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAINING & LOGGING (FIX KRITERIA 2) ---
mlflow.set_experiment("Diabetes_Fix_Artifacts")

with mlflow.start_run(run_name="Run_Fixed_Model"):
    # Matikan autolog biar kita manual saja (lebih pasti)
    mlflow.sklearn.autolog(disable=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Accuracy: {acc}")
    
    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    
    # --- INI SOLUSINYA: SIMPAN MODEL SECARA EKSPLISIT ---
    # Ini akan membuat folder 'model' di DagsHub Artifacts
    mlflow.sklearn.log_model(model, "model") 
    
    # Simpan file lokal juga (untuk backup)
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print("✅ Model berhasil disimpan ke MLflow Artifacts.")