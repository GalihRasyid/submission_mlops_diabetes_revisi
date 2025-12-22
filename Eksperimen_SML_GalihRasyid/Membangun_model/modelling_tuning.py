import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. SETUP AUTHENTICATION (HARDCORE MODE) ---
# PENTING: Masukkan Token Asli di bawah ini! Jangan pakai placeholder.
TOKEN_ASLI = "d1f669853cea910190197feb84d64f7cb5691026"  # <--- GANTI INI!!!

# Paksa Environment Variables sebelum import library lain yang sensitif
os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN_ASLI
os.environ["DAGSHUB_USER_TOKEN"] = TOKEN_ASLI

# Import library DagsHub/MLflow setelah env var diset
import dagshub
import mlflow
import mlflow.sklearn
from dagshub.auth import add_app_token

# Paksa Auth lagi biar yakin 100%
try:
    add_app_token(TOKEN_ASLI)
    print("✅ Token berhasil dipaksa masuk.")
except Exception as e:
    print(f"⚠️ Warning auth: {e}")

# Setup Repo
dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")

# --- 2. LOAD DATASET ---
try:
    df = pd.read_csv('preprocessing/diabetes_clean.csv')
    print("✅ Dataset ditemukan di folder 'preprocessing/'")
except FileNotFoundError:
    try:
        df = pd.read_csv('../preprocessing/diabetes_clean.csv')
        print("✅ Dataset ditemukan di path '../preprocessing/'")
    except FileNotFoundError:
        print("❌ Error: File diabetes_clean.csv tidak ditemukan!")
        exit()

# --- 3. SPLIT DATA ---
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. HYPERPARAMETER TUNING ---
param_grid = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 20}
]

mlflow.set_experiment("Diabetes_Hyperparameter_Tuning")

best_acc = 0
best_model = None

print("Mulai proses tuning...")

for params in param_grid:
    run_name = f"Run_n{params['n_estimators']}_d{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.autolog(disable=True)
        
        model = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                       max_depth=params['max_depth'], 
                                       random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        
        print(f"👉 {run_name} | Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            mlflow.sklearn.log_model(model, "best_model_diabetes")

# --- 5. SIMPAN ARTEFAK ---
if best_model:
    joblib.dump(best_model, 'model_diabetes.pkl')
    print(f"\n🏆 Tuning Selesai! Model disimpan.")