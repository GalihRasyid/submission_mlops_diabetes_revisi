import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub
import os # <--- Tambah ini

# --- 1. SETUP AUTHENTICATION (PAKSA LOGIN) ---
# Masukkan Token yang tadi dicopy dari Settings DagsHub Galih
os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d1f669853cea910190197feb84d64f7cb5691026"

# Setup Repo
dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")

# --- 2. LOAD DATASET ---
try:
    df = pd.read_csv('preprocessing/diabetes_clean.csv')
except FileNotFoundError:
    df = pd.read_csv('../preprocessing/diabetes_clean.csv')

# --- 3. TRAINING ---
X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Diabetes_Basic_Autolog")

with mlflow.start_run(run_name="Basic_RandomForest"):
    mlflow.sklearn.autolog()
    
    print("Mulai training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"✅ Sukses! Akurasi: {accuracy_score(y_test, y_pred)}")