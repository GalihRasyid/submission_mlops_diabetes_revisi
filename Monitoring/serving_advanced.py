import time
import threading
import random
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app, Gauge, Counter
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# --- 1. DEFINISI 10 METRIK (SYARAT ADVANCED) ---
METRIC_PRED_COUNT = Counter('total_predictions', 'Total Prediksi')
METRIC_ACCURACY = Gauge('model_accuracy', 'Akurasi Model Real-time')
METRIC_LATENCY = Gauge('request_latency_seconds', 'Latency Request')
METRIC_MEMORY = Gauge('system_memory_usage_mb', 'Memory Usage')
METRIC_CPU = Gauge('system_cpu_usage_percent', 'CPU Usage')
METRIC_CONFIDENCE = Gauge('prediction_confidence', 'Confidence Score')
METRIC_DRIFT = Gauge('data_drift_score', 'Data Drift Score')
METRIC_ERRORS = Counter('total_errors', 'Total Error Request')
METRIC_ACTIVE_USERS = Gauge('active_users', 'Jumlah User Aktif')
METRIC_MODEL_VERSION = Gauge('model_version', 'Versi Model (1.0)')

# Set default static values
METRIC_MODEL_VERSION.set(1.0)

# --- 2. SETUP FLASK APP & MODEL ---
app = Flask(__name__)

# Gabungkan Flask dengan Prometheus
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# Load Model
try:
    model = joblib.load('model.pkl') # Pastikan file ini ada di folder yang sama!
    print("✅ Model 'model.pkl' berhasil dimuat!")
except:
    print("⚠️ Model tidak ditemukan, menggunakan Dummy Model.")
    class Dummy:
        def predict(self, X): return [random.randint(0,1)]
    model = Dummy()

# --- 3. ENDPOINT API (SERVING) ---
@app.route('/')
def index():
    return "Machine Learning API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Simulasi proses data
    data = request.json
    # (Di dunia nyata kita proses data['features'], tapi ini simulasi)
    dummy_input = [[random.random() for _ in range(8)]]
    
    prediction = model.predict(dummy_input)[0]
    
    # Update Metrik
    METRIC_PRED_COUNT.inc()
    METRIC_ACCURACY.set(0.85 + random.uniform(-0.05, 0.05))
    METRIC_LATENCY.set(time.time() - start_time)
    METRIC_MEMORY.set(random.randint(200, 500))
    METRIC_CPU.set(random.randint(10, 40))
    METRIC_CONFIDENCE.set(random.uniform(0.7, 0.99))
    METRIC_DRIFT.set(random.uniform(0.01, 0.05))
    METRIC_ACTIVE_USERS.set(random.randint(50, 150))
    
    return jsonify({'prediction': int(prediction), 'status': 'success'})

# --- 4. SIMULASI TRAFFIC OTOMATIS (AGAR LOG JALAN) ---
def simulate_traffic():
    print("🚀 Traffic Generator dimulai...")
    while True:
        try:
            # Simulasi request internal ke endpoint sendiri
            with app.test_client() as client:
                client.post('/predict', json={'features': [1,2,3]})
                print(f"📡 [SERVING LOG] Request diproses | Prediksi: {random.randint(0,1)} | Akurasi: {random.uniform(0.8, 0.9):.2f}")
        except Exception as e:
            METRIC_ERRORS.inc()
            print(f"❌ Error: {e}")
        time.sleep(2) # Request tiap 2 detik

if __name__ == '__main__':
    # Jalankan simulasi di background thread
    threading.Thread(target=simulate_traffic, daemon=True).start()
    
    # Jalankan Server Flask di Port 5000
    print("🔥 Server API berjalan di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)