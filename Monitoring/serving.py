import time
import random
import joblib
from prometheus_client import start_http_server, Gauge, Counter, Summary, Histogram

# --- 10 METRICS BERBEDA (SYARAT ADVANCED) ---
# 1. Business Metrics
ACCURACY = Gauge('model_accuracy', 'Model Accuracy')
PRECISION = Gauge('model_precision', 'Model Precision')
RECALL = Gauge('model_recall', 'Model Recall')
F1_SCORE = Gauge('model_f1', 'Model F1 Score')

# 2. System Metrics
MEMORY_USAGE = Gauge('system_memory_usage_mb', 'Memory Usage in MB')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU Usage Percent')

# 3. Operational Metrics
REQUEST_COUNT = Counter('total_requests', 'Total Requests Received')
PREDICTION_0_COUNT = Counter('prediction_class_0_count', 'Total Non-Diabetes Predictions')
PREDICTION_1_COUNT = Counter('prediction_class_1_count', 'Total Diabetes Predictions')
LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')

# Load Model
try:
    model = joblib.load('../Membangun_model/model_diabetes.pkl')
    print("Model Loaded.")
except:
    print("Model dummy active.")
    class Dummy:
        def predict(self, X): return [random.randint(0,1)]
    model = Dummy()

def process_request():
    # Simulasi Latency
    with LATENCY.time():
        # Input Dummy (8 Fitur)
        dummy_data = [[random.random() for _ in range(8)]]
        pred = model.predict(dummy_data)[0]
        
        # Update Counters
        REQUEST_COUNT.inc()
        if pred == 0:
            PREDICTION_0_COUNT.inc()
        else:
            PREDICTION_1_COUNT.inc()
        
        # Simulasi Metrics Model (Fluktuatif)
        ACCURACY.set(0.80 + random.uniform(-0.05, 0.05))
        PRECISION.set(0.75 + random.uniform(-0.05, 0.05))
        RECALL.set(0.70 + random.uniform(-0.05, 0.05))
        F1_SCORE.set(0.72 + random.uniform(-0.05, 0.05))
        
        # Simulasi System
        MEMORY_USAGE.set(random.uniform(200, 500))
        CPU_USAGE.set(random.uniform(10, 40))
        
        print(f"Pred: {pred} | Acc: {ACCURACY._value.get()}")

if __name__ == '__main__':
    start_http_server(8000)
    print("Server running on port 8000...")
    while True:
        process_request()
        time.sleep(2)