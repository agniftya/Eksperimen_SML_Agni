from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn
import random
import time

app = FastAPI(title="Prometheus Exporter Restaurant API")

# --- MEMBUAT 3 METRIK WAJIB UNTUK DICODING ---
# 1. Metrik Counter: Menghitung total request
REQUEST_COUNT = Counter('api_requests_total', 'Total HTTP requests yang masuk')
# 2. Metrik Histogram: Mengukur kecepatan/waktu respon API
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Waktu proses request API')
# 3. Metrik Counter Custom: Menghitung sebaran hasil prediksi (High vs Low)
PREDICTION_COUNT = Counter('api_predictions_total', 'Total prediksi berdasarkan label kepuasan', ['label'])

class RestaurantData(BaseModel):
    Income: float
    AverageSpend: float
    ServiceRating: float
    FoodRating: float
    WaitTime: float

@app.get("/metrics")
def metrics():
    # Halaman ini yang akan dibaca oleh Prometheus nanti
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(data: RestaurantData):
    REQUEST_COUNT.inc() # Tambah hitungan total request
    start_time = time.time() # Mulai stopwatch

    # Simulasi prediksi
    score = (data.ServiceRating + data.FoodRating) / 2
    prediction = 1 if score >= 3.5 else 0
    label = "High" if prediction == 1 else "Low"
    
    PREDICTION_COUNT.labels(label=label).inc() # Tambah hitungan hasil High/Low

    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency) # Catat waktu proses

    return {"prediction": prediction, "satisfaction_label": label}

if __name__ == "__main__":
    # Kita jalankan di port 8001 agar tidak tabrakan dengan Inference.py
    uvicorn.run(app, host="127.0.0.1", port=8001)