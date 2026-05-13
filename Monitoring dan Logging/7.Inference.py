from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random

app = FastAPI(title="Agni Restaurant Satisfaction API")

# Schema input data
class RestaurantData(BaseModel):
    Income: float
    AverageSpend: float
    ServiceRating: float
    FoodRating: float
    WaitTime: float

@app.get("/")
def home():
    return {"status": "Online", "message": "API Agni siap melayani prediksi!"}

@app.post("/predict")
def predict(data: RestaurantData):
    # Simulasi logika prediksi sederhana
    score = (data.ServiceRating + data.FoodRating) / 2
    prediction = 1 if score >= 3.5 else 0
    
    return {
        "prediction": prediction,
        "satisfaction_label": "High" if prediction == 1 else "Low",
        "confidence": round(random.uniform(0.7, 0.95), 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)