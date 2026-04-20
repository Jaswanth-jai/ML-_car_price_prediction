



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ------------------ LOAD MODEL ------------------
try:
    with open('car_price.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ------------------ APP ------------------
app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ INPUT MODEL ------------------
class CarInput(BaseModel):
    name: Literal[
        "Hyundai Santro Xing", "Mahindra Jeep CL550", "Hyundai Grand i10",
        "Ford EcoSport Titanium", "Ford Figo", "Hyundai Eon",
        "Maruti Suzuki Alto", "Skoda Fabia Classic",
        "Hyundai Elite i20", "Mahindra Scorpio SLE",
        "Audi A8", "Audi Q7",
        "Honda City", "Toyota Innova",
        "Renault Duster", "Volkswagen Polo",
        "BMW 3 Series", "Mercedes Benz"
    ] = Field(..., description="Full car name")

    year: int = Field(..., ge=1990, le=2025)
    kms_driven: int = Field(..., ge=0)
    fuel_type: Literal["Petrol", "Diesel", "LPG"]

# ------------------ ROUTES ------------------

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running 🚀"}

@app.post("/predict")
def predict_price(data: CarInput):
    try:
        # ✅ IMPORTANT: match training columns EXACTLY
        input_df = pd.DataFrame([{
            'name': data.name,
            'year': data.year,
            'kms_driven': data.kms_driven,
            'fuel_type': data.fuel_type
        }])

        # Debug (remove later)
        print("INPUT DATA:\n", input_df)

        # Prediction
        prediction = model.predict(input_df)[0]

        return {
            "prediction": round(float(prediction), 2)
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))