from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from fastapi import FastAPI
import pandas as pd
import joblib

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://loan-prediction-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model=load("loan_prediction.pkl")

@app.get("/")
def test():
    return{"message":"api is working "}

@app.post("/predict")
def predict(self_employed: str,annual_income: int,loan_amount: int,loan_term: int,cibil_score: int,residential_assets_value: int,bank_asset_value: int):

    if self_employed=="yes" or "Yes":
        self_employed=1
    else:
        self_employed=0

    new_data=pd.DataFrame({
    "self_employed":[self_employed],
    "income_annum":[annual_income],
    "loan_amount":[loan_amount],
    "loan_term":[loan_term],
    "cibil_score":[cibil_score],
    "residential_assets_value":[residential_assets_value],
    "bank_asset_value":[bank_asset_value]
    })

    prediction=model.predict(new_data)

    if prediction[0]==1:
        loan="Loan Approved"
    else:
        loan="Loan Rejected"

    return{
    "self_employed":self_employed,
    "income_annum":annual_income,
    "loan_amount":loan_amount,
    "loan_term":loan_term,
    "cibil_score":cibil_score,
    "residential_assets_value":residential_assets_value,
    "bank_asset_value":bank_asset_value,
    "prediction":float(prediction[0]),
    "Loan":loan
    }

