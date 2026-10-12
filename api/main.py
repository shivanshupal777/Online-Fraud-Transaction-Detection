from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
 
app = FastAPI()
model = joblib.load('../models/model.pkl')
 
class TransactionData(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
 
@app.get("/health")
def health_check():
    return {"status": "healthy"}
 
@app.post("/predict")
def predict_transaction(data: TransactionData):
    try:
        input_data = pd.DataFrame([data.model_dump()])
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        return {
            "predicted_class": int(prediction[0]),
            "fraud_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
 