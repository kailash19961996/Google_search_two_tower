from fastapi import FastAPI
import torch
from pydantic import BaseModel
from model.Inference import predict_passages
import numpy as np

# Instantiate FastAPI app
app = FastAPI()


class PredictionOut(BaseModel):
    docs: list

class PredictionIn(BaseModel):
    query: str

@app.get("/")
def home():
    return {"health_check": "OK"}

# Define inference endpoint
@app.post("/predict/", response_model=PredictionOut)
def predict(query: str):
    
    Documents = predict_passages(query)
    
    result = PredictionOut(docs=Documents)
    
    return result

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

