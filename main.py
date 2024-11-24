from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.predict_retrain import predict, retrain_model
from src.model import load_model

app = FastAPI()

model = load_model()

class PredictRequest(BaseModel):
    data: list

class RetrainRequest(BaseModel):
    data: list
    labels: list

@app.post('/predict')
def predict_endpoint(request: PredictRequest):
    try:
        # Create a DataFrame from the request data
        X_df = pd.DataFrame([request.data])
        
        # Make predictions
        predictions = predict(X_df)
        
        return {'predictions': predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/retrain')
def retrain_endpoint(request: RetrainRequest):
    try:
        # Convert the request data to DataFrame
        X_df = pd.DataFrame(request.data)
        y_df = pd.Series(request.labels)
        
        # Retrain the model
        model = retrain_model(X_df, y_df)
        
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
