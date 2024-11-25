from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import logging

from src.model import load_model
from src.predict_retrain import predict, retrain_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Dependency to manage model loading
def get_model():
    try:
        return load_model()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail="Model initialization error")

class PredictRequest(BaseModel):
    data: List[float] = Field(..., min_items=1, description="Input features for prediction")

class RetrainRequest(BaseModel):
    data: List[List[float]] = Field(..., min_items=1, description="Training data features")
    labels: List[float] = Field(..., description="Corresponding training labels")

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Prediction Service. Use /predict and /retrain endpoints."}

@app.post('/predict')
def predict_endpoint(request: PredictRequest, model=Depends(get_model)):
    try:
        X_df = pd.DataFrame([request.data])
        predictions = predict(X_df)
        return {'predictions': predictions.tolist()}
    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post('/retrain')
def retrain_endpoint(request: RetrainRequest):
    try:
        X_df = pd.DataFrame(request.data)
        y_df = pd.Series(request.labels)
        
        # Validate input dimensions
        if len(X_df.columns) != len(request.data[0]):
            raise ValueError("Inconsistent input dimensions")
        
        # Retrain model
        retrained_model = retrain_model(X_df, y_df)
        
        return {"message": "Model retrained successfully"}
    except ValueError as e:
        logger.error(f"Retraining validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid training data: {str(e)}")
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
