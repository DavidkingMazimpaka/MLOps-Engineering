import numpy as np
from .model import load_model
from .preprocessing import preprocess_data, apply_smote

def predict(data):
    model = load_model()
    input_data = np.array(data).reshape(1, -1)
    predictions = model.predict(input_data)
    return predictions

def retrain_model(X, y, model_path='model.h5', epochs=1):
    # Preprocess the data
    X_processed, _, _ = preprocess_data(X)
    
    # Apply SMOTE to handle class imbalance
    X_resampled, y_resampled = apply_smote(X_processed, y)
    
    # Load the model
    model = load_model(model_path)
    
    # Retrain the model
    model.fit(X_resampled, y_resampled, epochs=epochs)
    
    # Save the updated model
    model.save(model_path)
    
    return model
