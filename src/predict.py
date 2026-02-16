import joblib
import pandas as pd
import os
import sys

# Add src to path to allow import if needed, though loading pipeline via joblib usually handles dependencies if installed
# But we might need custom functions if they were part of the pickled object's classes (less likely here as we use sklearn classes)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def predict(model_path, data):
    """
    Predict risk for new data.
    data: DataFrame or list of dicts
    """
    artifact = load_model(model_path)
    pipeline = artifact['pipeline']
    le = artifact['label_encoder']
    
    # If data is dict, convert to DF
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data
        
    # Preprocessing is part of the pipeline, so we just pass raw-ish data
    # (assuming clean_data and generate_features logic needs to be applied if not in pipeline)
    # WAIT: Our train.py applies clean_data and generate_features BEFORE the pipeline.
    # We need to replicate that here.
    
    from data_preprocessing import clean_data
    from feature_engineering import generate_features
    
    df = clean_data(df)
    df = generate_features(df)
    
    predictions_encoded = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)
    
    predictions = le.inverse_transform(predictions_encoded)
    
    return predictions, probabilities

if __name__ == "__main__":
    # Example usage
    # Define a sample input (would usually come from API/CLI)
    # For now, just placeholder
    print("Predict script ready. Import 'predict' function to use.")
