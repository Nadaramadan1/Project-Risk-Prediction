import sys
import os

# Add current directory to sys.path so that imports in predict.py (like 'from data_preprocessing import ...') work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from predict import predict # Now we can import directly if src is in path, or use relative. 
# But since we added src to path, 'import predict' might work if we are inside src? 
# No, if we run from root, src is not in path. 
# If we add os.path.dirname(__file__) (which is src) to path, then 'import predict' works.


app = FastAPI(title="Project Risk Prediction API", description="API to predict project risk level")

# Define the input data model using Pydantic
class ProjectData(BaseModel):
    Project_Type: str
    Team_Size: int
    Project_Budget_USD: float
    Estimated_Timeline_Months: float
    Complexity_Score: float
    Stakeholder_Count: int
    Methodology_Used: str
    Team_Experience_Level: str
    Past_Similar_Projects: int
    External_Dependencies_Count: int
    Change_Request_Frequency: float
    Project_Phase: str
    Requirement_Stability: str
    Team_Turnover_Rate: float
    Vendor_Reliability_Score: float
    Historical_Risk_Incidents: int
    Communication_Frequency: float
    Regulatory_Compliance_Level: str
    Technology_Familiarity: str
    Geographical_Distribution: int
    Stakeholder_Engagement_Level: str
    Schedule_Pressure: float
    Budget_Utilization_Rate: float
    Executive_Sponsorship: str
    Funding_Source: str
    Market_Volatility: float
    Integration_Complexity: float
    Resource_Availability: float
    Priority_Level: str
    Organizational_Change_Frequency: float
    Cross_Functional_Dependencies: int
    Previous_Delivery_Success_Rate: float
    Technical_Debt_Level: float
    Project_Manager_Experience: str
    Org_Process_Maturity: str
    Data_Security_Requirements: str
    Key_Stakeholder_Availability: str
    Contract_Type: str
    Resource_Contention_Level: str
    Industry_Volatility: str
    Client_Experience_Level: str
    Change_Control_Maturity: str
    Risk_Management_Maturity: str
    Team_Colocation: str
    Documentation_Quality: str
    Project_Start_Month: int
    Current_Phase_Duration_Months: int
    Seasonal_Risk_Factor: float

@app.get("/")
def read_root():
    return {"message": "Project Risk Prediction API is running"}

@app.post("/predict")
def predict_risk(data: ProjectData):
    try:
        # Convert Pydantic model to Dict, then to DataFrame (list of 1 dict)
        data_dict = data.dict()
        
        # We need to replicate the calling convention of src.predict.predict
        # It expects raw data that it will clean and feature-engineer
        # The predict function expects 'data' which can be a dict or list of dicts or DF
        
        # Define model path (assuming it's in a standard location relative to this file)
        # In Docker, we'll copy models to a specific path or map a volume.
        # For local dev, we assume it's in 'models/stacking_classifier_model.pkl' or similar
        # But wait, predict.py loads it. Let's check where predict.py expects it or if we pass it.
        # predict.predict(model_path, data)
        
        model_path = os.path.join("models", "stacking_classifier_pipeline.pkl") # Standardize this name
        # If the file doesn't exist there, we might need to adjust.
        # For now, let's assume the user has trained the model and it exists.
        
        if not os.path.exists(model_path):
             # Try looking in 'artifacts' or root if 'models' doesn't exist
             if os.path.exists("stacking_classifier_pipeline.pkl"):
                 model_path = "stacking_classifier_pipeline.pkl"
             else:
                 # Fallback for now or error
                 pass

        predictions, probabilities = predict(model_path, data_dict)
        
        return {
            "prediction": predictions[0],
            "probability": probabilities[0].tolist() if probabilities is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
