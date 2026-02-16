import pandas as pd
import numpy as np

def generate_features(df):
    """
    Generate interaction features as defined in the notebook.
    """
    df = df.copy()
    
    # Interaction features identified from the notebook
    
    # Complexity * Turnover
    if 'Complexity_Score' in df.columns and 'Team_Turnover_Rate' in df.columns:
        df['Complexity_Turnover'] = df['Complexity_Score'] * df['Team_Turnover_Rate']
    
    # Complexity * Timeline
    if 'Complexity_Score' in df.columns and 'Estimated_Timeline_Months' in df.columns:
        df['Complexity_Timeline'] = df['Complexity_Score'] * df['Estimated_Timeline_Months']
        
    # Budget / Timeline (avoid division by zero)
    if 'Project_Budget_USD' in df.columns and 'Estimated_Timeline_Months' in df.columns:
        df['Budget_Timeline_Ratio'] = df['Project_Budget_USD'] / (df['Estimated_Timeline_Months'] + 1e-6)
        
    # Market Volatility * Integration Complexity
    if 'Market_Volatility' in df.columns and 'Integration_Complexity' in df.columns:
        df['Market_Integration'] = df['Market_Volatility'] * df['Integration_Complexity']
        
    # Additional interactions could be added here based on further notebook analysis if needed
    
    print("Feature engineering completed. New features added.")
    return df
