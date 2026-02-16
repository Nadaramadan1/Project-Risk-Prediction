import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_data(df):
    """
    Perform basic data cleaning:
    - Drop unnecessary columns (Project_ID, Tech_Environment_Stability)
    - Handle missing values
    """
    df = df.copy()
    
    # Drop columns as identified in the notebook
    cols_to_drop = ['Project_ID', 'Tech_Environment_Stability']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"Dropped columns: {existing_cols_to_drop}")
    
    # Impute missing values for specific columns
    # Based on notebook analysis:
    # 'Change_Control_Maturity' -> 'Unknown' (or mode, depending on notebook logic, but 'Unknown' is safer if categorical)
    # 'Risk_Management_Maturity' -> 'Unknown'
    
    fill_unknown_cols = ['Change_Control_Maturity', 'Risk_Management_Maturity']
    for col in fill_unknown_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            print(f"Filled missing values in {col} with 'Unknown'")

    return df
