import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from data_preprocessing import load_data, clean_data
from feature_engineering import generate_features

# Base path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'project_risk_raw_dataset.csv') # User provided dataset
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stacking_model.joblib')

def train_model(data_path, model_path):
    # 1. Load and Clean Data
    print("Loading data...")
    df = load_data(data_path)
    if df is None:
        return
    
    df = clean_data(df)
    
    # 2. Feature Engineering
    print("Generating features...")
    df = generate_features(df)
    
    # 3. Split Data
    target_col = 'Risk_Level' # Adjust if validation shows different target name
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encoder for Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 4. Define Preprocessor
    # Identify categorical and numerical columns
    # Note: excluding the new features from specific lists to let automatic selector or hardcoded lists work
    # ideally we select by dtype or hardcoded list. For robustness, let's select by dtype.
    
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )
    
    # 5. Define Models (EXACT Hyperparameters from notebook)
    
    # RandomForest
    rf_params = {
        'n_estimators': 300,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'log2',
        'max_depth': None,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    best_rf = RandomForestClassifier(**rf_params)
    
    # XGBoost
    xgb_params = {
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 7,
        'scale_pos_weight': 9,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1
    }
    xgb = XGBClassifier(**xgb_params)
    
    # LightGBM
    lgbm_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_data_in_leaf': 10,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    lgbm = LGBMClassifier(**lgbm_params)
    
    # Stacking Classifier
    stack_model = StackingClassifier(
        estimators=[('rf', best_rf), ('xgb', xgb), ('lgbm', lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    # 6. Create Pipeline with SMOTE
    # Note: StackingClassifier essentially trains base models. 
    # SMOTE should be applied to the training data BEFORE the classifier.
    # We use imbalanced-learn pipeline for this.
    
    if len(X_train) < 50:
        print("Dataset too small for SMOTE. Skipping oversampling.")
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('classifier', stack_model)
        ])
    else:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', stack_model)
        ])
    
    # MLflow Tracking
    import mlflow
    import mlflow.sklearn
    
    # Set experiment (optional, otherwise uses Default)
    mlflow.set_experiment("Project_Risk_Prediction")
    
    with mlflow.start_run():
        # Log Parameters
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
        mlflow.log_params({f"lgbm_{k}": v for k, v in lgbm_params.items()})
        mlflow.log_param("stacking_cv", 5)
        
        # 7. Train
        print("Training Stacking Classifier...")
        pipeline.fit(X_train, y_train)
        
        # 8. Evaluate
        print("Evaluating model...")
        y_pred_encoded = pipeline.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
        y_test_orig = le.inverse_transform(y_test)
        
        print("\nStacking Ensemble Results:")
        report_dict = classification_report(y_test_orig, y_pred, output_dict=True)
        report_str = classification_report(y_test_orig, y_pred)
        print(report_str)
        
        # Log Metrics
        mlflow.log_metric("accuracy", report_dict["accuracy"])
        mlflow.log_metric("macro_avg_f1", report_dict["macro avg"]["f1-score"])
        mlflow.log_metric("weighted_avg_f1", report_dict["weighted avg"]["f1-score"])
        
        # Log Model
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(pipeline, "stacking_ensemble_model")
        
        # 9. Save Code locally as well
        print(f"Saving model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({'pipeline': pipeline, 'label_encoder': le}, model_path)
        print("Done.")

if __name__ == "__main__":
    train_model(DATA_PATH, MODEL_PATH)
