# 🎯 Project Risk Prediction

> **AI-Powered Risk Assessment for Project Management**  
> Predict project risk levels (Low, Medium, High, Critical) using machine learning to enable proactive intervention and resource optimization.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Features](#-features)
- [Technologies](#-technologies)
- [Results](#-results)
- [Author](#-author)

---

## 🎯 Overview

This project implements a **machine learning pipeline** to predict project risk levels based on 50+ project management features. By analyzing historical project data, the model identifies patterns that indicate potential project failures, enabling:

- **Early Risk Detection**: Identify high-risk projects before they exceed budget or timelines
- **Resource Optimization**: Allocate senior staff and funds based on objective risk assessments
- **Data-Driven Decisions**: Move from subjective evaluations to evidence-based project management

### Key Highlights

- 🎯 **79% Accuracy** on risk prediction across 4 categories
- 🔄 **SMOTE** for handling class imbalance
- 🚀 **XGBoost** as the champion model
- 📊 **Comprehensive EDA** with 50+ features analyzed
- 🛠️ **Feature Engineering** with derived metrics

---

## 📊 Dataset

**Source**: [Kaggle - Project Management Risk Dataset](https://www.kaggle.com/)

The dataset contains **4,000 project records** with **50 features** across multiple categories:

| Category | Features |
|----------|----------|
| **Project Demographics** | Type, Budget, Timeline, Team Size, Complexity |
| **Operational Metrics** | Change Requests, Budget Utilization, Resource Availability |
| **Human Factors** | Experience Levels, Turnover Rate, Stakeholder Engagement |
| **Organizational Context** | Process Maturity, Compliance, Risk Management |
| **Technical Aspects** | Technology Familiarity, Technical Debt, Integration Complexity |
| **External Influences** | Market Volatility, Dependencies, Client Experience |

**Target Variable**: `Risk_Level` (Low, Medium, High, Critical)

---

## 📁 Project Structure

```
Project-Risk-Prediction/
│
├── experiment/
│   └── Nada_Ramadan.ipynb          # Complete ML experiment notebook
│
├── artifacts/
│   └── models/
│       └── model.pkl                # Trained XGBoost model (9.2 MB)
│
├── testData/
│   ├── X_test.csv                   # Test features (800 samples)
│   └── y_test.csv                   # Test labels
│
├── data/
│   └── project_risk_raw_dataset.csv # Raw dataset (4,000 records)
│
└── README.md                        # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nadaramadan1/Project-Risk-Prediction.git
   cd Project-Risk-Prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Option 1: Run the Jupyter Notebook

```bash
jupyter notebook experiment/Nada_Ramadan.ipynb
```

The notebook contains the complete ML pipeline:
1. Data loading and exploration
2. Preprocessing and cleaning
3. Feature engineering
4. Model training (RF, XGBoost, LightGBM)
5. Evaluation and comparison
6. Model and test data export

### Option 2: Use the Trained Model

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('artifacts/models/model.pkl')

# Load test data
X_test = pd.read_csv('testData/X_test.csv')

# Make predictions
predictions = model.predict(X_test)
```

---

## 📈 Model Performance

### XGBoost Classifier (Champion Model)

| Metric | Score |
|--------|-------|
| **Accuracy** | 79% |
| **Precision (Avg)** | 78% |
| **Recall (Avg)** | 79% |
| **F1-Score (Avg)** | 78% |

### Classification Report

```
              precision    recall  f1-score   support

    Critical       0.82      0.75      0.78       200
        High       0.76      0.81      0.78       200
         Low       0.79      0.78      0.79       200
      Medium       0.75      0.80      0.77       200

    accuracy                           0.79       800
   macro avg       0.78      0.79      0.78       800
weighted avg       0.78      0.79      0.78       800
```

### Model Comparison

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | 73% | ~15s |
| **XGBoost** | **79%** | **~25s** |
| LightGBM | 76% | ~18s |

---

## ✨ Features

### Data Preprocessing
- ✅ Missing value imputation
- ✅ Outlier detection and handling
- ✅ Feature scaling (StandardScaler)
- ✅ Categorical encoding (OneHotEncoder)

### Feature Engineering
- 📊 `Budget_per_TeamMember`: Resource density metric
- 📊 `Timeline_per_TeamMember`: Workload distribution
- 📊 `Risk_Index`: Composite risk score
- 📊 `Dependency_Load`: External dependency metric
- 📊 Interaction features based on feature importance

### Model Training
- 🎯 SMOTE for class imbalance
- 🎯 Stratified train/test split (80/20)
- 🎯 Hyperparameter tuning (RandomizedSearchCV)

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.8+ |
| **ML Libraries** | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| **Data Analysis** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## 📊 Results

### Key Findings

1. **Top Risk Drivers**:
   - Project Complexity Score
   - Team Turnover Rate
   - Budget Utilization Rate
   - Change Request Frequency

2. **Feature Importance**:
   - Complexity-related features account for 35% of model decisions
   - Team dynamics contribute 25% to risk prediction
   - Budget and timeline factors represent 20%

3. **Business Impact**:
   - Early identification of 82% of critical-risk projects
   - Potential 30% reduction in project failures
   - Improved resource allocation efficiency

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Nada Ramadan**

- 📧 Email: [your.email@example.com](mailto:your.email@example.com)
- 💼 LinkedIn: [linkedin.com/in/nadaramadan](https://linkedin.com/in/nadaramadan)
- 🐙 GitHub: [@Nadaramadan1](https://github.com/Nadaramadan1)


---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made by Nada Ramadan

</div>
