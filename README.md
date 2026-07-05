# 🛡️ FraudShield AI

### Machine Learning Financial Fraud Intelligence Platform

FraudShield AI is an end-to-end machine learning financial fraud detection and investigation platform designed to identify suspicious financial transactions, calculate fraud probability, prioritize high-risk activity, and provide explainable fraud intelligence through an interactive dashboard.

The project demonstrates a complete data science workflow — from transaction preprocessing and class imbalance handling to model comparison, fraud probability analysis, risk intelligence, and explainable AI.

---

## 🚀 Live Application

🔗 **Live Demo:** [Launch FraudShield AI](https://fraudshield-ai-700.streamlit.app/)

🔗 **GitHub Repository:** [View Source Code](https://github.com/suryansh700/FraudShield-AI)

---

## 📌 Project Overview

Financial fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small percentage of the complete transaction population.

FraudShield AI was developed to demonstrate how machine learning can support financial fraud investigation by transforming raw transaction data into prioritized fraud intelligence.

The system evaluates financial transaction patterns using a trained machine learning model and provides:

- Fraud transaction identification
- Fraud probability scoring
- Risk-based transaction prioritization
- High-risk investigation queues
- Fraud severity intelligence
- Global model explainability
- Transaction-level case explanations
- Model performance analysis

The platform is designed as a **decision-support intelligence system** rather than an automated financial enforcement system.

---

## 🎯 Problem Statement

Financial institutions process large volumes of transactions every day.

Manually reviewing every transaction is inefficient and traditional rule-based systems may fail to identify complex fraud patterns.

The key challenges include:

- Extremely imbalanced transaction data
- Very limited fraudulent transaction examples
- High cost of missed fraud cases
- Large transaction populations
- Difficulty prioritizing suspicious activity
- Limited interpretability of machine learning predictions

FraudShield AI addresses these challenges by applying machine learning to calculate fraud probability and convert model predictions into operational fraud intelligence.

---

## 💡 Proposed Solution

FraudShield AI provides an intelligent fraud detection workflow that:

1. Processes historical financial transaction data
2. Prepares transaction features for machine learning
3. Handles class imbalance using SMOTE
4. Trains and compares multiple classification models
5. Selects the best fraud detection model
6. Calculates fraud probability for transactions
7. Converts probability into fraud risk intelligence
8. Prioritizes suspicious transactions for investigation
9. Explains global model behaviour
10. Generates case-level transaction explanations

---

## 🧠 Machine Learning Workflow

```text
Financial Transaction Dataset
            │
            ▼
     Data Validation
            │
            ▼
     Data Preprocessing
            │
            ▼
   Train-Test Data Split
            │
            ▼
  SMOTE Class Balancing
   (Training Data Only)
            │
            ▼
   Multiple ML Models
            │
            ▼
    Model Evaluation
            │
            ▼
  Best Model Selection
            │
            ▼
 Fraud Probability Engine
            │
            ▼
    Risk Intelligence
            │
            ▼
   Fraud Investigation
            │
            ▼
     Explainable AI
```

---

## 🏗️ System Architecture

FraudShield AI follows a modular machine learning architecture.

### 1. Transaction Dataset

Historical financial transaction records form the machine learning model development population.

### 2. Data Preprocessing

Transaction features are processed using a fitted preprocessing pipeline generated during model training.

### 3. Class Imbalance Handling

SMOTE is applied only to the training population to improve the representation of fraudulent transaction examples.

### 4. Model Comparison

Multiple machine learning classification algorithms are trained and evaluated using fraud-focused performance metrics.

### 5. Fraud Probability Engine

The selected machine learning model calculates fraud probability for evaluated financial transactions.

### 6. Risk Intelligence

Fraud probabilities are converted into operational risk segments for investigation prioritization.

### 7. Global Explainability

Permutation importance measures the global influence of transaction features on fraud detection performance.

### 8. Case-Level Explainability

Transaction perturbation analysis estimates how individual features influence fraud probability for a selected transaction.

---

## ✨ Key Features

### 🖥️ Financial Fraud Command Center

The Command Center provides a high-level overview of the fraud detection environment.

It displays:

- Transactions monitored
- Fraud signals
- Critical threats
- Average fraud risk
- Transaction classification
- Threat severity distribution
- Fraud intelligence visualizations

---

### 🔎 Fraud Investigation

The Fraud Investigation module provides a prioritized investigation workflow for suspicious financial activity.

Key capabilities include:

- High-risk transaction queue
- Fraud probability ranking
- Transaction-level risk analysis
- Suspicious transaction inspection
- Investigation prioritization
- Case-level fraud intelligence

---

### 🚨 Risk Intelligence

The Risk Intelligence module analyzes the overall fraud risk landscape.

The module provides:

- Fraud probability distribution
- Risk severity segmentation
- High-risk transaction analysis
- Critical transaction monitoring
- Model-driven transaction risk intelligence

Transactions are categorized into operational risk segments based on fraud probability.

---

### 🧠 Explainable AI

FraudShield AI includes global machine learning explainability.

Permutation importance is used to measure how strongly transaction features influence fraud detection performance.

The Explainable AI module provides:

- Top detection feature
- Number of features analyzed
- Positive feature influence
- Feature importance ranking
- Global model behaviour analysis

Global explainability describes the overall behaviour of the fraud detection model.

---

### 🔬 Case-Level Fraud Explainability

FraudShield AI also provides transaction-level explainability.

The system evaluates how individual transaction features influence fraud probability for a selected case.

The case explanation workflow provides:

- Transaction fraud probability
- Risk classification
- Feature-level influence
- Fraud probability change
- Supporting risk signals
- Reducing risk signals

This allows investigators to understand why a specific transaction received a high fraud probability.

---

### 📊 Model Performance Intelligence

The Model Performance module provides detailed machine learning evaluation results.

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve
- ROC Curve

Fraud detection performance is evaluated using metrics suitable for highly imbalanced classification problems.

---

## 🤖 Machine Learning Models

The following classification algorithms were evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Random Forest | Ensemble tree-based classification |
| Gradient Boosting | Sequential boosting-based fraud classification |

The final detection model was selected based on fraud-focused model evaluation metrics.

### Active Detection Model

**Gradient Boosting**

The trained model is stored as a serialized machine learning artifact and loaded by the FraudShield AI application for transaction evaluation.

---

## 📈 Model Performance

The selected fraud detection model achieved strong classification performance on the evaluation population.

| Metric | Score |
|---|---:|
| Accuracy | 99.92% |
| Precision | 73.50% |
| Recall | 87.76% |
| F1 Score | 80.00% |
| ROC-AUC | 98.28% |

### Why Accuracy Alone Is Not Enough

Financial fraud datasets are extremely imbalanced.

A model can achieve very high accuracy by predicting most transactions as legitimate.

Therefore, FraudShield AI focuses on:

- **Precision** — How many detected fraud signals are actually fraudulent
- **Recall** — How many fraudulent transactions are successfully detected
- **F1 Score** — Balance between precision and recall
- **ROC-AUC** — Ability to distinguish fraudulent and legitimate transactions

---

## ⚖️ Handling Class Imbalance

Fraud transactions represent only a very small percentage of the original transaction population.

To address this problem, FraudShield AI uses:

### SMOTE — Synthetic Minority Over-sampling Technique

SMOTE generates synthetic minority class examples to improve fraudulent transaction representation during model training.

### Important Data Science Practice

SMOTE is applied **only to the training dataset**.

The testing dataset retains its original class distribution.

This prevents data leakage and provides a more realistic model evaluation.

---

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized financial transaction features.

### Original Model Development Dataset

| Attribute | Value |
|---|---:|
| Transactions | 284,807 |
| Features | 30 |
| Fraud Transactions | 492 |
| Legitimate Transactions | 284,315 |
| Fraud Percentage | ~0.17% |

The original dataset is used for machine learning model development and evaluation.

Due to repository file size limitations, the complete original dataset is not stored directly in this repository.

### Deployment Demonstration Dataset

A smaller representative transaction population is used by the deployed FraudShield AI dashboard.

The deployment dataset retains fraud examples required to demonstrate:

- Fraud investigation
- Risk intelligence
- Fraud probability analysis
- Explainable AI workflows

The trained model remains based on the complete model development workflow.

---

## 🧪 Explainability Methodology

### Global Explainability

FraudShield AI uses **Permutation Importance** to analyze global model behaviour.

The process:

1. Evaluate baseline model performance
2. Shuffle one transaction feature
3. Re-evaluate model performance
4. Measure performance change
5. Rank features by influence

A larger permutation importance score indicates that disturbing the feature causes a greater change in fraud detection performance.

---

### Case-Level Explainability

Transaction-level explanations are generated using feature perturbation analysis.

The system:

1. Selects a transaction
2. Calculates baseline fraud probability
3. Perturbs individual transaction features
4. Recalculates fraud probability
5. Measures probability change
6. Ranks feature influence

This produces an interpretable fraud intelligence explanation for individual transactions.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Machine Learning | Scikit-learn |
| Class Imbalance | Imbalanced-learn |
| Visualization | Plotly |
| Application Framework | Streamlit |
| Model Serialization | Joblib / Pickle |
| Version Control | Git |
| Repository Hosting | GitHub |
| Application Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
FraudShield-AI/
│
├── assets/
│
├── data/
│   └── fraudshield_transactions.csv
│
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
│
├── notebooks/
│
├── reports/
│   ├── evaluation_predictions.csv
│   ├── feature_importance.csv
│   ├── model_results.csv
│   ├── precision_recall_curve.csv
│   ├── roc_curve.csv
│   └── sample_case_explanation.csv
│
├── src/
│   ├── __init__.py
│   ├── case_explainer.py
│   ├── create_deployment_data.py
│   ├── generate_explainability.py
│   ├── generate_metadata.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Local Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project

```bash
cd FraudShield-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run FraudShield AI

```bash
streamlit run app.py
```

The application will start locally in your browser.

---

## 🔄 Model Development Workflow

To reproduce the machine learning development workflow, obtain the original transaction dataset and place it inside:

```text
data/creditcard.csv
```

Then execute:

```bash
python src/train_model.py
```

Generate model metadata:

```bash
python src/generate_metadata.py
```

Generate global explainability artifacts:

```bash
python src/generate_explainability.py
```

Generate deployment transaction data:

```bash
python src/create_deployment_data.py
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔐 Responsible AI & Model Limitations

FraudShield AI is a machine learning and data science portfolio project.

Model-generated fraud probability represents **decision-support intelligence** and should not be treated as proof of financial misconduct.

Real-world financial fraud systems require additional components including:

- Human fraud analyst review
- Transaction context
- Customer behaviour history
- Identity intelligence
- Real-time feature pipelines
- Model monitoring
- Drift detection
- Governance controls
- Regulatory compliance
- Security controls

Machine learning predictions should support investigation decisions rather than automatically determine financial misconduct.

---

## 🔮 Future Enhancements

Potential future improvements include:

- Real-time transaction streaming
- REST API fraud scoring
- Database integration
- User authentication
- Analyst case management
- Investigation status tracking
- SHAP-based local explainability
- Model drift monitoring
- Fraud alert notifications
- Cloud-based model inference
- Automated model retraining pipeline
- MLOps integration

---

## 🎓 Project Learning Outcomes

This project demonstrates practical knowledge of:

- End-to-end machine learning development
- Financial fraud detection
- Imbalanced classification
- SMOTE oversampling
- Model comparison
- Machine learning evaluation
- Probability-based risk scoring
- Explainable AI
- Feature importance
- Transaction-level explanations
- Data visualization
- Streamlit application development
- Git version control
- Machine learning application deployment

---

## 👨‍💻 Author

### Suryansh Singh

B.Tech Information Technology Student  
Data Science & Machine Learning Enthusiast

**Areas of Interest:**

- Data Science
- Machine Learning
- Artificial Intelligence
- Data Analytics
- Python Development

---

## ⭐ Support

If you find FraudShield AI useful or interesting, consider giving the repository a ⭐.

Your support is appreciated.

---

## 📄 Disclaimer

FraudShield AI is developed for educational, research, and portfolio demonstration purposes.

The application does not provide financial, legal, or compliance advice and should not be used as a production fraud enforcement system without appropriate validation, governance, security, and regulatory controls.

---

<p align="center">
  <b>🛡️ FraudShield AI</b>
</p>

<p align="center">
  Machine Learning Financial Fraud Intelligence Platform
</p>

<p align="center">
  Built with Python • Scikit-learn • Streamlit • Explainable AI
</p>
