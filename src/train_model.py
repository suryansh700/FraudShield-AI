# ============================================================
# FRAUDSHIELD AI
# FINANCIAL FRAUD DETECTION MODEL TRAINING ENGINE
# ============================================================

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "creditcard.csv"
)

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

REPORTS_DIR = os.path.join(
    PROJECT_DIR,
    "reports"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fraud_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "preprocessor.pkl"
)

METADATA_PATH = os.path.join(
    MODELS_DIR,
    "model_metadata.json"
)

RESULTS_PATH = os.path.join(
    REPORTS_DIR,
    "model_results.csv"
)

ROC_CURVE_PATH = os.path.join(
    REPORTS_DIR,
    "roc_curve.csv"
)

PR_CURVE_PATH = os.path.join(
    REPORTS_DIR,
    "precision_recall_curve.csv"
)

PREDICTIONS_PATH = os.path.join(
    REPORTS_DIR,
    "evaluation_predictions.csv"
)


# ============================================================
# CREATE PROJECT DIRECTORIES
# ============================================================

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ============================================================
# DISPLAY HEADER
# ============================================================

print()
print("=" * 72)
print("              FRAUDSHIELD AI - MODEL TRAINING")
print("=" * 72)
print()


# ============================================================
# STEP 1 - LOAD DATASET
# ============================================================

print(
    "[1/9] Loading transaction dataset..."
)


if not os.path.exists(
    DATA_PATH
):

    raise FileNotFoundError(
        "creditcard.csv was not found.\n"
        f"Expected path: {DATA_PATH}"
    )


data = pd.read_csv(
    DATA_PATH
)


if "Class" not in data.columns:

    raise ValueError(
        "Dataset must contain a 'Class' target column."
    )


print(
    "Dataset loaded successfully."
)

print(
    f"Total Transactions : {len(data):,}"
)

print(
    f"Total Features     : {len(data.columns) - 1}"
)

print()


# ============================================================
# STEP 2 - ANALYZE FRAUD DISTRIBUTION
# ============================================================

print(
    "[2/9] Analyzing fraud distribution..."
)


legitimate_transactions = int(
    (data["Class"] == 0).sum()
)

fraud_transactions = int(
    (data["Class"] == 1).sum()
)

fraud_percentage = (
    fraud_transactions
    / len(data)
) * 100


print(
    f"Legitimate Transactions : "
    f"{legitimate_transactions:,}"
)

print(
    f"Fraud Transactions      : "
    f"{fraud_transactions:,}"
)

print(
    f"Fraud Percentage        : "
    f"{fraud_percentage:.4f}%"
)

print()


# ============================================================
# STEP 3 - PREPARE FEATURES
# ============================================================

print(
    "[3/9] Preprocessing transaction data..."
)


X = data.drop(
    columns=["Class"]
).copy()

y = data[
    "Class"
].astype(int).copy()


feature_columns = X.columns.tolist()


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


preprocessor = StandardScaler()


X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


print(
    "Data preprocessing completed."
)

print(
    f"Training Transactions : {len(X_train):,}"
)

print(
    f"Testing Transactions  : {len(X_test):,}"
)

print()


# ============================================================
# STEP 4 - APPLY SMOTE
# ============================================================

print(
    "[4/9] Applying SMOTE..."
)


smote = SMOTE(
    random_state=42,
    sampling_strategy=0.10
)


X_train_balanced, y_train_balanced = (
    smote.fit_resample(
        X_train_processed,
        y_train
    )
)


balanced_legitimate = int(
    (y_train_balanced == 0).sum()
)

balanced_fraud = int(
    (y_train_balanced == 1).sum()
)


print(
    "SMOTE completed successfully."
)

print(
    f"Balanced Legitimate : "
    f"{balanced_legitimate:,}"
)

print(
    f"Balanced Fraud      : "
    f"{balanced_fraud:,}"
)

print(
    f"Balanced Samples    : "
    f"{len(y_train_balanced):,}"
)

print()


# ============================================================
# STEP 5 - DEFINE MODELS
# ============================================================

print(
    "[5/9] Initializing fraud detection models..."
)


models = {

    "Gradient Boosting": (
        HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    ),

    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
}


print(
    f"Models Initialized : {len(models)}"
)

print()


# ============================================================
# STEP 6 - TRAIN MODELS
# ============================================================

print(
    "[6/9] Training fraud detection models..."
)

print()


trained_models = {}


for model_name, model in models.items():

    print(
        f"Training {model_name}..."
    )

    model.fit(
        X_train_balanced,
        y_train_balanced
    )

    trained_models[
        model_name
    ] = model

    print(
        f"{model_name} training completed."
    )

    print()


# ============================================================
# STEP 7 - EVALUATE MODELS
# ============================================================

print(
    "[7/9] Evaluating trained models..."
)

print()


model_results = []

evaluation_cache = {}


for model_name, model in trained_models.items():

    predictions = model.predict(
        X_test_processed
    )

    fraud_probabilities = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        fraud_probabilities
    )


    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()


    model_results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "True Positives": int(tp),
            "False Positives": int(fp),
            "True Negatives": int(tn),
            "False Negatives": int(fn)
        }
    )


    evaluation_cache[
        model_name
    ] = {
        "predictions": predictions,
        "probabilities": fraud_probabilities
    }


results_df = pd.DataFrame(
    model_results
)


results_df = results_df.sort_values(
    by=[
        "F1 Score",
        "ROC AUC"
    ],
    ascending=False
).reset_index(
    drop=True
)


print(
    results_df.to_string(
        index=False
    )
)

print()


# ============================================================
# STEP 8 - SELECT BEST MODEL
# ============================================================

print(
    "[8/9] Selecting best fraud detection model..."
)


best_model_name = str(
    results_df.iloc[0][
        "Model"
    ]
)

best_model = trained_models[
    best_model_name
]


selected_results = (
    results_df[
        results_df["Model"]
        == best_model_name
    ]
    .iloc[0]
)


best_predictions = (
    evaluation_cache[
        best_model_name
    ][
        "predictions"
    ]
)

best_probabilities = (
    evaluation_cache[
        best_model_name
    ][
        "probabilities"
    ]
)


print(
    f"Selected Model : {best_model_name}"
)

print(
    f"F1 Score       : "
    f"{selected_results['F1 Score']:.4f}"
)

print(
    f"ROC AUC        : "
    f"{selected_results['ROC AUC']:.4f}"
)

print()


# ============================================================
# GENERATE ROC CURVE DATA
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    best_probabilities
)


roc_curve_df = pd.DataFrame(
    {
        "False Positive Rate": fpr,
        "True Positive Rate": tpr,
        "Threshold": roc_thresholds
    }
)


roc_curve_df.to_csv(
    ROC_CURVE_PATH,
    index=False
)


# ============================================================
# GENERATE PRECISION RECALL CURVE DATA
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        best_probabilities
    )
)


pr_threshold_column = np.append(
    pr_thresholds,
    np.nan
)


precision_recall_df = pd.DataFrame(
    {
        "Precision": precision_curve,
        "Recall": recall_curve,
        "Threshold": pr_threshold_column
    }
)


precision_recall_df.to_csv(
    PR_CURVE_PATH,
    index=False
)


# ============================================================
# SAVE EVALUATION PREDICTIONS
# ============================================================

evaluation_predictions = X_test.copy()


evaluation_predictions[
    "Actual_Class"
] = y_test.values


evaluation_predictions[
    "Predicted_Class"
] = best_predictions


evaluation_predictions[
    "Fraud_Probability"
] = best_probabilities


evaluation_predictions[
    "Risk_Score"
] = (
    best_probabilities * 100
)


evaluation_predictions[
    "Risk_Level"
] = pd.cut(
    evaluation_predictions[
        "Fraud_Probability"
    ],
    bins=[
        -0.01,
        0.40,
        0.70,
        0.90,
        1.01
    ],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
)


evaluation_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# STEP 9 - SAVE MODEL ARTIFACTS
# ============================================================

print(
    "[9/9] Saving FraudShield AI artifacts..."
)


joblib.dump(
    best_model,
    MODEL_PATH
)


joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


model_metadata = {

    "project_name": "FraudShield AI",

    "model_name": best_model_name,

    "model_class": (
        best_model.__class__.__name__
    ),

    "dataset": "creditcard.csv",

    "total_transactions": int(
        len(data)
    ),

    "total_features": int(
        len(feature_columns)
    ),

    "feature_columns": (
        feature_columns
    ),

    "fraud_transactions": int(
        fraud_transactions
    ),

    "legitimate_transactions": int(
        legitimate_transactions
    ),

    "fraud_percentage": float(
        fraud_percentage
    ),

    "training_transactions": int(
        len(X_train)
    ),

    "testing_transactions": int(
        len(X_test)
    ),

    "accuracy": float(
        selected_results[
            "Accuracy"
        ]
    ),

    "precision": float(
        selected_results[
            "Precision"
        ]
    ),

    "recall": float(
        selected_results[
            "Recall"
        ]
    ),

    "f1_score": float(
        selected_results[
            "F1 Score"
        ]
    ),

    "roc_auc": float(
        selected_results[
            "ROC AUC"
        ]
    ),

    "true_positives": int(
        selected_results[
            "True Positives"
        ]
    ),

    "false_positives": int(
        selected_results[
            "False Positives"
        ]
    ),

    "true_negatives": int(
        selected_results[
            "True Negatives"
        ]
    ),

    "false_negatives": int(
        selected_results[
            "False Negatives"
        ]
    ),

    "preprocessing": (
        "StandardScaler"
    ),

    "imbalance_strategy": (
        "SMOTE"
    ),

    "smote_sampling_strategy": 0.10,

    "classification_threshold": 0.50,

    "risk_thresholds": {

        "low": 0.40,

        "medium": 0.70,

        "high": 0.90
    }
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as metadata_file:

    json.dump(
        model_metadata,
        metadata_file,
        indent=4
    )


print()
print("=" * 72)
print("                  MODEL TRAINING COMPLETE")
print("=" * 72)
print()

print(
    f"BEST MODEL : {best_model_name}"
)

print()

print(
    f"Model Path       : {MODEL_PATH}"
)

print(
    f"Preprocessor Path: {PREPROCESSOR_PATH}"
)

print(
    f"Metadata Path    : {METADATA_PATH}"
)

print(
    f"Results Path     : {RESULTS_PATH}"
)

print()

print(
    f"{'Model':<25}: "
    f"{best_model_name}"
)

print(
    f"{'Accuracy':<25}: "
    f"{selected_results['Accuracy']:.4f}"
)

print(
    f"{'Precision':<25}: "
    f"{selected_results['Precision']:.4f}"
)

print(
    f"{'Recall':<25}: "
    f"{selected_results['Recall']:.4f}"
)

print(
    f"{'F1 Score':<25}: "
    f"{selected_results['F1 Score']:.4f}"
)

print(
    f"{'ROC-AUC':<25}: "
    f"{selected_results['ROC AUC']:.4f}"
)

print(
    f"{'True Positives':<25}: "
    f"{int(selected_results['True Positives'])}"
)

print(
    f"{'False Positives':<25}: "
    f"{int(selected_results['False Positives'])}"
)

print(
    f"{'True Negatives':<25}: "
    f"{int(selected_results['True Negatives'])}"
)

print(
    f"{'False Negatives':<25}: "
    f"{int(selected_results['False Negatives'])}"
)

print()
print("=" * 72)

print(
    "FraudShield AI is ready for fraud prediction."
)

print("=" * 72)