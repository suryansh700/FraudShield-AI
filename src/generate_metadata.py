import os
import json
from datetime import datetime

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)


MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "fraud_model.pkl"
)


RESULTS_PATH = os.path.join(
    PROJECT_DIR,
    "reports",
    "model_results.csv"
)


DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "creditcard.csv"
)


METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "model_metadata.json"
)


# ============================================================
# VALIDATE PROJECT FILES
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        "fraud_model.pkl was not found."
    )


if not os.path.exists(
    RESULTS_PATH
):

    raise FileNotFoundError(
        "model_results.csv was not found."
    )


if not os.path.exists(
    DATA_PATH
):

    raise FileNotFoundError(
        "creditcard.csv was not found."
    )


# ============================================================
# LOAD PROJECT ARTIFACTS
# ============================================================

print(
    "Loading FraudShield AI artifacts..."
)


model = joblib.load(
    MODEL_PATH
)


model_results = pd.read_csv(
    RESULTS_PATH
)


transaction_data = pd.read_csv(
    DATA_PATH
)


# ============================================================
# IDENTIFY SAVED MODEL
# ============================================================

model_class_name = (
    model.__class__.__name__
)


model_name_mapping = {

    "GradientBoostingClassifier":
        "Gradient Boosting",

    "HistGradientBoostingClassifier":
    "Gradient Boosting",

    "RandomForestClassifier":
        "Random Forest",

    "LogisticRegression":
        "Logistic Regression"

}


selected_model_name = (
    model_name_mapping.get(
        model_class_name,
        model_class_name
    )
)


print(
    f"Detected Model : {selected_model_name}"
)


# ============================================================
# FIND MODEL PERFORMANCE
# ============================================================

selected_results = model_results[
    model_results["Model"]
    == selected_model_name
]


if selected_results.empty:

    raise ValueError(
        f"Performance results for "
        f"{selected_model_name} were not found."
    )


selected_results = (
    selected_results.iloc[0]
)


# ============================================================
# CREATE MODEL METADATA
# ============================================================

feature_columns = [
    column
    for column in transaction_data.columns
    if column != "Class"
]


model_metadata = {

    "project_name":
        "FraudShield AI",

    "system_version":
        "1.0.0",

    "model_name":
        selected_model_name,

    "model_type":
        model_class_name,

    "training_metadata_generated":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

    "dataset_transactions":
        int(
            len(transaction_data)
        ),

    "feature_count":
        int(
            len(feature_columns)
        ),

    "target_column":
        "Class",

    "accuracy":
        float(
            selected_results["Accuracy"]
        ),

    "precision":
        float(
            selected_results["Precision"]
        ),

    "recall":
        float(
            selected_results["Recall"]
        ),

    "f1_score":
        float(
            selected_results["F1 Score"]
        ),

    "roc_auc":
        float(
            selected_results["ROC AUC"]
        ),

    "confusion_matrix": {

        "true_positives":
            int(
                selected_results[
                    "True Positives"
                ]
            ),

        "false_positives":
            int(
                selected_results[
                    "False Positives"
                ]
            ),

        "true_negatives":
            int(
                selected_results[
                    "True Negatives"
                ]
            ),

        "false_negatives":
            int(
                selected_results[
                    "False Negatives"
                ]
            )

    },

    "risk_thresholds": {

        "low": {
            "minimum": 0,
            "maximum": 29
        },

        "medium": {
            "minimum": 30,
            "maximum": 59
        },

        "high": {
            "minimum": 60,
            "maximum": 79
        },

        "critical": {
            "minimum": 80,
            "maximum": 100
        }

    },

    "features":
        feature_columns

}


# ============================================================
# SAVE MODEL METADATA
# ============================================================

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


# ============================================================
# COMPLETE
# ============================================================

print()
print(
    "=" * 65
)

print(
    "FRAUDSHIELD AI - MODEL METADATA GENERATED"
)

print(
    "=" * 65
)

print(
    f"Model       : {selected_model_name}"
)

print(
    f"Model Type  : {model_class_name}"
)

print(
    f"Transactions: {len(transaction_data):,}"
)

print(
    f"Features    : {len(feature_columns)}"
)

print(
    f"Accuracy    : "
    f"{selected_results['Accuracy']:.4f}"
)

print(
    f"Precision   : "
    f"{selected_results['Precision']:.4f}"
)

print(
    f"Recall      : "
    f"{selected_results['Recall']:.4f}"
)

print(
    f"F1 Score    : "
    f"{selected_results['F1 Score']:.4f}"
)

print(
    f"ROC-AUC     : "
    f"{selected_results['ROC AUC']:.4f}"
)

print()
print(
    f"Metadata Path: {METADATA_PATH}"
)

print(
    "=" * 65
)