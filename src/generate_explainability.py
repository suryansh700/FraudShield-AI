# ============================================================
# FRAUDSHIELD AI
# MODEL EXPLAINABILITY ARTIFACT GENERATOR
# ============================================================

import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(
        0,
        PROJECT_DIR
    )


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "creditcard.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "fraud_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "preprocessor.pkl"
)

REPORTS_DIR = os.path.join(
    PROJECT_DIR,
    "reports"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    REPORTS_DIR,
    "feature_importance.csv"
)


# ============================================================
# EXPLAINABILITY CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_EXPLAINABILITY_SAMPLES = 5000

PERMUTATION_REPEATS = 5


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file(
    file_path,
    file_description
):

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"{file_description} was not found.\n"
            f"Expected path: {file_path}"
        )


# ============================================================
# APPLICATION HEADER
# ============================================================

print("=" * 70)

print(
    "FRAUDSHIELD AI - EXPLAINABILITY GENERATOR"
)

print("=" * 70)


# ============================================================
# STEP 1 - VALIDATE ARTIFACTS
# ============================================================

print(
    "\n[1/8] Validating FraudShield artifacts..."
)


validate_file(
    DATA_PATH,
    "creditcard.csv"
)

validate_file(
    MODEL_PATH,
    "Fraud detection model"
)

validate_file(
    PREPROCESSOR_PATH,
    "Fraud preprocessing pipeline"
)


print(
    "All required artifacts were found."
)


# ============================================================
# STEP 2 - LOAD DATASET
# ============================================================

print(
    "\n[2/8] Loading financial transaction dataset..."
)


data = pd.read_csv(
    DATA_PATH
)


print(
    f"Transactions : {len(data):,}"
)

print(
    f"Columns      : {len(data.columns):,}"
)


if "Class" not in data.columns:

    raise ValueError(
        "Dataset does not contain the Class target column."
    )


# ============================================================
# STEP 3 - PREPARE FEATURES
# ============================================================

print(
    "\n[3/8] Preparing explainability dataset..."
)


X = data.drop(
    columns=["Class"]
)

y = data["Class"]


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    f"Training Transactions : {len(X_train):,}"
)

print(
    f"Testing Transactions  : {len(X_test):,}"
)


# ============================================================
# STEP 4 - LOAD TRAINED MODEL
# ============================================================

print(
    "\n[4/8] Loading trained fraud detection model..."
)


model = joblib.load(
    MODEL_PATH
)


print(
    f"Detected Model : {model.__class__.__name__}"
)


# ============================================================
# STEP 5 - LOAD PREPROCESSOR
# ============================================================

print(
    "\n[5/8] Loading transaction preprocessing pipeline..."
)


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


print(
    "Detected Preprocessor : "
    f"{preprocessor.__class__.__name__}"
)


# ============================================================
# STEP 6 - TRANSFORM EVALUATION DATA
# ============================================================

print(
    "\n[6/8] Transforming evaluation transactions..."
)


X_test_processed = preprocessor.transform(
    X_test
)


if hasattr(
    X_test_processed,
    "toarray"
):

    X_test_processed = (
        X_test_processed.toarray()
    )


X_test_processed = np.asarray(
    X_test_processed
)


print(
    "Original Feature Shape  : "
    f"{X_test.shape}"
)

print(
    "Processed Feature Shape : "
    f"{X_test_processed.shape}"
)


# ============================================================
# FEATURE NAME RESOLUTION
# ============================================================

print(
    "\nResolving processed feature names..."
)


try:

    processed_feature_names = list(
        preprocessor.get_feature_names_out()
    )


    processed_feature_names = [
        str(feature_name)
        .replace(
            "remainder__",
            ""
        )
        .replace(
            "transform__",
            ""
        )
        .replace(
            "scale__",
            ""
        )
        .replace(
            "num__",
            ""
        )
        for feature_name
        in processed_feature_names
    ]


except Exception:

    processed_feature_names = list(
        X.columns
    )


if (
    len(processed_feature_names)
    != X_test_processed.shape[1]
):

    print(
        "Feature-name count mismatch detected."
    )

    print(
        "Using generated feature identifiers."
    )


    processed_feature_names = [
        f"Feature_{index + 1}"
        for index in range(
            X_test_processed.shape[1]
        )
    ]


print(
    "Resolved Features : "
    f"{len(processed_feature_names)}"
)


# ============================================================
# STEP 7 - CREATE EXPLAINABILITY SAMPLE
# ============================================================

print(
    "\n[7/8] Creating explainability evaluation sample..."
)


sample_size = min(
    MAX_EXPLAINABILITY_SAMPLES,
    len(X_test_processed)
)


random_generator = np.random.default_rng(
    RANDOM_STATE
)


sample_indices = random_generator.choice(
    len(X_test_processed),
    size=sample_size,
    replace=False
)


X_explain = X_test_processed[
    sample_indices
]


y_explain = (
    y_test
    .iloc[sample_indices]
    .to_numpy()
)


fraud_count = int(
    np.sum(
        y_explain == 1
    )
)


legitimate_count = int(
    np.sum(
        y_explain == 0
    )
)


print(
    f"Explainability Sample : {sample_size:,}"
)

print(
    f"Legitimate Samples    : {legitimate_count:,}"
)

print(
    f"Fraud Samples         : {fraud_count:,}"
)


# ============================================================
# SAFETY CHECK FOR FRAUD CLASS
# ============================================================

if fraud_count == 0:

    print(
        "\nRandom sample did not contain fraud transactions."
    )

    print(
        "Creating stratified explainability sample..."
    )


    test_indices = np.arange(
        len(y_test)
    )


    _, explain_indices = train_test_split(
        test_indices,
        test_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=y_test
    )


    X_explain = X_test_processed[
        explain_indices
    ]


    y_explain = (
        y_test
        .iloc[explain_indices]
        .to_numpy()
    )


    fraud_count = int(
        np.sum(
            y_explain == 1
        )
    )


    legitimate_count = int(
        np.sum(
            y_explain == 0
        )
    )


    print(
        f"Legitimate Samples : {legitimate_count:,}"
    )

    print(
        f"Fraud Samples      : {fraud_count:,}"
    )


# ============================================================
# STEP 8 - CALCULATE PERMUTATION IMPORTANCE
# ============================================================

print(
    "\n[8/8] Calculating fraud feature importance..."
)

print(
    "Scoring Metric : Average Precision"
)

print(
    f"Permutation Repeats : {PERMUTATION_REPEATS}"
)

print(
    "This operation may take a few minutes..."
)


importance_result = permutation_importance(
    model,
    X_explain,
    y_explain,
    scoring="average_precision",
    n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ============================================================
# CREATE FEATURE IMPORTANCE REPORT
# ============================================================

feature_importance = pd.DataFrame(
    {
        "Feature": processed_feature_names,

        "Importance":
            importance_result.importances_mean,

        "Importance_Std":
            importance_result.importances_std
    }
)


feature_importance[
    "Absolute_Importance"
] = feature_importance[
    "Importance"
].abs()


feature_importance = (
    feature_importance
    .sort_values(
        by="Absolute_Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


feature_importance[
    "Rank"
] = np.arange(
    1,
    len(feature_importance) + 1
)


maximum_importance = feature_importance[
    "Absolute_Importance"
].max()


if maximum_importance > 0:

    feature_importance[
        "Relative_Importance"
    ] = (
        feature_importance[
            "Absolute_Importance"
        ]
        /
        maximum_importance
        *
        100
    )

else:

    feature_importance[
        "Relative_Importance"
    ] = 0.0


feature_importance[
    "Influence"
] = np.where(
    feature_importance[
        "Importance"
    ] >= 0,
    "Positive Detection Contribution",
    "Negative Detection Contribution"
)


feature_importance = feature_importance[
    [
        "Rank",
        "Feature",
        "Importance",
        "Importance_Std",
        "Absolute_Importance",
        "Relative_Importance",
        "Influence"
    ]
]


# ============================================================
# SAVE EXPLAINABILITY REPORT
# ============================================================

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


feature_importance.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "EXPLAINABILITY ARTIFACT GENERATED"
)

print("=" * 70)


print(
    f"\nReport Path : {FEATURE_IMPORTANCE_PATH}"
)


print(
    "\nTOP 10 FRAUD DETECTION FEATURES"
)


top_features = feature_importance[
    [
        "Rank",
        "Feature",
        "Importance",
        "Relative_Importance"
    ]
].head(10)


print(
    top_features.to_string(
        index=False,
        formatters={
            "Importance":
                "{:.6f}".format,

            "Relative_Importance":
                "{:.2f}%".format
        }
    )
)


print()

print("=" * 70)

print(
    "FraudShield AI explainability intelligence is ready."
)

print("=" * 70)