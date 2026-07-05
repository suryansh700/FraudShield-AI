# ============================================================
# FRAUDSHIELD AI
# TRANSACTION-LEVEL FRAUD EXPLANATION ENGINE
# ============================================================

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd


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

METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "model_metadata.json"
)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "creditcard.csv"
)

REPORTS_DIR = os.path.join(
    PROJECT_DIR,
    "reports"
)

CASE_EXPLANATION_PATH = os.path.join(
    REPORTS_DIR,
    "sample_case_explanation.csv"
)


# ============================================================
# CREATE REPORT DIRECTORY
# ============================================================

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ============================================================
# FRAUD CASE EXPLAINER
# ============================================================

class FraudCaseExplainer:

    def __init__(self):

        self.model = self._load_artifact(
            MODEL_PATH,
            "Fraud detection model"
        )

        self.preprocessor = self._load_artifact(
            PREPROCESSOR_PATH,
            "Fraud preprocessing pipeline"
        )

        self.metadata = self._load_metadata()

        self.reference_data = (
            self._load_reference_data()
        )

        self.feature_columns = (
            self._resolve_feature_columns()
        )

        self.baseline_values = (
            self.reference_data[
                self.feature_columns
            ]
            .median()
        )


    # ========================================================
    # LOAD JOBLIB ARTIFACT
    # ========================================================

    @staticmethod
    def _load_artifact(
        artifact_path,
        artifact_name
    ):

        if not os.path.exists(
            artifact_path
        ):

            raise FileNotFoundError(
                f"{artifact_name} was not found.\n"
                f"Expected path: {artifact_path}"
            )


        try:

            artifact = joblib.load(
                artifact_path
            )

            return artifact

        except Exception as error:

            raise RuntimeError(
                f"Unable to load {artifact_name}.\n"
                f"Artifact path: {artifact_path}\n"
                f"Original error: {error}\n\n"
                "Run 'python src/train_model.py' "
                "to regenerate compatible artifacts."
            ) from error


    # ========================================================
    # LOAD MODEL METADATA
    # ========================================================

    @staticmethod
    def _load_metadata():

        if not os.path.exists(
            METADATA_PATH
        ):

            return {}


        try:

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8"
            ) as metadata_file:

                return json.load(
                    metadata_file
                )

        except Exception:

            return {}


    # ========================================================
    # LOAD REFERENCE DATASET
    # ========================================================

    @staticmethod
    def _load_reference_data():

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
                "Dataset must contain a "
                "'Class' target column."
            )


        return data


    # ========================================================
    # RESOLVE FEATURE COLUMNS
    # ========================================================

    def _resolve_feature_columns(
        self
    ):

        metadata_features = (
            self.metadata.get(
                "feature_columns"
            )
        )


        if metadata_features:

            missing_columns = [

                column

                for column in metadata_features

                if column
                not in self.reference_data.columns

            ]


            if missing_columns:

                raise ValueError(
                    "Metadata contains features "
                    "missing from the dataset: "
                    + ", ".join(
                        missing_columns
                    )
                )


            return metadata_features


        return [

            column

            for column
            in self.reference_data.columns

            if column != "Class"

        ]


    # ========================================================
    # PREPARE TRANSACTION
    # ========================================================

    def _prepare_transaction(
        self,
        transaction
    ):

        if isinstance(
            transaction,
            pd.Series
        ):

            transaction = (
                transaction
                .to_frame()
                .T
            )


        elif isinstance(
            transaction,
            dict
        ):

            transaction = pd.DataFrame(
                [transaction]
            )


        elif not isinstance(
            transaction,
            pd.DataFrame
        ):

            raise TypeError(
                "Transaction must be a pandas "
                "Series, DataFrame or dictionary."
            )


        transaction = transaction.copy()


        missing_features = [

            feature

            for feature in self.feature_columns

            if feature
            not in transaction.columns

        ]


        if missing_features:

            raise ValueError(
                "Transaction is missing "
                "required features: "
                + ", ".join(
                    missing_features
                )
            )


        prepared_transaction = transaction[
            self.feature_columns
        ].copy()


        prepared_transaction = (
            prepared_transaction
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )


        if prepared_transaction.isnull().any().any():

            invalid_columns = (

                prepared_transaction
                .columns[
                    prepared_transaction
                    .isnull()
                    .any()
                ]
                .tolist()

            )


            raise ValueError(
                "Transaction contains invalid "
                "numeric values in: "
                + ", ".join(
                    invalid_columns
                )
            )


        return prepared_transaction


    # ========================================================
    # TRANSFORM TRANSACTION
    # ========================================================

    def _transform_transaction(
        self,
        transaction
    ):

        prepared_transaction = (
            self._prepare_transaction(
                transaction
            )
        )


        transformed_transaction = (
            self.preprocessor.transform(
                prepared_transaction
            )
        )


        return transformed_transaction


    # ========================================================
    # PREDICT FRAUD PROBABILITY
    # ========================================================

    def predict_fraud_probability(
        self,
        transaction
    ):

        transformed_transaction = (
            self._transform_transaction(
                transaction
            )
        )


        if not hasattr(
            self.model,
            "predict_proba"
        ):

            raise AttributeError(
                "The trained fraud model does not "
                "support predict_proba()."
            )


        probabilities = (
            self.model.predict_proba(
                transformed_transaction
            )
        )


        fraud_probability = float(
            probabilities[0][1]
        )


        return fraud_probability


    # ========================================================
    # GET RISK LEVEL
    # ========================================================

    @staticmethod
    def get_risk_level(
        fraud_probability
    ):

        if fraud_probability >= 0.90:

            return "Critical"

        if fraud_probability >= 0.70:

            return "High"

        if fraud_probability >= 0.40:

            return "Medium"

        return "Low"


    # ========================================================
    # EXPLAIN TRANSACTION
    # ========================================================

    def explain_transaction(
        self,
        transaction,
        top_n=10
    ):

        transaction = (
            self._prepare_transaction(
                transaction
            )
        )


        original_probability = (
            self.predict_fraud_probability(
                transaction
            )
        )


        feature_explanations = []


        for feature in self.feature_columns:

            perturbed_transaction = (
                transaction.copy()
            )


            original_value = float(
                perturbed_transaction[
                    feature
                ].iloc[0]
            )


            baseline_value = float(
                self.baseline_values[
                    feature
                ]
            )


            perturbed_transaction.loc[
                perturbed_transaction.index[0],
                feature
            ] = baseline_value


            perturbed_probability = (
                self.predict_fraud_probability(
                    perturbed_transaction
                )
            )


            probability_change = (
                original_probability
                -
                perturbed_probability
            )


            contribution_points = (
                probability_change * 100
            )


            if contribution_points > 0:

                influence = (
                    "Risk Increasing"
                )

            elif contribution_points < 0:

                influence = (
                    "Risk Reducing"
                )

            else:

                influence = "Neutral"


            feature_explanations.append(
                {

                    "Feature":
                        feature,

                    "Original_Value":
                        original_value,

                    "Baseline_Value":
                        baseline_value,

                    "Original_Probability":
                        original_probability * 100,

                    "Perturbed_Probability":
                        perturbed_probability * 100,

                    "Risk_Contribution":
                        contribution_points,

                    "Absolute_Contribution":
                        abs(
                            contribution_points
                        ),

                    "Influence":
                        influence
                }
            )


        explanation_data = pd.DataFrame(
            feature_explanations
        )


        explanation_data = (
            explanation_data
            .sort_values(
                by="Absolute_Contribution",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        explanation_data.insert(
            0,
            "Rank",
            range(
                1,
                len(explanation_data) + 1
            )
        )


        top_n = min(
            int(top_n),
            len(explanation_data)
        )


        return explanation_data.head(
            top_n
        )


    # ========================================================
    # GENERATE CASE SUMMARY
    # ========================================================

    def generate_case_summary(
        self,
        transaction,
        top_n=5
    ):

        fraud_probability = (
            self.predict_fraud_probability(
                transaction
            )
        )


        explanation = (
            self.explain_transaction(
                transaction,
                top_n=len(
                    self.feature_columns
                )
            )
        )


        risk_increasing = explanation[
            explanation[
                "Risk_Contribution"
            ] > 0
        ].copy()


        risk_reducing = explanation[
            explanation[
                "Risk_Contribution"
            ] < 0
        ].copy()


        risk_level = self.get_risk_level(
            fraud_probability
        )


        if not risk_increasing.empty:

            strongest_feature = str(
                risk_increasing.iloc[0][
                    "Feature"
                ]
            )


            strongest_contribution = float(
                risk_increasing.iloc[0][
                    "Risk_Contribution"
                ]
            )

        else:

            strongest_feature = (
                "No dominant risk feature"
            )

            strongest_contribution = 0.0


        top_risk_features = (

            risk_increasing[
                "Feature"
            ]

            .head(
                top_n
            )

            .tolist()

        )


        return {

            "fraud_probability": round(
                fraud_probability * 100,
                4
            ),

            "risk_level":
                risk_level,

            "model_decision": (
                "Fraudulent"
                if fraud_probability >= 0.50
                else "Legitimate"
            ),

            "strongest_risk_feature":
                strongest_feature,

            "strongest_contribution": round(
                strongest_contribution,
                4
            ),

            "top_risk_features":
                top_risk_features,

            "risk_increasing_features": int(
                len(
                    risk_increasing
                )
            ),

            "risk_reducing_features": int(
                len(
                    risk_reducing
                )
            )
        }


    # ========================================================
    # GENERATE RISK INTERPRETATION
    # ========================================================

    def generate_risk_interpretation(
        self,
        transaction
    ):

        summary = (
            self.generate_case_summary(
                transaction,
                top_n=3
            )
        )


        probability = summary[
            "fraud_probability"
        ]

        risk_level = summary[
            "risk_level"
        ]

        top_features = summary[
            "top_risk_features"
        ]


        if top_features:

            feature_text = ", ".join(
                top_features
            )

        else:

            feature_text = (
                "no dominant individual features"
            )


        if risk_level == "Critical":

            interpretation = (

                "The transaction demonstrates a "
                "critical fraud-risk pattern. "

                f"The active model assigned a "
                f"{probability:.2f}% fraud probability. "

                f"The strongest transaction-level "
                f"risk signals were observed across "
                f"{feature_text}. "

                "The combined feature pattern is "
                "highly suspicious and should receive "
                "immediate investigation priority."

            )


        elif risk_level == "High":

            interpretation = (

                "The transaction demonstrates a "
                "high fraud-risk pattern. "

                f"The model assigned a "
                f"{probability:.2f}% fraud probability. "

                f"Important risk signals were detected "
                f"across {feature_text}. "

                "The transaction should be reviewed "
                "by a fraud analyst."

            )


        elif risk_level == "Medium":

            interpretation = (

                "The transaction contains moderate "
                "fraud-risk signals. "

                f"The model assigned a "
                f"{probability:.2f}% fraud probability. "

                f"Notable model influences include "
                f"{feature_text}. "

                "Additional transaction context may "
                "be required before escalation."

            )


        else:

            interpretation = (

                "The transaction currently demonstrates "
                "a low fraud-risk profile. "

                f"The model assigned a "
                f"{probability:.2f}% fraud probability. "

                "The observed transaction pattern does "
                "not currently indicate a strong fraud "
                "signal."

            )


        return interpretation


# ============================================================
# LOCAL CASE EXPLANATION TEST
# ============================================================

def main():

    print()
    print("=" * 72)
    print("        FRAUDSHIELD AI - CASE EXPLANATION ENGINE")
    print("=" * 72)
    print()


    print(
        "[1/5] Loading FraudShield AI artifacts..."
    )


    explainer = FraudCaseExplainer()


    print(
        "Model and preprocessor loaded successfully."
    )

    print()


    print(
        "[2/5] Loading transaction dataset..."
    )


    data = pd.read_csv(
        DATA_PATH
    )


    print(
        f"Transactions Available : {len(data):,}"
    )

    print()


    print(
        "[3/5] Selecting suspicious transaction..."
    )


    fraud_transactions = data[
        data["Class"] == 1
    ]


    if fraud_transactions.empty:

        raise ValueError(
            "No fraudulent transactions "
            "were found in the dataset."
        )


    sample_transaction = (
        fraud_transactions
        .iloc[0]
        .copy()
    )


    transaction_index = (
        fraud_transactions
        .index[0]
    )


    case_id = (
        f"CASE-{transaction_index:06d}"
    )


    print(
        f"Selected Case : {case_id}"
    )

    print()


    print(
        "[4/5] Generating transaction explanation..."
    )


    summary = (
        explainer.generate_case_summary(
            sample_transaction,
            top_n=5
        )
    )


    explanation = (
        explainer.explain_transaction(
            sample_transaction,
            top_n=10
        )
    )


    interpretation = (
        explainer.generate_risk_interpretation(
            sample_transaction
        )
    )


    explanation.insert(
        0,
        "Case_ID",
        case_id
    )


    explanation.to_csv(
        CASE_EXPLANATION_PATH,
        index=False
    )


    print(
        "Transaction explanation generated."
    )

    print()


    print(
        "[5/5] Fraud case intelligence report..."
    )

    print()

    print(
        "=" * 72
    )

    print(
        "CASE INTELLIGENCE SUMMARY"
    )

    print(
        "=" * 72
    )

    print()


    print(
        f"{'Case ID':<30}: "
        f"{case_id}"
    )

    print(
        f"{'Fraud Probability':<30}: "
        f"{summary['fraud_probability']:.4f}%"
    )

    print(
        f"{'Risk Level':<30}: "
        f"{summary['risk_level']}"
    )

    print(
        f"{'Model Decision':<30}: "
        f"{summary['model_decision']}"
    )

    print(
        f"{'Strongest Risk Feature':<30}: "
        f"{summary['strongest_risk_feature']}"
    )

    print(
        f"{'Strongest Contribution':<30}: "
        f"{summary['strongest_contribution']:.4f} pp"
    )

    print(
        f"{'Risk Increasing Features':<30}: "
        f"{summary['risk_increasing_features']}"
    )

    print(
        f"{'Risk Reducing Features':<30}: "
        f"{summary['risk_reducing_features']}"
    )

    print()


    print(
        "TOP TRANSACTION-LEVEL MODEL INFLUENCES"
    )

    print(
        "-" * 72
    )

    print()


    display_columns = [

        "Rank",

        "Feature",

        "Original_Value",

        "Baseline_Value",

        "Risk_Contribution",

        "Influence"

    ]


    print(
        explanation[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()


    print(
        "FRAUD RISK INTERPRETATION"
    )

    print(
        "-" * 72
    )

    print()

    print(
        interpretation
    )

    print()


    print(
        f"Case Explanation Report : "
        f"{CASE_EXPLANATION_PATH}"
    )

    print()

    print("=" * 72)

    print(
        "CASE EXPLANATION COMPLETE"
    )

    print("=" * 72)


# ============================================================
# RUN LOCAL TEST
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("=" * 72)
        print("CASE EXPLANATION FAILED")
        print("=" * 72)
        print()

        print(
            f"Error Type : "
            f"{type(error).__name__}"
        )

        print(
            f"Error      : {error}"
        )

        print()

        raise