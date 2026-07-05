import os
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

sys.path.insert(
    0,
    PROJECT_DIR
)


# ============================================================
# MODEL PATHS
# ============================================================

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


# ============================================================
# FRAUD PREDICTION ENGINE
# ============================================================

class FraudPredictionEngine:

    def __init__(self):

        self.model = None

        self.preprocessor = None

        self.model_loaded = False

        self.load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                "Fraud detection model not found. "
                "Run python src/train_model.py first."
            )


        if not os.path.exists(PREPROCESSOR_PATH):

            raise FileNotFoundError(
                "Fraud preprocessing model not found. "
                "Run python src/train_model.py first."
            )


        self.model = joblib.load(
            MODEL_PATH
        )


        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )


        self.model_loaded = True


    # ========================================================
    # VALIDATE TRANSACTION DATA
    # ========================================================

    def validate_transactions(
        self,
        data
    ):

        required_columns = [
            "Time"
        ]


        required_columns.extend(
            [
                f"V{i}"
                for i in range(1, 29)
            ]
        )


        required_columns.append(
            "Amount"
        )


        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]


        if missing_columns:

            raise ValueError(
                "Transaction data is missing required columns: "
                + ", ".join(missing_columns)
            )


        return True


    # ========================================================
    # RISK LEVEL
    # ========================================================

    @staticmethod
    def calculate_risk_level(
        probability
    ):

        risk_score = (
            probability * 100
        )


        if risk_score >= 80:

            return "Critical"


        if risk_score >= 60:

            return "High"


        if risk_score >= 30:

            return "Medium"


        return "Low"


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    @staticmethod
    def calculate_risk_message(
        risk_level
    ):

        risk_messages = {

            "Critical":
                "Immediate analyst investigation required.",

            "High":
                "Elevated fraud indicators detected.",

            "Medium":
                "Moderate transaction anomalies identified.",

            "Low":
                "No significant fraud indicators detected."

        }


        return risk_messages[
            risk_level
        ]


    # ========================================================
    # DETECT ANOMALOUS FEATURES
    # ========================================================

    @staticmethod
    def detect_anomalous_features(
        transaction
    ):

        pca_features = [
            f"V{i}"
            for i in range(1, 29)
        ]


        feature_values = transaction[
            pca_features
        ].abs()


        strongest_features = (
            feature_values
            .sort_values(
                ascending=False
            )
            .head(3)
        )


        anomaly_features = []


        for feature_name, value in strongest_features.items():

            if value >= 3:

                anomaly_features.append(
                    f"{feature_name} ({value:.2f})"
                )


        return anomaly_features


    # ========================================================
    # GENERATE FRAUD REASON
    # ========================================================

    def generate_fraud_reason(
        self,
        transaction,
        probability
    ):

        reasons = []


        risk_score = (
            probability * 100
        )


        anomaly_features = (
            self.detect_anomalous_features(
                transaction
            )
        )


        if risk_score >= 80:

            reasons.append(
                "Model probability exceeds the critical fraud threshold"
            )


        elif risk_score >= 60:

            reasons.append(
                "Model probability exceeds the elevated fraud threshold"
            )


        elif risk_score >= 30:

            reasons.append(
                "Moderate fraud probability detected by the model"
            )


        if anomaly_features:

            reasons.append(
                "Strong statistical deviations detected in "
                + ", ".join(anomaly_features)
            )


        amount = float(
            transaction["Amount"]
        )


        if amount >= 1000:

            reasons.append(
                "High-value transaction amount"
            )


        elif amount <= 1:

            reasons.append(
                "Near-zero transaction amount pattern"
            )


        if not reasons:

            reasons.append(
                "Transaction remains within low-risk model patterns"
            )


        return " | ".join(
            reasons
        )


    # ========================================================
    # GENERATE CASE ID
    # ========================================================

    @staticmethod
    def generate_case_ids(
        total_records
    ):

        return [
            f"FS-{index:07d}"
            for index in range(
                1,
                total_records + 1
            )
        ]


    # ========================================================
    # PREDICT TRANSACTIONS
    # ========================================================

    def predict_transactions(
        self,
        data
    ):

        if not self.model_loaded:

            raise RuntimeError(
                "Fraud detection model is not loaded."
            )


        original_data = data.copy()


        self.validate_transactions(
            original_data
        )


        feature_data = (
            original_data.copy()
        )


        if "Class" in feature_data.columns:

            feature_data = feature_data.drop(
                columns=[
                    "Class"
                ]
            )


        processed_data = (
            self.preprocessor.transform(
                feature_data
            )
        )


        fraud_probabilities = (
            self.model.predict_proba(
                processed_data
            )[:, 1]
        )


        predictions = (
            self.model.predict(
                processed_data
            )
        )


        results = (
            original_data.copy()
        )


        results.insert(
            0,
            "Case_ID",
            self.generate_case_ids(
                len(results)
            )
        )


        results[
            "Fraud_Probability"
        ] = fraud_probabilities


        results[
            "Risk_Score"
        ] = (
            fraud_probabilities * 100
        ).round(2)


        results[
            "Prediction"
        ] = np.where(
            predictions == 1,
            "Fraudulent",
            "Legitimate"
        )


        results[
            "Risk_Level"
        ] = [
            self.calculate_risk_level(
                probability
            )
            for probability in fraud_probabilities
        ]


        results[
            "Risk_Message"
        ] = [
            self.calculate_risk_message(
                risk_level
            )
            for risk_level in results[
                "Risk_Level"
            ]
        ]


        fraud_reasons = []


        for row_position, (
            _,
            transaction
        ) in enumerate(
            original_data.iterrows()
        ):

            fraud_reasons.append(
                self.generate_fraud_reason(
                    transaction,
                    fraud_probabilities[
                        row_position
                    ]
                )
            )


        results[
            "Fraud_Reason"
        ] = fraud_reasons


        return results


    # ========================================================
    # GENERATE SUMMARY
    # ========================================================

    def generate_summary(
        self,
        results
    ):

        total_transactions = len(
            results
        )


        fraud_transactions = (
            results["Prediction"]
            .eq("Fraudulent")
            .sum()
        )


        legitimate_transactions = (
            results["Prediction"]
            .eq("Legitimate")
            .sum()
        )


        critical_transactions = (
            results["Risk_Level"]
            .eq("Critical")
            .sum()
        )


        high_risk_transactions = (
            results["Risk_Level"]
            .eq("High")
            .sum()
        )


        medium_risk_transactions = (
            results["Risk_Level"]
            .eq("Medium")
            .sum()
        )


        low_risk_transactions = (
            results["Risk_Level"]
            .eq("Low")
            .sum()
        )


        fraud_percentage = (
            fraud_transactions
            / total_transactions
            * 100
        ) if total_transactions else 0


        average_risk_score = (
            results["Risk_Score"]
            .mean()
        )


        highest_risk_score = (
            results["Risk_Score"]
            .max()
        )


        return {

            "total_transactions":
                int(total_transactions),

            "fraud_transactions":
                int(fraud_transactions),

            "legitimate_transactions":
                int(legitimate_transactions),

            "fraud_percentage":
                round(
                    fraud_percentage,
                    2
                ),

            "critical_transactions":
                int(critical_transactions),

            "high_risk_transactions":
                int(high_risk_transactions),

            "medium_risk_transactions":
                int(medium_risk_transactions),

            "low_risk_transactions":
                int(low_risk_transactions),

            "average_risk_score":
                round(
                    float(
                        average_risk_score
                    ),
                    2
                ),

            "highest_risk_score":
                round(
                    float(
                        highest_risk_score
                    ),
                    2
                )

        }


    # ========================================================
    # GET SUSPICIOUS TRANSACTIONS
    # ========================================================

    def get_suspicious_transactions(
        self,
        results,
        minimum_risk=60
    ):

        return (
            results[
                results["Risk_Score"]
                >= minimum_risk
            ]
            .sort_values(
                by="Risk_Score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )