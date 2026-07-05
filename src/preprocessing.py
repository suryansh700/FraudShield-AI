import numpy as np

from sklearn.preprocessing import RobustScaler


class FraudDataPreprocessor:

    def __init__(self):

        self.amount_scaler = RobustScaler()

        self.time_scaler = RobustScaler()

        self.is_fitted = False


    def validate_dataset(self, data):
        """
        Validate transaction dataset.
        """

        required_columns = [
            "Time",
            "Amount"
        ]

        missing_columns = [

            column

            for column in required_columns

            if column not in data.columns

        ]

        if missing_columns:

            raise ValueError(

                f"Missing required columns: {missing_columns}"

            )

        return True


    def clean_data(self, data):
        """
        Clean transaction features.

        Duplicate removal is NOT performed here because
        removing rows from X without removing matching
        rows from y causes sample mismatch.
        """

        data = data.copy()

        data.replace(

            [np.inf, -np.inf],

            np.nan,

            inplace=True

        )


        numeric_columns = data.select_dtypes(

            include=np.number

        ).columns


        for column in numeric_columns:

            if data[column].isnull().sum() > 0:

                median_value = data[column].median()

                data[column] = data[column].fillna(

                    median_value

                )


        return data


    def fit_transform(self, data):
        """
        Fit scalers using training data.
        """

        data = data.copy()


        self.validate_dataset(data)


        data = self.clean_data(data)


        data["scaled_amount"] = (

            self.amount_scaler.fit_transform(

                data[["Amount"]]

            ).ravel()

        )


        data["scaled_time"] = (

            self.time_scaler.fit_transform(

                data[["Time"]]

            ).ravel()

        )


        data.drop(

            columns=[

                "Amount",

                "Time"

            ],

            inplace=True

        )


        self.is_fitted = True


        return data


    def transform(self, data):
        """
        Transform new transaction data.
        """

        if not self.is_fitted:

            raise ValueError(

                "Preprocessor is not fitted."

            )


        data = data.copy()


        self.validate_dataset(data)


        data = self.clean_data(data)


        data["scaled_amount"] = (

            self.amount_scaler.transform(

                data[["Amount"]]

            ).ravel()

        )


        data["scaled_time"] = (

            self.time_scaler.transform(

                data[["Time"]]

            ).ravel()

        )


        data.drop(

            columns=[

                "Amount",

                "Time"

            ],

            inplace=True

        )


        return data