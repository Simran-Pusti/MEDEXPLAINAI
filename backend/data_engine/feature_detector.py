import pandas as pd

class FeatureDetector:

    def detect_features(self, df):

        numeric_features = []
        categorical_features = []
        binary_features = []

        for col in df.columns:

            if df[col].nunique() == 2:
                binary_features.append(col)

            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

            else:
                categorical_features.append(col)

        return {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "binary": binary_features
        }