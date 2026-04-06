from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:

    def prepare_data(self, df, target):

        X = df.drop(columns=[target])
        y = df[target]

        for col in X.columns:

            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        return X_train, X_test, y_train, y_test


    def train(self, model, X_train, y_train):

        model.fit(X_train, y_train)

        return model

    # --------------------------------------------------
    #  ADDED: DUAL MODEL TRAINING (NEW FUNCTION)
    # --------------------------------------------------
    def train_dual_models(self, models_dict, X_train, y_train):

        """
        Train both:
        1. Prediction model
        2. Explanation model
        """

        prediction_model = models_dict["prediction_model"]
        explanation_model = models_dict["explanation_model"]

        # -----------------------------
        # TRAIN BOTH MODELS
        # -----------------------------
        prediction_model.fit(X_train, y_train)
        explanation_model.fit(X_train, y_train)

        # -----------------------------
        # RETURN TRAINED MODELS
        # -----------------------------
        return {
            "prediction_model": prediction_model,
            "explanation_model": explanation_model
        }