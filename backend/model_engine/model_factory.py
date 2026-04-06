from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


class ModelFactory:

    def create(self, model_name):

        models = {

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "Random Forest": RandomForestClassifier(
                n_estimators=200
            ),

            "Decision Tree": DecisionTreeClassifier(),

            "SVM": SVC(probability=True),

            "XGBoost": XGBClassifier(
                eval_metric="logloss"
            )
        }

        return models[model_name]

    # --------------------------------------------------
    #  ADDED: DUAL MODEL CREATION (NEW FUNCTION)
    # --------------------------------------------------
    def create_dual_models(self, recommendation_dict):

        """
        Creates two models:
        1. Prediction model (best model from recommender)
        2. Explanation model (always Logistic Regression)
        """

        # -----------------------------
        # GET MODEL NAMES
        # -----------------------------
        prediction_model_name = recommendation_dict["prediction_model"]
        explanation_model_name = recommendation_dict["explanation_model"]

        # -----------------------------
        # CREATE MODELS (USING EXISTING METHOD)
        # -----------------------------
        prediction_model = self.create(prediction_model_name)

        #  FORCE Logistic for explanation (stable SHAP + CF)
        explanation_model = self.create(explanation_model_name)

        # -----------------------------
        # RETURN BOTH MODELS
        # -----------------------------
        return {
            "prediction_model": prediction_model,
            "explanation_model": explanation_model
        }