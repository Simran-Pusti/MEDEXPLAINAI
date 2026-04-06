class ModelRecommender:

    def recommend(self, df):

        rows = df.shape[0]
        columns = df.shape[1]

        # -----------------------------
        # EXISTING LOGIC (UNCHANGED)
        # -----------------------------
        if rows < 1000:
            best_model = "Logistic Regression"

        elif rows < 5000:
            best_model = "Random Forest"

        else:
            best_model = "XGBoost"

        # -----------------------------
        #  ADDED: DUAL MODEL STRATEGY
        # -----------------------------
        # Prediction model = best model (your logic)
        prediction_model = best_model

        # Explanation model = always Logistic (for SHAP + Counterfactual stability)
        explanation_model = "Logistic Regression"

        # -----------------------------
        #  MODIFIED RETURN STRUCTURE
        # -----------------------------
        return {
            "best_model": best_model,                 # original recommendation
            "prediction_model": prediction_model,     # used for prediction
            "explanation_model": explanation_model    # used for SHAP + CF
        }