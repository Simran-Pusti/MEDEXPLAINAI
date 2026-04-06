import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SHAPEngine:

    def compute_shap(self, model, X, explanation_model=None):

        # ---------- FIX 1: Ensure numeric ----------
        if isinstance(X, pd.DataFrame):
            X = X.apply(pd.to_numeric, errors="coerce")

        # --------------------------------------------------
        #  FORCE EXPLANATION MODEL (LOGISTIC)
        # --------------------------------------------------
        if explanation_model is not None:
            model = explanation_model

        model_name = type(model).__name__

        # -----------------------------
        # TREE MODELS
        # -----------------------------
        if model_name in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "XGBClassifier"
        ]:

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            except:
                background = X.sample(min(100, len(X)), random_state=42)
                explainer = shap.KernelExplainer(
                    model.predict_proba,
                    background
                )
                shap_values = explainer.shap_values(X)

        # -----------------------------
        # LINEAR MODELS (FIXED)
        # -----------------------------
        elif model_name == "LogisticRegression":

            try:
                explainer = shap.LinearExplainer(
                    model,
                    X,
                    feature_perturbation="independent"
                )
            except:
                explainer = shap.LinearExplainer(model, X)

            shap_values = explainer.shap_values(X)

        # -----------------------------
        # OTHER MODELS
        # -----------------------------
        else:

            background = X.sample(min(100, len(X)), random_state=42)

            explainer = shap.KernelExplainer(
                model.predict_proba,
                background
            )

            shap_values = explainer.shap_values(X)

        # ---------- FIX 2: Binary classification ----------
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values


    # --------------------------------------------------
    #  NEW: HUMAN READABLE EXPLANATION
    # --------------------------------------------------
    def generate_human_explanation(self, shap_values, X):

        explanations = []

        # Mean impact of features
        mean_impact = np.mean(shap_values, axis=0)

        for feature, value in zip(X.columns, mean_impact):

            if abs(value) < 0.01:
                continue

            # Determine direction
            if value > 0:
                direction = "increases disease risk"
            else:
                direction = "reduces disease risk"

            # Strength classification
            strength_val = abs(value)

            if strength_val > 0.5:
                strength = "strongly"
            elif strength_val > 0.2:
                strength = "moderately"
            else:
                strength = "slightly"

            explanations.append(
                f"{feature} {strength} {direction}"
            )

        # Sort by importance
        explanations = explanations[:5]

        if not explanations:
            explanations.append(
                "No significant risk factors detected. Patient appears stable."
            )

        return explanations


    def global_summary_plot(self, shap_values, X):

        fig = plt.figure(figsize=(8, 5))

        shap.summary_plot(
            shap_values,
            X,
            show=False
        )

        return fig


    def feature_importance_plot(self, shap_values, X):

        fig = plt.figure(figsize=(8, 5))

        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False
        )

        return fig