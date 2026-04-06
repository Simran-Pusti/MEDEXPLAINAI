from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LimeEngine:

    def explain(self, model, X_train, instance, explanation_model=None):

        # --------------------------------------------------
        #  NEW: USE EXPLANATION MODEL IF PROVIDED
        # --------------------------------------------------
        if explanation_model is not None:
            model = explanation_model

        # --------------------------------------------------
        # FIX: Ensure correct format
        # --------------------------------------------------
        if isinstance(instance, pd.DataFrame):
            instance = instance.iloc[0].values
        elif isinstance(instance, dict):
            instance = np.array(list(instance.values()))
        else:
            instance = np.array(instance)

        # Ensure numeric
        X_train = X_train.apply(pd.to_numeric, errors="coerce")

        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["No Disease", "Disease"],
            discretize_continuous=True
        )

        explanation = explainer.explain_instance(
            instance,
            model.predict_proba
        )

        explanation_list = explanation.as_list()

        # -----------------------------
        # IMPROVED READABLE OUTPUT
        # -----------------------------
        readable_explanations = []

        for condition, weight in explanation_list:

            if weight > 0:
                impact = "increases disease risk"
            else:
                impact = "reduces disease risk"

            readable_explanations.append(
                f"{condition} → {impact} (strength: {round(abs(weight), 3)})"
            )

        return readable_explanations


    def plot(self, model, X_train, instance, explanation_model=None):

        # --------------------------------------------------
        #  NEW: USE EXPLANATION MODEL
        # --------------------------------------------------
        if explanation_model is not None:
            model = explanation_model

        # Fix instance format
        if isinstance(instance, pd.DataFrame):
            instance = instance.iloc[0].values
        elif isinstance(instance, dict):
            instance = np.array(list(instance.values()))
        else:
            instance = np.array(instance)

        X_train = X_train.apply(pd.to_numeric, errors="coerce")

        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["No Disease", "Disease"],
            discretize_continuous=True
        )

        exp = explainer.explain_instance(
            instance,
            model.predict_proba
        )

        fig = exp.as_pyplot_figure()

        return fig