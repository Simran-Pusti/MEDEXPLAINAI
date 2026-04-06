import streamlit as st
import pandas as pd
import numpy as np
from backend.explainability_engine.shap_engine import SHAPEngine
from backend.explainability_engine.lime_engine import LimeEngine
from utils.styles import load_css
load_css()

st.title("Explainable AI Analysis")

if "model" not in st.session_state:
    st.warning("Train model first")

else:

    model = st.session_state["model"]
    X_train = st.session_state["X_train"]

    #  NEW: GET EXPLANATION MODEL
    explanation_model = st.session_state.get("explanation_model", model)

    # Ensure numeric values
    X_train = X_train.apply(pd.to_numeric, errors="coerce")

    shap_engine = SHAPEngine()

    # -------------------------------
    # SHAP GLOBAL EXPLANATION
    # -------------------------------

    try:

        #  MODIFIED: pass explanation_model
        shap_values = shap_engine.compute_shap(
            model,
            X_train,
            explanation_model=explanation_model
        )

        st.subheader("Global Feature Importance")

        fig1 = shap_engine.global_summary_plot(shap_values, X_train)
        st.pyplot(fig1)

        st.markdown("""
        **Explanation**

        SHAP values measure how much each feature contributes
        to the prediction.

        Features at the top have the strongest influence on
        disease prediction.
        """)

        st.subheader("Feature Importance Ranking")

        fig2 = shap_engine.feature_importance_plot(shap_values, X_train)
        st.pyplot(fig2)

        # -------------------------------
        # AI Risk Factor Analysis
        # -------------------------------

        st.subheader("Top Risk Factors Identified by AI")

        importance = np.abs(shap_values).mean(axis=0)

        feature_importance = sorted(
            zip(X_train.columns, importance),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = feature_importance[:3]

        for f, score in top_features:
            st.write(f"• {f} (Impact Score: {round(score,3)})")

        st.subheader("AI Generated Medical Recommendations")

        medical_rules = {
            "glucose": "Reduce sugar intake and monitor blood glucose levels regularly.",
            "sugar": "Reduce sugar intake and monitor blood glucose levels regularly.",
            "bmi": "Maintain healthy body weight through balanced diet and regular exercise.",
            "weight": "Maintain healthy body weight through balanced diet and regular exercise.",
            "bloodpressure": "Reduce salt intake, manage stress, and monitor blood pressure regularly.",
            "bp": "Reduce salt intake, manage stress, and monitor blood pressure regularly.",
            "cholesterol": "Adopt a heart-healthy diet, avoid fatty foods, and exercise regularly.",
            "smoking": "Smoking cessation is strongly recommended to reduce cardiovascular and lung risks.",
            "smoke": "Smoking cessation is strongly recommended to reduce cardiovascular and lung risks.",
            "exercise": "Increase daily physical activity levels with cardio and strength training.",
            "activity": "Increase daily physical activity levels with cardio and strength training.",
            "insulin": "Regular monitoring of insulin levels and consultation with healthcare provider is recommended.",
            "age": "Regular health checkups and preventive care are recommended.",
            "fatigue": "Monitor energy levels, rest adequately, and consult a doctor if persistent.",
            "dizziness": "Stay hydrated, avoid sudden movements, and monitor blood pressure; seek medical advice if frequent.",
            "chest_pain": "Seek immediate medical attention if chest pain occurs; maintain heart health.",
            "cough": "Avoid pollutants, stay hydrated, and consult a doctor if cough persists.",
            "swelling": "Monitor fluid intake, reduce salt, and consult a doctor; may indicate kidney or heart issues.",
            "edema": "Monitor fluid intake, reduce salt, and consult a doctor; may indicate kidney or heart issues.",
            "shortness_of_breath": "Practice breathing exercises, avoid smoke/pollution, and consult a doctor if persistent.",
            "resp_rate": "Monitor breathing rate and consult a healthcare provider for abnormalities.",
            "oxygen_level": "Ensure adequate oxygenation; seek medical evaluation if consistently low.",
            "lung_capacity": "Perform respiratory exercises, avoid pollutants, and maintain good lung health.",
            "urea": "Monitor kidney function regularly; maintain hydration and consult a doctor for abnormal levels.",
            "creatinine": "Monitor kidney function regularly; maintain hydration and consult a doctor for abnormal levels.",
            "hemoglobin": "Maintain hemoglobin through iron-rich diet; consult a doctor if anemia is suspected.",
            "sodium": "Maintain electrolyte balance through diet and hydration; consult a doctor for abnormal levels.",
            "potassium": "Maintain electrolyte balance through diet and hydration; consult a doctor for abnormal levels.",
            "max_hr": "Monitor maximum heart rate during exercise and avoid overexertion.",
            "oldpeak": "Monitor heart strain during exercise and consult a healthcare provider if high.",
        }

        for f, _ in top_features:
            advice = medical_rules.get(
                f.lower(),
                "Lifestyle improvement is recommended for this factor."
            )
            st.write(f"• {f}: {advice}")

    except Exception as e:

        st.error("SHAP explanation could not be generated.")
        st.write(e)

    st.divider()

    # -------------------------------
    # LIME LOCAL EXPLANATION
    # -------------------------------

    lime_engine = LimeEngine()

    st.subheader("Local Explanation (LIME)")

    row_index = st.number_input(
        "Select Patient Row to Explain",
        min_value=0,
        max_value=len(X_train)-1,
        value=0
    )

    st.markdown("### Selected Patient Data")

    st.dataframe(
        X_train.iloc[[row_index]]
    )

    instance = X_train.iloc[row_index].values

    #  MODIFIED: pass explanation_model
    explanation = lime_engine.explain(
        model,
        X_train,
        instance,
        explanation_model=explanation_model
    )

    st.markdown("### Explanation")

    for item in explanation:
        st.write("•", item)

    # LIME visualization
    st.markdown("### LIME Impact Graph")

    fig_lime = lime_engine.plot(
        model,
        X_train,
        instance,
        explanation_model=explanation_model
    )

    st.pyplot(fig_lime)