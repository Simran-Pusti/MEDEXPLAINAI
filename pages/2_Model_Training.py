import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from state_manager import init_state

from backend.model_engine.model_recommender import ModelRecommender
from backend.model_engine.model_factory import ModelFactory
from backend.model_engine.trainer import ModelTrainer
from backend.model_engine.evaluator import ModelEvaluator
from utils.styles import load_css
load_css()

init_state()

st.title("Model Training & Evaluation")

# -----------------------------
# CHECK DATA
# -----------------------------
if st.session_state["dataset"] is None:
    st.warning("Upload dataset first")
    st.stop()

df = st.session_state["dataset"]
target = st.session_state["target"]

if target is None:
    st.warning("Select target in Technocrat Panel")
    st.stop()

df = df.apply(pd.to_numeric, errors="ignore")

# -----------------------------
# MODEL SELECTION
# -----------------------------
st.subheader("Model Selection")

recommender = ModelRecommender()

# MODIFIED: now returns dict
recommendation = recommender.recommend(df)

st.info(f"Recommended Model: {recommendation['best_model']}")

model_name = st.selectbox(
    "Choose Model",
    ["Logistic Regression","Random Forest","Decision Tree","SVM","XGBoost"]
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
if st.button("Train Model"):

    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    factory = ModelFactory()

    X_train, X_test, y_train, y_test = trainer.prepare_data(df, target)

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    # --------------------------------------------------
    #  NEW: CREATE DUAL MODELS
    # --------------------------------------------------
    # Override prediction model with user selection
    recommendation["prediction_model"] = model_name

    models = factory.create_dual_models(recommendation)

    # --------------------------------------------------
    #  NEW: TRAIN BOTH MODELS
    # --------------------------------------------------
    trained_models = trainer.train_dual_models(
        models,
        X_train,
        y_train
    )

    prediction_model = trained_models["prediction_model"]
    explanation_model = trained_models["explanation_model"]

    # --------------------------------------------------
    # EVALUATE USING PREDICTION MODEL
    # --------------------------------------------------
    metrics = evaluator.evaluate(prediction_model, X_test, y_test)

    # -----------------------------
    # STORE MODELS
    # -----------------------------
    st.session_state["model"] = prediction_model
    st.session_state["explanation_model"] = explanation_model
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test

    # -----------------------------
    # DISPLAY METRICS
    # -----------------------------
    st.subheader("Model Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", round(metrics["accuracy"], 3))
    col2.metric("Precision", round(metrics["precision"], 3))
    col3.metric("Recall", round(metrics["recall"], 3))
    col4.metric("F1 Score", round(metrics["f1_score"], 3))
    col5.metric("ROC-AUC", round(metrics["roc_auc"], 3))

    # -----------------------------
    # VISUAL CHART
    # -----------------------------
    st.subheader("Performance Visualization")

    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"]
    ]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics")

    st.pyplot(fig)

    # -----------------------------
    # INTERPRETATION
    # -----------------------------
    st.subheader("Performance Interpretation")

    if metrics["accuracy"] > 0.85:
        st.success("Excellent model performance with high reliability.")
    elif metrics["accuracy"] > 0.70:
        st.info("Good performance. Model is usable for decision support.")
    elif metrics["accuracy"] > 0.60:
        st.warning("Moderate performance. Improvements recommended.")
    else:
        st.error("Low performance. Model needs improvement.")

    # -----------------------------
    # DETAILED INSIGHT
    # -----------------------------
    st.subheader("Detailed Insights")

    if metrics["precision"] < metrics["recall"]:
        st.write("• Model detects most disease cases but may include false positives.")
    else:
        st.write("• Model is conservative and avoids false alarms.")

    if metrics["roc_auc"] > 0.8:
        st.write("• Model has strong ability to distinguish between classes.")
    else:
        st.write("• Model discrimination ability is moderate.")

    st.success("Model training completed successfully!")