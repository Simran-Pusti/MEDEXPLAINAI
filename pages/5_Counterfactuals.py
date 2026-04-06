import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend.counterfactual_engine.counterfactual_generator import CounterfactualGenerator
from backend.recommendation_engine.medical_recommender import MedicalRecommender
from utils.styles import load_css
load_css()


st.title("Counterfactual Reasoning")

if "model" not in st.session_state:

    st.warning("Train model first")

else:

    model = st.session_state["model"]
    X_train = st.session_state["X_train"]
    df = st.session_state["dataset"]
    target = st.session_state["target"]

    # Ensure numeric safety
    X_train = X_train.apply(pd.to_numeric, errors="coerce")

    # ------------------------------
    # MUTABLE FEATURES
    # ------------------------------

    st.subheader("Feature Mutability Settings")

    mutable_features = st.multiselect(
        "Select features that CAN change",
        X_train.columns,
        default=st.session_state.get("mutable", list(X_train.columns))
    )

    st.session_state["mutable"] = mutable_features

    immutable_features = [
        f for f in X_train.columns if f not in mutable_features
    ]

    st.write("Immutable features:", immutable_features)

    # ------------------------------
    # USER INPUT
    # ------------------------------

    st.subheader("Enter Patient Data")

    inputs = {}

    for feature in X_train.columns:

        default_val = float(X_train[feature].mean())

        inputs[feature] = st.number_input(
            feature,
            value=default_val
        )

    # ------------------------------
    # GENERATE COUNTERFACTUALS
    # ------------------------------

    if st.button("Generate Counterfactuals"):

        query = pd.DataFrame([inputs])
        query = query[X_train.columns]

        current_pred = model.predict(query)[0]
        current_prob = model.predict_proba(query)[0][1]

        st.session_state["current_pred"] = current_pred
        st.session_state["current_prob"] = current_prob
        st.session_state["query"] = query

        generator = CounterfactualGenerator()

        #  MODIFIED: now returns structured_output
        cf_df, structured_output = generator.generate(
            df,
            model,
            target,
            query,
            mutable_features
        )

        # ------------------------------
        # FILTER UNREALISTIC VALUES
        # ------------------------------
        clean_cf = []

        for i in range(len(cf_df)):

            valid = True

            for col in cf_df.columns:

                try:
                    val = float(cf_df.iloc[i][col])
                    min_val = df[col].min()
                    max_val = df[col].max()

                    if val < min_val or val > max_val:
                        valid = False
                        break
                except:
                    continue

            if valid:
                clean_cf.append(cf_df.iloc[i])

        if len(clean_cf) > 0:
            cf_df = pd.DataFrame(clean_cf)

        st.session_state["cf_df"] = cf_df
        st.session_state["structured_cf"] = structured_output  # ✅ NEW

    # ------------------------------
    # DISPLAY RESULTS
    # ------------------------------

    if "cf_df" in st.session_state:

        cf_df = st.session_state["cf_df"]
        query = st.session_state["query"]

        st.subheader("Current Prediction")

        st.write("Predicted Class:", st.session_state["current_pred"])
        st.write("Risk Probability:", round(st.session_state["current_prob"], 3))

        if not cf_df.empty:

            structured_output = st.session_state.get("structured_cf", [])

            # ------------------------------
            # SELECT BEST CF
            # ------------------------------
            best_cf = None
            best_prob = 1
            best_index = 0

            for i in range(len(cf_df)):

                cf_features = cf_df.iloc[[i]][X_train.columns]
                prob = model.predict_proba(cf_features)[0][1]

                if prob < best_prob:
                    best_prob = prob
                    best_cf = cf_df.iloc[[i]]
                    best_index = i

            # ------------------------------
            #  NEW: STRUCTURED OUTPUT DISPLAY
            # ------------------------------
            st.subheader("Optimal Counterfactual Explanation")

            if structured_output and best_index < len(structured_output):

                cf_item = structured_output[best_index]

                #  Counterfactual Sentence
                st.markdown(f"### 🔹 Counterfactual")
                st.success(cf_item["counterfactual"])

                #  Reason
                st.markdown("### 🔹 Reason")
                st.info(cf_item["reason"])

                #  Recommendations
                st.markdown("### 🔹 Recommendations")
                for rec in cf_item["recommendations"]:
                    st.write(f"- {rec}")

            # ------------------------------
            # RISK COMPARISON (UNCHANGED)
            # ------------------------------
            st.subheader("Risk Impact")

            current = st.session_state["current_prob"]
            new = best_prob

            st.write("New Risk Probability:", round(new, 3))
            st.write("Risk Reduction:", round(current - new, 3))

            fig, ax = plt.subplots()

            ax.bar(
                ["Current", "After Change"],
                [current, new]
            )

            ax.set_ylabel("Risk")

            st.pyplot(fig)

            # ------------------------------
            # EXISTING MEDICAL RECOMMENDER (KEPT)
            # ------------------------------
            st.subheader("Additional Medical Explanation")

            recommender = MedicalRecommender()

            recommendations = recommender.generate(
                query,
                best_cf
            )

            for r in recommendations:
                st.info(r)

            st.success(
                "This is the most effective minimal change to reduce disease risk."
            )

        else:

            st.warning("No valid counterfactual found. Try changing mutable features.")