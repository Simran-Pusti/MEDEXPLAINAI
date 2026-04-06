import streamlit as st
import pandas as pd
from backend.report_engine.report_builder import ReportBuilder
from utils.styles import load_css
load_css()

st.title("Decision Support Panel")

if "model" not in st.session_state:

    st.warning("Train model first")

else:

    model = st.session_state["model"]
    X_train = st.session_state["X_train"]

    # Ensure numeric features
    X_train = X_train.apply(pd.to_numeric, errors="coerce")

    st.subheader("Enter Input Details")

    inputs = {}

    for feature in X_train.columns:
        value = st.number_input(feature)
        inputs[feature] = value

    #  FIX 1: Persist Predict (ONLY CHANGE)
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    if st.button("Predict"):
        st.session_state.predict_clicked = True

    #  SAME BLOCK (UNCHANGED LOGIC)
    if st.session_state.predict_clicked:

        df = pd.DataFrame([inputs])
        df = df[X_train.columns]

        prediction = model.predict(df)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df)[0].max()

        st.subheader("Prediction Result")

        st.write("Predicted Class:", prediction)

        if probability is not None:
            st.write("Prediction Confidence:", round(probability, 3))

        # -------------------------
        # Interpretation
        # -------------------------

        st.subheader("Interpretation")

        if probability is not None:

            if probability > 0.75:
                st.error("Model confidence is HIGH for this prediction.")

            elif probability > 0.40:
                st.warning("Model shows MODERATE confidence.")

            else:
                st.success("Model confidence is LOW.")

        else:
            st.info("Model does not provide probability scores.")

        # -------------------------
        # Explanation
        # -------------------------

        st.subheader("Explanation")

        st.write(
            "This prediction is generated using the trained machine learning model "
            "based on the provided feature values."
        )

        # -------------------------
        # General Recommendations
        # -------------------------

        st.subheader("General Recommendations")

        st.write("• Maintain healthy lifestyle habits")
        st.write("• Monitor important indicators regularly")
        st.write("• Follow expert guidance if risks are detected")

        # -------------------------
        # Personalized Recommendations (UNCHANGED FULL)
        # -------------------------

        st.subheader("Personalized Recommendations")

        recommendations = []

        for feature, value in inputs.items():

            f = feature.lower()

            if "glucose" in f or "sugar" in f or "gluc" in f:
                if value > 140:
                    recommendations.append("High glucose detected. Reduce sugar intake and monitor blood glucose.Take fiber-rich foods, avoid sugary drinks, and maintain a balanced diet. Regular physical activity can help improve insulin sensitivity.")

            if "bmi" in f or "weight" in f:
                if value > 30:
                    recommendations.append("BMI indicates obesity risk. Weight reduction recommended.Maintain a healthy weight through a balanced diet and regular exercise. Focus on portion control, include more fruits and vegetables, and limit processed foods.")

            if "bloodpressure" in f or "bp" in f:
                if value > 130:
                    recommendations.append("Elevated blood pressure detected. Reduce salt and stress.")

            if "cholesterol" in f and value > 200:
                recommendations.append("High cholesterol detected. Avoid fatty foods.")

            if "smoke" in f or "smoking" in f or "smok" in f:
                if value > 0:
                    recommendations.append("Smoking detected. Quit smoking immediately.")

            if "exercise" in f or "activity" in f:
                if value < 2:
                    recommendations.append("Low activity detected. Increase exercise.")

            if "insulin" in f and value > 25:
                recommendations.append("High insulin levels. Consult a doctor for insulin management.")

            if "urea" in f and value > 50:
                recommendations.append("Elevated urea levels. Monitor kidney function and stay hydrated.")

            if "creatinine" in f and value > 1.3:
                recommendations.append("High creatinine detected. Check kidney health.")

            if "hemoglobin" in f and value < 12:
                recommendations.append("Low hemoglobin. Consume iron-rich foods and consult a doctor.")

            if "sodium" in f and (value < 135 or value > 145):
                recommendations.append("Abnormal sodium level. Maintain electrolyte balance.")

            if "potassium" in f and (value < 3.5 or value > 5.0):
                recommendations.append("Abnormal potassium level. Maintain electrolyte balance.")

            if "oxygen_level" in f and value < 95:
                recommendations.append("Low oxygen saturation. Seek medical advice if persistent.")

            if "resp_rate" in f and (value < 12 or value > 20):
                recommendations.append("Abnormal respiration rate. Monitor breathing and consult a doctor.")

            if "heart_rate" in f or "pulse" in f:
                if value < 60 or value > 100:
                    recommendations.append("Abnormal heart rate detected. Monitor cardiovascular health.")

            if "fatigue" in f and value > 2:
                recommendations.append("High fatigue levels. Rest adequately and monitor health.")

            if "chest_pain" in f and value > 0:
                recommendations.append("Chest pain reported. Seek immediate medical attention.")

            if "dizziness" in f and value > 0:
                recommendations.append("Dizziness detected. Monitor blood pressure and consult a doctor if frequent.")

            if "shortness_of_breath" in f or "breathlessness" in f:
                if value > 0:
                    recommendations.append("Shortness of breath reported. Evaluate lung and heart health.Navigate to a well-ventilated area, avoid exertion, and seek medical evaluation if symptoms persist.")

            if "swelling" in f or "edema" in f:
                if value > 0:
                    recommendations.append("Swelling detected. Monitor fluid intake and check kidney/heart health.")

            if "lung_capacity" in f:
                if value < 80:
                    recommendations.append("Low lung capacity. Perform breathing exercises and maintain lung health.")

            if "oldpeak" in f:
                if value > 2:
                    recommendations.append("High oldpeak indicates heart strain. Consult a cardiologist.")

            if "max_hr" in f:
                if value > 180:
                    recommendations.append("Maximum heart rate too high during exertion. Monitor exercise intensity.Avoid pushing beyond your maximum heart rate during exercise, which can be calculated as 220 minus your age. Use a heart rate monitor to stay within safe limits and consult a healthcare provider if you experience irregular heartbeats or other cardiac symptoms.")

            if "stress" in f:
                if value > 7:
                    recommendations.append("High stress detected. Practice relaxation techniques and consult a professional.Deep breathing exercises, meditation, and regular physical activity can help manage stress levels. Consider seeking support from a mental health professional if stress is overwhelming.")

            if "alcohol" in f or "alco" in f:
                if value > 2:
                    recommendations.append("High alcohol consumption. Reduce intake to lower health risks.")

            if "sleep_hours" in f:
                if value < 6:
                    recommendations.append("Insufficient sleep detected. Aim for 7 to 8 hours per night.")

            if "waist_circumference" in f:
                if value > 102:
                    recommendations.append("High waist circumference indicates visceral fat risk. Consider weight reduction.Do exercises that target abdominal fat, such as cardio and core strengthening. Maintain a balanced diet rich in fiber and low in processed foods to help reduce visceral fat.")

            if "triglycerides" in f:
                if value > 150:
                    recommendations.append("High triglycerides detected. Reduce sugar and fatty food intake.")

            if "hdl" in f:
                if value < 40:
                    recommendations.append("Low HDL cholesterol. Increase physical activity and healthy fats in diet.")

            if "ldl" in f:
                if value > 130:
                    recommendations.append("High LDL cholesterol. Adopt heart-healthy diet and exercise regularly.")

        if recommendations:
            for r in recommendations:
                st.info(r)
        else:
            st.success("No critical lifestyle risk factors detected.")

        # -------------------------
        # Counterfactual Insights (UNCHANGED 100%)
        # -------------------------
        if "structured_cf" in st.session_state:

            st.subheader("Counterfactual Insights")

            cf_data = st.session_state["structured_cf"]

            all_cf_recommendations = []

            for cf in cf_data:

                st.markdown(f"### {cf['scenario']}")
                st.success(cf["counterfactual"])
                st.info(cf["reason"])

                for rec in cf["recommendations"]:
                    st.write(f"- {rec}")
                    all_cf_recommendations.append(rec)

            combined_recommendations = list(set(recommendations + all_cf_recommendations))

        else:
            combined_recommendations = recommendations

        # -------------------------
        # Doctor Notes (ONLY FIXED PART)
        # -------------------------
        st.subheader("Doctor's Notes")

        if "doctor_notes" not in st.session_state:
            st.session_state.doctor_notes = ""

        notes_input = st.text_area(
            "Write clinical observations or prescription:",
            value=st.session_state.doctor_notes,
            height=150
        )

        if st.button("Apply Notes"):
            st.session_state.doctor_notes = notes_input
            st.success("Doctor notes saved!")

        doctor_notes = st.session_state.doctor_notes

        # -------------------------
        # Report Generation
        # -------------------------
        builder = ReportBuilder()

        final_recommendations_text = (
            "\n".join(combined_recommendations)
            if combined_recommendations else
            "Maintain healthy lifestyle habits."
        )

        cf_text = ""
        if "structured_cf" in st.session_state:
            for cf in st.session_state["structured_cf"]:
                cf_text += f"{cf['scenario']}:\n{cf['counterfactual']}\nReason: {cf['reason']}\n\n"

        sections = [
            ("Prediction Result", str(prediction)),
            ("Prediction Confidence", str(round(probability, 3)) if probability else "Not Available"),
            ("Interpretation", "Model-based prediction generated from input features."),
            ("Counterfactual Insights", cf_text if cf_text else "Not available"),
            ("Recommendations", final_recommendations_text),
            ("Doctor Notes", doctor_notes if doctor_notes else "No additional notes.")
        ]

        file_path = builder.build("medical_report.pdf", sections)

        st.subheader("Download Report")

        with open(file_path, "rb") as file:
            st.download_button(
                label="Download Full Medical Report",
                data=file,
                file_name="medical_report.pdf",
                mime="application/pdf"
            )