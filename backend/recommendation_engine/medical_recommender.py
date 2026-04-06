class MedicalRecommender:

    def generate(self, original, counterfactual):

        recommendations = []

        for feature in original.columns:

            if feature not in counterfactual.columns:
                continue

            try:
                o = float(original.iloc[0][feature])
                n = float(counterfactual.iloc[0][feature])
            except:
                continue

            diff = n - o

            if abs(diff) < 0.01:
                continue

            feature_lower = feature.lower()

            # -----------------------------
            # MEDICAL LOGIC (CLEANED TEXT)
            # -----------------------------
            if "age" in feature_lower:
                advice = (
                    "Age cannot be modified. Focus on preventive healthcare, "
                    "regular check-ups, and maintaining a healthy lifestyle."
                )

            elif "weight" in feature_lower or "bmi" in feature_lower:
                advice = (
                    "Maintain optimal body weight through a balanced diet and regular exercise. "
                    "Achieving a healthy BMI reduces the risk of diabetes, heart, and kidney diseases."
                    "Take protein-rich foods, limit processed foods, and engage in both cardio and strength training."
                )

            elif "pressure" in feature_lower or "bp" in feature_lower:
                advice = (
                    "Manage blood pressure by reducing salt intake, controlling stress, "
                    "exercising regularly, and monitoring BP. Medication may be required if advised."
                )

            elif "cholesterol" in feature_lower:
                advice = (
                    "Adopt a heart-healthy diet low in saturated fats and high in fiber. "
                    "Regular exercise and periodic medical consultation are recommended."
                    "Avoid fried foods, choose lean proteins, and include plenty of fruits and vegetables."
                )

            elif "glucose" in feature_lower or "sugar" in feature_lower:
                advice = (
                    "Control blood sugar through diet, exercise, and regular monitoring. "
                    "Avoid high-sugar foods, maintain healthy meals, and monitor fasting glucose."
                    "Aim for a balanced diet with complex carbs, fiber, and lean proteins. Regular physical activity helps improve insulin sensitivity."
                )

            elif "insulin" in feature_lower:
                advice = (
                    "Consult a healthcare provider for insulin management. "
                    "Regular monitoring, proper dosage, and adherence to medical advice are essential."
                    "If insulin levels are high, take injections as prescribed, monitor blood sugar closely, and maintain a healthy lifestyle to improve insulin sensitivity."
                )

            elif "smok" in feature_lower or "smoke" in feature_lower or "smoking" in feature_lower:
                advice = (
                    "Smoking cessation is strongly recommended. Quitting smoking "
                    "reduces risks of heart disease, stroke, lung disorders, and cancer."
                    "Nicotine replacement therapy, counseling, and support groups can assist in quitting smoking. Avoiding secondhand smoke is also important for overall health."
                )

            elif "exercise" in feature_lower or "activity" in feature_lower:
                advice = (
                    "Increase physical activity with a consistent exercise routine. "
                    "Include both cardio and strength training for optimal health benefits."
                    "Aim for at least 150 minutes of moderate-intensity exercise per week, such as brisk walking, cycling, or swimming. Strength training should be done at least twice a week to build muscle and improve metabolism."
                )

            elif "fatigue" in feature_lower:
                advice = (
                    "Monitor energy levels and ensure adequate rest. Persistent fatigue "
                    "may indicate underlying health issues; consult a doctor if necessary."
                )

            elif "dizziness" in feature_lower:
                advice = (
                    "Stay hydrated, avoid sudden movements, and monitor blood pressure. "
                    "Seek medical advice if dizziness is frequent, severe, or associated with other symptoms."
                )

            elif "chest_pain" in feature_lower:
                advice = (
                    "Seek immediate medical attention if chest pain occurs. "
                    "Maintain cardiovascular health through diet, exercise, and stress management."
                )

            elif "cough" in feature_lower:
                advice = (
                    "Avoid pollutants, maintain hydration, and monitor respiratory health. "
                    "Persistent cough should be evaluated by a healthcare provider."
                    "try levocitizin, a non-drowsy antihistamine, to relieve cough symptoms. Avoid irritants and consider using a humidifier to soothe the respiratory tract."
                )

            elif "swelling" in feature_lower or "edema" in feature_lower:
                advice = (
                    "Monitor fluid intake, reduce salt consumption, and consult a doctor. "
                    "Swelling may indicate kidney, heart, or liver issues."
                )

            elif "shortness_of_breath" in feature_lower or "resp_rate" in feature_lower or "oxygen_level" in feature_lower:
                advice = (
                    "Practice breathing exercises, avoid smoke or pollution, "
                    "and seek medical evaluation if symptoms persist. Maintain good lung health."
                    "Do less strenuous activities, practice deep breathing exercises, and avoid exposure to pollutants. If shortness of breath is severe or persistent, seek immediate medical attention."
                )

            elif "lung_capacity" in feature_lower:
                advice = (
                    "Engage in respiratory exercises, maintain clean air quality, "
                    "and avoid smoking to improve lung capacity and overall breathing efficiency."
                )

            elif "urea" in feature_lower or "creatinine" in feature_lower:
                advice = (
                    "Monitor kidney function regularly. Maintain hydration, a balanced diet, "
                    "and consult a doctor for abnormalities in urea or creatinine levels."
                    "Drink at least 8 glasses of water per day, limit salt and protein intake, and avoid nephrotoxic medications. Regular check-ups with a healthcare provider are essential for managing kidney health."
                )

            elif "hemoglobin" in feature_lower:
                advice = (
                    "Maintain adequate hemoglobin levels through iron-rich foods. "
                    "Consult a doctor if anemia symptoms are present."
                    "Include iron-rich foods such as lean meats, beans, lentils, and leafy greens in your diet. If anemia is suspected, seek medical evaluation for proper diagnosis and treatment."
                )

            elif "sodium" in feature_lower or "potassium" in feature_lower:
                advice = (
                    "Maintain electrolyte balance through diet and hydration. "
                    "Abnormal levels may indicate kidney or heart issues and require medical attention."
                    "Avoid excessive salt intake to manage sodium levels, and consume potassium-rich foods like bananas, oranges, and spinach. Regular monitoring of electrolyte levels is important for overall health."
                )

            elif "max_hr" in feature_lower:
                advice = (
                    "Monitor maximum heart rate during exercise and avoid overexertion. "
                    "Consult a doctor for heart rate abnormalities."
                    "Avoid pushing beyond your maximum heart rate during exercise, which can be calculated as 220 minus your age. Use a heart rate monitor to stay within safe limits and consult a healthcare provider if you experience irregular heartbeats or other cardiac symptoms."
                )

            elif "oldpeak" in feature_lower:
                advice = (
                    "Oldpeak indicates heart strain during exercise. Maintain cardiovascular fitness "
                    "and consult a healthcare provider for high-risk conditions."
                )

            else:
                advice = (
                    "Improving this parameter through lifestyle modification, diet, and medical guidance "
                    "can reduce health risks and improve overall well-being."
                )

            # -----------------------------
            # BUILD OUTPUT (IMPROVED FORMAT)
            # -----------------------------
            if diff > 0:
                change_line = f"Increase {feature} from {round(o,2)} to {round(n,2)}"
            else:
                change_line = f"Reduce {feature} from {round(o,2)} to {round(n,2)}"

            final_text = (
                f"{change_line}.\n"
                f"→ Recommendation: {advice}"
            )

            recommendations.append(final_text)

        # -----------------------------
        # FALLBACK
        # -----------------------------
        if not recommendations:
            recommendations.append(
                "All parameters are within normal range. Maintain a healthy lifestyle and regular monitoring."
            )

        return recommendations