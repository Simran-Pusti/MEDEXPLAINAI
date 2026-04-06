import dice_ml
import pandas as pd


class CounterfactualGenerator:


    def generate(self, df, model, target, query, mutable):


        # -----------------------------
        # STEP 1: VALIDATION
        # -----------------------------
        if not isinstance(query, pd.DataFrame):
            query = pd.DataFrame([query])


        feature_cols = df.drop(columns=[target]).columns
        query = query[feature_cols]


        # -----------------------------
        # STEP 2: DiCE SETUP
        # -----------------------------
        data = dice_ml.Data(
            dataframe=df,
            continuous_features=feature_cols.tolist(),
            outcome_name=target
        )


        model_dice = dice_ml.Model(
            model=model,
            backend="sklearn"
        )


        dice = dice_ml.Dice(data, model_dice)


        # -----------------------------
        # STEP 3: GENERATE CFs
        # -----------------------------
        cf = dice.generate_counterfactuals(
            query,
            total_CFs=3,
            desired_class="opposite",
            features_to_vary=mutable
        )


        cf_df = cf.cf_examples_list[0].final_cfs_df


        if target in cf_df.columns:
            cf_df = cf_df.drop(columns=[target])


        # -----------------------------
        # STEP 4: AUTO SAFE LIMITS
        # -----------------------------
        for col in cf_df.columns:
            try:
                min_val = df[col].min()
                max_val = df[col].max()
                cf_df[col] = cf_df[col].clip(min_val, max_val)
            except:
                continue


        # -----------------------------
        # STEP 5: BUILD STRUCTURED OUTPUT
        # -----------------------------
        structured_output = []
        original = query.iloc[0]


        for idx, row in cf_df.iterrows():


            changes = []
            recommendations = []


            for feature in mutable:


                if feature not in row or feature not in original:
                    continue


                try:
                    o = float(original[feature])
                    n = float(row[feature])
                except:
                    continue


                if abs(o - n) < 0.01:
                    continue


                feature_lower = feature.lower()


                # -----------------------------
                # REASON
                # -----------------------------
                if "glucose" in feature_lower or "sugar" in feature_lower or "gluc" in feature_lower:
                    reason = "Blood glucose levels strongly influence diabetes risk."
                elif "bmi" in feature_lower or "weight" in feature_lower:
                    reason = "Higher body weight increases insulin resistance."
                elif "pressure" in feature_lower or "bp" in feature_lower or "bloodpressure" in feature_lower:
                    reason = "Blood pressure is linked with metabolic and heart health."
                elif "cholesterol" in feature_lower:
                    reason = "Cholesterol affects cardiovascular and metabolic health."
                elif "insulin" in feature_lower:
                    reason = "Insulin levels directly affect blood sugar regulation. Main indicator of diabetes risk."
                elif "fatigue" in feature_lower:
                    reason = "Fatigue can indicate underlying metabolic or organ stress. Connected to overall health status."
                elif "dizziness" in feature_lower:
                    reason = "Dizziness may be linked to blood pressure, oxygen levels, or anemia."
                elif "chest_pain" in feature_lower:
                    reason = "Chest pain is a key indicator of cardiovascular issues."
                elif "cough" in feature_lower:
                    reason = "Cough can signal lung function issues or respiratory infections."
                elif "swelling" in feature_lower or "edema" in feature_lower:
                    reason = "Swelling may indicate kidney, heart, or liver problems."
                elif "shortness_of_breath" in feature_lower or "resp_rate" in feature_lower or "oxygen_level" in feature_lower:
                    reason = "Shortness of breath reflects oxygenation, lung capacity, or heart function."
                elif "lung_capacity" in feature_lower:
                    reason = "Low lung capacity can indicate chronic lung conditions or poor respiratory fitness."
                elif "urea" in feature_lower or "creatinine" in feature_lower:
                    reason = "Elevated urea/creatinine levels indicate reduced kidney function."
                elif "hemoglobin" in feature_lower:
                    reason = "Low hemoglobin may indicate anemia or blood disorders."
                elif "sodium" in feature_lower or "potassium" in feature_lower:
                    reason = "Electrolyte imbalance can affect heart, kidney, and muscle function."
                elif "max_hr" in feature_lower:
                    reason = "Maximum heart rate during exercise indicates cardiovascular fitness and stress response."
                elif "oldpeak" in feature_lower:
                    reason = "Oldpeak shows heart strain during exertion; higher values indicate potential risk."
                else:
                    reason = "This feature significantly impacts the model prediction."


                # -----------------------------
                # ADVICE
                # -----------------------------
                if "age" in feature_lower:
                    advice = "Age cannot be changed. Focus on preventive care, regular checkups, and maintaining a healthy lifestyle."
                elif "weight" in feature_lower or "bmi" in feature_lower:
                    advice = "Maintain healthy weight with a balanced diet and regular exercise. Take protein-rich foods, limit processed foods, and engage in both cardio and strength training."
                elif "pressure" in feature_lower or "bp" in feature_lower or "bloodpressure" in feature_lower:
                    advice = "Reduce salt intake, manage stress, monitor blood pressure, and follow medical advice."
                elif "cholesterol" in feature_lower:
                    advice = "Adopt a low-fat, high-fiber diet, exercise regularly, and consult a doctor periodically.Avoid fried foods, choose lean proteins, and include plenty of fruits and vegetables."
                elif "sugar" in feature_lower or "glucose" in feature_lower:
                    advice = "Control sugar intake, monitor glucose levels, and maintain a healthy diet. Avoid high-sugar foods, maintain healthy meals, and monitor fasting glucose. Aim for a balanced diet with complex carbs, fiber, and lean proteins. Regular physical activity helps improve insulin sensitivity."
                elif "smok" in feature_lower or "smoking" in feature_lower or "smoke" in feature_lower:
                    advice = "Quit smoking; seek support or medical help if needed.Nicotine replacement therapy, counseling, and support groups can assist in quitting smoking. Avoiding secondhand smoke is also important for overall health."
                elif "exercise" in feature_lower or "activity" in feature_lower:
                    advice = "Increase daily physical activity gradually; include cardio and strength training.Do exercises you enjoy to stay consistent. Aim for at least 150 minutes of moderate-intensity exercise per week, such as brisk walking, cycling, or swimming. Strength training should be done at least twice a week to build muscle and improve metabolism."
                elif "insulin" in feature_lower:
                    advice = "Consult a doctor for proper insulin management and follow prescribed doses."
                elif "fatigue" in feature_lower:
                    advice = "Rest adequately, monitor energy levels, and consult a doctor if persistent."
                elif "dizziness" in feature_lower:
                    advice = "Stay hydrated, avoid sudden movements, check BP regularly, and consult a doctor if frequent."
                elif "chest_pain" in feature_lower:
                    advice = "Seek medical attention immediately if severe; maintain heart health with diet and exercise."
                elif "cough" in feature_lower:
                    advice = "Avoid pollutants, stay hydrated, and consult a doctor if cough persists."
                elif "swelling" in feature_lower or "edema" in feature_lower:
                    advice = "Monitor fluid intake, reduce salt, and consult a doctor if swelling occurs."
                elif "shortness_of_breath" in feature_lower or "resp_rate" in feature_lower or "oxygen_level" in feature_lower:
                    advice = "Practice breathing exercises, maintain good air quality, and consult a doctor if persistent."
                elif "lung_capacity" in feature_lower:
                    advice = "Perform respiratory exercises, avoid smoking, and maintain lung health."
                elif "urea" in feature_lower or "creatinine" in feature_lower:
                    advice = "Monitor kidney function, stay hydrated, maintain a healthy diet, and consult a doctor if abnormal.Atleast 8 glasses of water per day, limit salt and protein intake, and avoid nephrotoxic medications. Regular check-ups with a healthcare provider are essential for managing kidney health."
                elif "hemoglobin" in feature_lower:
                    advice = "Maintain hemoglobin levels through iron-rich diet and regular checkups.Add iron-rich foods such as lean meats, beans, lentils, and leafy greens in your diet. If anemia is suspected, seek medical evaluation for proper diagnosis and treatment."
                elif "sodium" in feature_lower or "potassium" in feature_lower:
                    advice = "Ensure proper electrolyte balance via diet and hydration; consult a doctor if abnormal."
                elif "max_hr" in feature_lower:
                    advice = "Monitor maximum heart rate during exercise and avoid overexertion."
                elif "oldpeak" in feature_lower:
                    advice = "Maintain cardiovascular fitness; consult a doctor if oldpeak indicates high strain."
                else:
                    advice = "Improve lifestyle habits, maintain a healthy diet and exercise routine, and consult a healthcare provider."


                # -----------------------------
                # BUILD CHANGE SENTENCE
                # -----------------------------
                if n > o:
                    change_text = f"increase {feature} from {round(o,2)} to {round(n,2)}"
                else:
                    change_text = f"reduce {feature} from {round(o,2)} to {round(n,2)}"


                changes.append(change_text)
                recommendations.append(f"{feature}: {advice}")


            # -----------------------------
            # FINAL SENTENCE BUILD
            # -----------------------------
            if changes:
                counterfactual_sentence = (
                    "If the patient " + ", ".join(changes) +
                    ", the prediction is likely to change."
                )


                structured_output.append({
                    "scenario": f"Scenario {idx+1}",
                    "counterfactual": counterfactual_sentence,
                    "reason": reason,
                    "recommendations": list(set(recommendations))  # remove duplicates
                })


        # -----------------------------
        # RETURN UPDATED OUTPUT
        # -----------------------------
        return cf_df, structured_output