# MedExplain AI  
### Causality Aware Explainable AI Recommendation using Counterfactual Reasoning 
## Live Application
**Deployed App:** https://medexplainai-db5vrw6qyp27frkc69kq4b.streamlit.app/ 

## Problem Statement
Traditional machine learning models in healthcare often act as **black boxes**, providing predictions without explanations. This lack of transparency reduces trust among doctors and patients.

##  Proposed Solution
MedExplain AI addresses this issue by combining **disease prediction** with **Explainable AI (XAI)** techniques, allowing users to understand:
- *Why* a prediction was made  
- *Which features influenced the outcome*  
- *What changes could alter the result*  

## Objectives
- Develop an accurate disease prediction system  
- Integrate explainability techniques (SHAP, LIME)  
- Provide actionable insights using counterfactual analysis  
- Build a user-friendly medical decision support interface  

## Key Features
-  **Disease Prediction Engine** (ML-based)  
-  **Explainability Module** (SHAP & LIME visualizations)  
-  **Counterfactual Generator** (What-if analysis)  
-  **Causal Analysis Engine**  
-  **Doctor Panel Interface**  
-  **Automated Medical Report Generation**  

---

## System Architecture
User Input
↓
Data Engine → Validation & Feature Detection
↓
Model Engine → Training & Prediction
↓
Explainability Engine → SHAP / LIME
↓
Causal & Counterfactual Engine
↓
Recommendation Engine
↓
Report Generation → Output to UI


---

## Technology Stack

| Category            | Tools / Libraries |
|--------------------|------------------|
| Language           | Python           |
| Frontend           | Streamlit        |
| Machine Learning   | Scikit-learn     |
| Data Processing    | Pandas, NumPy    |
| Explainability     | SHAP, LIME       |
| Version Control    | Git & GitHub     |


---

##  Installation & Execution Guide

###  Clone Repository
```bash
git clone https://github.com/Simran-Pusti/MEDEXPLAINAI/
cd MedExplain-AI
pip install -r requirements.txt
streamlit run app.py
