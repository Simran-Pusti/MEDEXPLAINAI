import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="MedExplain AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD CUSTOM CSS
# -----------------------------
def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

# -----------------------------
# INITIALIZE SESSION STATE
# -----------------------------
def init_state():

    defaults = {
        "dataset": None,
        "target": None,
        "mutable": [],
        "immutable": [],
        "model": None,
        "X_train": None,
        "X_test": None,
        "selected_patient_index": 0,
        "cf_df": None,
        "auto_recommendations": None,
        "query": None,
        "current_pred": None,
        "current_prob": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# -----------------------------
# MAIN PAGE CONTENT
# -----------------------------
st.title("MedExplain AI")
st.subheader("Explainable Medical Decision Support System")

st.markdown("""
This system helps doctors and researchers analyze medical datasets
and generate explainable disease predictions.

---

###  System Capabilities

• Medical dataset analysis  
• Machine learning disease prediction  
• Explainable AI (SHAP & LIME)  
• Causal relationship discovery  
• Counterfactual reasoning  
• Doctor decision support  

---

###  How to Use

1. Go to **Technocrat Panel** → Upload dataset  
2. Select **Target variable & features**  
3. Train model in **Model Training**  
4. Explore insights in:
   - Explainable AI  
   - Causal Analysis  
   - Counterfactual Reasoning  
5. Use **Doctor Panel** for final decision support  

---

###  Important

- Target variable is selected only once  
- Data persists across pages (no reset)  
- Counterfactual results stay unless manually changed  
""")

# -----------------------------
# SIDEBAR INFO
# -----------------------------
st.sidebar.title("Navigation")

st.sidebar.info("""
Use the sidebar to navigate through modules:

1. Technocrat Panel  
2. Model Training  
3. Explainable AI  
4. Causal Analysis  
5. Counterfactuals  
6. Doctor Panel  
""")

# -----------------------------
# STATUS DISPLAY (VERY USEFUL)
# -----------------------------
st.sidebar.subheader("System Status")

if st.session_state["dataset"] is not None:
    st.sidebar.success("Dataset Loaded")
else:
    st.sidebar.warning("Dataset Not Loaded")

if st.session_state["target"] is not None:
    st.sidebar.success(f"Target: {st.session_state['target']}")
else:
    st.sidebar.warning("Target Not Selected")

if st.session_state["model"] is not None:
    st.sidebar.success("Model Trained")
else:
    st.sidebar.warning("Model Not Trained")