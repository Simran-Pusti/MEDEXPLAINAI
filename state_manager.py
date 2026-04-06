import streamlit as st

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
        "cf_results": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value