import streamlit as st
import pandas as pd
from state_manager import init_state
from backend.data_engine.dataset_loader import DatasetLoader
from backend.data_engine.dataset_validator import DatasetValidator
from backend.data_engine.feature_detector import FeatureDetector
from utils.styles import load_css
load_css()

init_state()

st.title("Technocrat Panel")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv","xlsx"])

if uploaded_file:

    loader = DatasetLoader()
    validator = DatasetValidator()
    detector = FeatureDetector()

    df = loader.load_dataset(uploaded_file)

    #  RESET STATE IF NEW DATASET
    if "dataset" not in st.session_state or st.session_state["dataset"] is None:
        st.session_state.clear()
        init_state()

    st.session_state["dataset"] = df

    # -----------------------------
    # DATA PREVIEW
    # -----------------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -----------------------------
    # VALIDATION REPORT
    # -----------------------------
    report = validator.validate(df)

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", int(report["rows"]))
    col2.metric("Columns", int(report["columns"]))
    col3.metric("Missing Values", int(report["missing_values"]))

    # -----------------------------
    # COLUMN TYPES (CLEAN TABLE)
    # -----------------------------
    st.subheader("Column Data Types")

    dtype_df = pd.DataFrame({
        "Column": list(report["column_types"].keys()),
        "Type": [
            str(v).replace("dtype(", "").replace(")", "").replace("'", "")
            for v in report["column_types"].values()
        ]
    })

    st.dataframe(dtype_df, use_container_width=True)

    # -----------------------------
    # FEATURE DETECTION
    # -----------------------------
    features = detector.detect_features(df)

    st.subheader("Feature Classification")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("###  Numeric")
        for f in features["numeric"]:
            st.write(f"• {f}")

    with f2:
        st.markdown("###  Categorical")
        if features["categorical"]:
            for f in features["categorical"]:
                st.write(f"• {f}")
        else:
            st.write("No categorical features")

    with f3:
        st.markdown("###  Binary")
        for f in features["binary"]:
            st.write(f"• {f}")

    # -----------------------------
    # TARGET SELECTION (SAFE)
    # -----------------------------
    st.subheader("Target Selection")

    target_index = 0
    if st.session_state["target"] in df.columns:
        target_index = df.columns.get_loc(st.session_state["target"])

    target = st.selectbox(
        "Select Target",
        df.columns,
        index=target_index
    )

    st.session_state["target"] = target

    # -----------------------------
    # MUTABLE FEATURES (FIXED)
    # -----------------------------
    st.subheader("Mutable Feature Selection")

    all_features = [c for c in df.columns if c != target]

    #  FIX: REMOVE INVALID DEFAULTS
    previous_mutable = st.session_state.get("mutable", [])

    valid_defaults = [
        col for col in previous_mutable
        if col in all_features
    ]

    if not valid_defaults:
        valid_defaults = all_features

    mutable = st.multiselect(
        "Mutable Features",
        options=all_features,
        default=valid_defaults
    )

    st.session_state["mutable"] = mutable

    immutable = [c for c in all_features if c not in mutable]

    st.write("Immutable Features:", immutable)

    st.session_state["immutable"] = immutable

    st.success("Dataset configured successfully")