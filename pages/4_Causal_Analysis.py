import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from backend.causal_engine.causal_builder import CausalBuilder
from utils.styles import load_css
load_css()


st.title("Causal Relationship Analysis")

if "dataset" not in st.session_state:

    st.warning("Upload dataset first")

else:

    df = st.session_state["dataset"]
    df = df.apply(pd.to_numeric, errors="coerce")

    st.subheader("Select Target Variable")

    #  FIX: persist target
    target = st.selectbox(
        "Target Variable",
        df.columns,
        index=list(df.columns).index(st.session_state.get("target", df.columns[-1]))
    )

    st.session_state["target"] = target

    builder = CausalBuilder()

    graph, weights = builder.build_weighted_graph(df, target)

    # -----------------------------
    # SHOW TOP IMPORTANT FACTORS ONLY
    # -----------------------------
    st.subheader("Top Causal Factors")

    if len(weights) == 0:
        st.warning("No causal relationships detected.")

    else:

        #  SHOW ONLY TOP 5 (IMPORTANT)
        top_weights = dict(list(weights.items())[:5])

        for k, v in top_weights.items():
            st.metric(label=f"{k} → {target}", value=f"{round(v,2)}% impact")

    # -----------------------------
    # CLEAN GRAPH (TOP FEATURES ONLY)
    # -----------------------------
    if len(weights) > 0:

        G = nx.DiGraph()

        for feature, value in top_weights.items():
            G.add_edge(feature, target, weight=value)

        fig = plt.figure(figsize=(7, 5))

        pos = nx.spring_layout(G, k=1.5)

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="#87CEEB",
            node_size=2500,
            font_size=10
        )

        edge_labels = {
            (u, v): f"{round(d['weight'],2)}%"
            for u, v, d in G.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels
        )

        st.pyplot(fig)

    # -----------------------------
    # USER FRIENDLY INTERPRETATION
    # -----------------------------
    st.subheader("Interpretation")

    if len(weights) > 0:
        top_feature = list(weights.keys())[0]

        st.success(
            f"The most influential factor affecting {target} is '{top_feature}'. "
            f"This means changes in this feature strongly impact the prediction."
        )