"""HAKS 2026 — Streamlit demo shell.

One tab per candidate use case. After 13:00, keep ONLY the chosen tab and flesh it
out end-to-end. Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="HAKS 2026 — Airbus × AWS", layout="wide")
st.title("HAKS 2026 — Airbus maintenance assistant")
st.caption("Scaffold — pick the use case after the 13:00 brief.")

tab_rag, tab_triage, tab_anomaly = st.tabs(
    ["📄 RAG doc assistant", "🗂️ Maintenance triage", "📈 Anomaly detection"]
)

with tab_rag:
    st.subheader("Ask the technical documentation")
    q = st.text_input("Question", placeholder="Montre-moi la procédure pour…")
    if st.button("Ask", key="rag_ask"):
        st.info("TODO: wire src.rag.answer_question(q) and render answer + citations.")

with tab_triage:
    st.subheader("Triage a maintenance report")
    report = st.text_area("Free-text entry", placeholder="LH MLG tyre worn, replaced…")
    if st.button("Triage", key="triage_btn"):
        st.info("TODO: wire src.nlp_triage.triage(report) and render structured fields.")

with tab_anomaly:
    st.subheader("Component health")
    st.info("TODO: wire src.anomaly.detect(component) and plot scores/health.")
