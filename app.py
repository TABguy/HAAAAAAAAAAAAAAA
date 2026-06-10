"""HAKS 2026 — Streamlit demo shell.

One tab per candidate use case, wired to the real src/ modules. After the 13:00
brief, keep ONLY the chosen tab and polish it. Run:  streamlit run app.py

Use cases #2 (triage) and #3 (anomaly) run fully offline. #1 (RAG) needs AWS
Bedrock creds + model IDs in .env (run `python src/common.py` to check).
"""
import pandas as pd
import streamlit as st

from src import anomaly, nlp_triage, rag

st.set_page_config(page_title="HAKS 2026 — Airbus × AWS", layout="wide")
st.title("HAKS 2026 — Airbus maintenance assistant")
st.caption("Scaffold wired to real modules — pick the use case after the 13:00 brief.")

tab_rag, tab_triage, tab_anomaly = st.tabs(
    ["📄 RAG doc assistant", "🗂️ Maintenance triage", "📈 Anomaly detection"]
)

# ---------------------------------------------------------------- Use case #1: RAG
with tab_rag:
    st.subheader("Ask the technical documentation")
    st.caption("Retrieval over input/ docs → Bedrock → answer with cited sources. Needs AWS.")

    ready = rag.index_exists()
    st.write(f"Vector index present: {'✅' if ready else '❌ — build it first'}")
    if st.button("Build / rebuild index", key="rag_build"):
        with st.spinner("Ingesting input/ into Chroma via Bedrock embeddings…"):
            try:
                n = __import__("src.ingest", fromlist=["build_index"]).build_index()
                st.success(f"Indexed {n} nodes.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Ingest failed: {e}")

    q = st.text_input("Question", placeholder="Montre-moi la procédure pour l'APU (ATA 49)…")
    if st.button("Ask", key="rag_ask") and q:
        try:
            ans = rag.answer_question(q)
            st.markdown(ans.answer)
            with st.expander(f"Sources ({len(ans.sources)})", expanded=True):
                for s in ans.sources:
                    st.markdown(
                        f"- **{s.get('ref','?')}** · ATA {s.get('ata') or '—'} "
                        f"· score {s.get('score'):.3f}" if s.get("score") is not None
                        else f"- **{s.get('ref','?')}** · ATA {s.get('ata') or '—'}"
                    )
                    st.caption(s.get("snippet", ""))
        except Exception as e:  # noqa: BLE001
            st.error(f"{e}")

# ----------------------------------------------------------- Use case #2: NLP triage
with tab_triage:
    st.subheader("Triage a maintenance report")
    st.caption("Stage 1 spaCy NER + rules (offline) → optional Bedrock enrichment.")

    report = st.text_area(
        "Free-text entry",
        value="[crew report] A320 reg F-GABC: crew reported pack overheat warning. "
        "ATA 21. FIN 12X3. replaced flow control valve.",
    )
    if st.button("Triage", key="triage_btn") and report.strip():
        r = nlp_triage.triage(report)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATA chapter", r.ata_chapter or "—")
        c2.metric("Component", r.component or "—")
        c3.metric("Criticality", r.criticality or "—")
        c4.metric("Entities", len(r.entities))
        st.json(
            {"symptom": r.symptom, "action": r.action,
             "duplicate_of": r.duplicate_of, "entities": r.entities}
        )

    st.divider()
    if st.button("Run batch over input/maintenance_logs.jsonl", key="triage_batch"):
        with st.spinner("Triaging corpus…"):
            results = nlp_triage.triage_file()
            ev = nlp_triage.evaluate()
        a, b, c = st.columns(3)
        a.metric("Records", ev["n"])
        b.metric("ATA accuracy (Stage 1)", f"{ev['ata_accuracy']:.0%}")
        c.metric("Duplicates found", ev["n_duplicates"])
        st.dataframe(pd.DataFrame([
            {"ata": x.ata_chapter, "component": x.component, "criticality": x.criticality,
             "dup_of": x.duplicate_of, "raw": x.raw[:80]} for x in results
        ]))

# ------------------------------------------------------ Use case #3: anomaly detection
with tab_anomaly:
    st.subheader("Component health")
    st.caption("IsolationForest over synthetic sensor series — runs fully offline.")

    comps = anomaly.list_components()
    if not comps:
        st.warning("No sensor data. Run: python src/gen_sensors.py")
    else:
        comp = st.selectbox("Sensor / component", comps)
        if comp:
            res = anomaly.detect(comp)
            ev = anomaly.evaluate(comp)
            a, b, c, d = st.columns(4)
            a.metric("Health", f"{res.health:.0%}")
            b.metric("Anomalies flagged", len(res.flagged))
            c.metric("Recall vs truth", f"{ev['recall']:.0%}")
            d.metric("Precision", f"{ev['precision']:.0%}")
            st.plotly_chart(anomaly.plot_component(comp), use_container_width=True)
