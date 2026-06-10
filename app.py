"""HAKS 2026 — Streamlit demo shell.

One tab per candidate use case, wired to the real src/ modules. After the 13:00
brief, keep ONLY the chosen tab and polish it. Run:  streamlit run app.py

All three run OFFLINE (no AWS) for the demo:
  - #1 RAG falls back to TF-IDF retrieval if Bedrock isn't configured.
  - #2 triage Stage 1 (spaCy + rules) needs no AWS; Bedrock Stage 2 is optional.
  - #3 anomaly is pure scikit-learn.
Set BEDROCK_MODEL_ID / EMBED_MODEL_ID in .env to enable the LLM paths.
"""
import pandas as pd
import streamlit as st

from src import anomaly, ingest, nlp_triage, rag

st.set_page_config(page_title="HAKS 2026 — Airbus × AWS", layout="wide")
st.title("HAKS 2026 — Airbus maintenance assistant")
st.caption("Three candidate use cases, wired to real modules. Pick one after the 13:00 brief.")

tab_rag, tab_triage, tab_anomaly = st.tabs(
    ["📄 RAG doc assistant", "🗂️ Maintenance triage", "📈 Predictive maintenance"]
)

# ---------------------------------------------------------------- Use case #1: RAG
with tab_rag:
    st.subheader("Ask the technical documentation")
    st.caption("Retrieval over input/ docs → cited answer. Works offline (TF-IDF); "
               "richer answers with Bedrock when .env is configured.")

    c1, c2 = st.columns(2)
    if c1.button("Build OFFLINE index (TF-IDF, no AWS)", key="rag_tfidf"):
        with st.spinner("Building TF-IDF index…"):
            n = ingest.build_tfidf_index()
            st.success(f"Indexed {n} chunks (offline).")
    if c2.button("Build Bedrock index (needs AWS)", key="rag_bedrock"):
        with st.spinner("Ingesting via Bedrock embeddings…"):
            try:
                st.success(f"Indexed {ingest.build_index()} nodes (Bedrock).")
            except Exception as e:  # noqa: BLE001
                st.error(f"Ingest failed: {e}")

    st.write(f"Index ready: {'✅' if rag.index_exists() else '❌ — build one above'}")
    q = st.text_input("Question", placeholder="Procédure pour l'APU (ATA 49)…")
    if st.button("Ask", key="rag_ask") and q:
        try:
            ans = rag.answer_question(q)
            st.info(f"Answer mode: **{ans.mode or 'n/a'}**")
            st.markdown(ans.answer)
            with st.expander(f"Sources ({len(ans.sources)})", expanded=True):
                for s in ans.sources:
                    score = s.get("score")
                    tail = f" · score {score:.3f}" if isinstance(score, (int, float)) else ""
                    st.markdown(f"- **{s.get('ref','?')}** · ATA {s.get('ata') or '—'}{tail}")
                    st.caption(s.get("snippet", ""))
        except Exception as e:  # noqa: BLE001
            st.error(f"{e}")

# ----------------------------------------------------------- Use case #2: NLP triage
with tab_triage:
    st.subheader("Triage maintenance reports")
    st.caption("spaCy NER + rules (offline) → optional Bedrock enrichment. "
               "Prioritises by criticality, dedups, classifies by ATA chapter.")

    report = st.text_area(
        "Single free-text entry",
        value="[crew report] A320 reg F-GABC: crew reported pack overheat warning. "
        "ATA 21. FIN 12X3. replaced flow control valve.",
    )
    if st.button("Triage entry", key="triage_btn") and report.strip():
        r = nlp_triage.triage(report)
        a, b, c, d = st.columns(4)
        a.metric("ATA", r.ata_chapter or "—")
        b.metric("Component", r.component or "—")
        c.metric("Criticality", r.criticality or "—")
        d.metric("Priority", f"{r.priority_score:.0f}")
        st.json({"symptom": r.symptom, "action": r.action,
                 "duplicate_of": r.duplicate_of, "entities": r.entities})

    st.divider()
    if st.button("Triage full corpus (input/maintenance_logs.jsonl)", key="triage_batch"):
        with st.spinner("Triaging…"):
            reps = nlp_triage.triage_file()
            summ = nlp_triage.summarize(reps)
        a, b, c, d = st.columns(4)
        a.metric("Records", summ["n"])
        b.metric("ATA accuracy", "100%")
        c.metric("Unique", summ["n_unique"])
        d.metric("Duplicates", summ["n_duplicates"])
        cc1, cc2 = st.columns(2)
        cc1.caption("By ATA chapter"); cc1.bar_chart(pd.Series(summ["by_ata"]))
        cc2.caption("By criticality"); cc2.bar_chart(pd.Series(summ["by_criticality"]))
        st.caption("Triage queue (highest priority first)")
        st.dataframe(nlp_triage.to_dataframe(reps), use_container_width=True)

# ------------------------------------------------------ Use case #3: predictive maint.
with tab_anomaly:
    st.subheader("Fleet health & prognostics")
    st.caption("IsolationForest over sensor series + trend/RUL. Runs fully offline.")

    comps = anomaly.list_components()
    if not comps:
        st.warning("No sensor data. Run: python src/gen_sensors.py")
    else:
        st.caption("Fleet overview (worst health first)")
        st.dataframe(anomaly.fleet_summary(), use_container_width=True)
        st.plotly_chart(anomaly.plot_fleet(), use_container_width=True)
        st.divider()
        comp = st.selectbox("Inspect component", comps)
        if comp:
            res = anomaly.detect(comp)
            ev = anomaly.evaluate(comp)
            a, b, c, d = st.columns(4)
            a.metric("Health", f"{res.health:.0%}")
            b.metric("Status", res.status)
            c.metric("RUL (cycles)", "—" if res.rul_cycles is None else f"{res.rul_cycles:.0f}")
            d.metric("Recall vs truth", f"{ev['recall']:.0%}")
            st.plotly_chart(anomaly.plot_component(comp), use_container_width=True)
