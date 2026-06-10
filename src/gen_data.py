"""Synthetic data generator — FIRST deliverable on demo day.

One generator per candidate use case. Keep output realistic: ATA chapter codes,
aviation acronyms, noisy free text, some duplicates. Writes to data/generated/.

Run:  python src/gen_data.py [rag|triage|sensors|all]
"""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "generated"

# A few real ATA chapters to seed realistic data.
ATA_CHAPTERS = {
    "21": "Air Conditioning",
    "24": "Electrical Power",
    "27": "Flight Controls",
    "29": "Hydraulic Power",
    "32": "Landing Gear",
    "34": "Navigation",
    "49": "APU",
    "52": "Doors",
}


def gen_rag_corpus():
    """Use case #1: synthetic AMM/IPC/Service-Bulletin style docs (PDF or .txt)."""
    # TODO: emit structured procedures with ref numbers + ATA chapters for citation.
    raise NotImplementedError("Fill after brief: synthetic technical doc corpus.")


def gen_triage_reports():
    """Use case #2: a few hundred noisy free-text maintenance/PIREP entries."""
    # TODO: emit free text with acronyms, symptoms, components, a few duplicates.
    raise NotImplementedError("Fill after brief: synthetic logbook entries.")


def gen_sensor_series():
    """Use case #3: sensor time series with injected degradation + labelled anomalies."""
    # TODO: emit CSV per component, drift + spikes, an `anomaly` ground-truth column.
    raise NotImplementedError("Fill after brief: synthetic sensor series.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"rag": gen_rag_corpus, "triage": gen_triage_reports, "sensors": gen_sensor_series}
    targets = jobs.values() if which == "all" else [jobs[which]]
    for job in targets:
        job()
