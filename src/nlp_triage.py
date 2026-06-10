"""Use case #2 (NLP triage) — two-stage pipeline.

Stage 1 (deterministic): spaCy NER extracts components/symptoms/actions.
Stage 2 (LLM): classify by ATA chapter + structured extraction + criticality.
Recall-first if treated as sensitive: bias toward over-flagging identifiers.
"""
from dataclasses import dataclass, field

_NLP = None


def nlp():
    """Lazy-load the French spaCy model (fr_core_news_lg)."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("fr_core_news_lg")
    return _NLP


@dataclass
class TriagedReport:
    raw: str
    ata_chapter: str = ""
    component: str = ""
    symptom: str = ""
    action: str = ""
    criticality: str = ""           # e.g. low / medium / high
    duplicate_of: int | None = None
    entities: list[dict] = field(default_factory=list)


def triage(report: str) -> TriagedReport:
    """Run NER then LLM enrichment on one free-text entry."""
    # TODO: ents = nlp()(report); then Bedrock classification/extraction.
    raise NotImplementedError("Fill after brief: NER + LLM triage.")
