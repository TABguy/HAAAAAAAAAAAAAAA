"""Use case #1 (RAG) — retrieval + Bedrock LLM. Returns answer + cited sources.

Traceability sells: always return the source procedure ref / ATA chapter.
"""
from dataclasses import dataclass, field


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)  # [{ref, ata, snippet, score}]


def answer_question(question: str) -> RagAnswer:
    """Retrieve from Chroma, synthesize via Bedrock, attach citations."""
    # TODO: load persisted index, query_engine with source nodes, map to citations.
    raise NotImplementedError("Fill after brief: RAG query + citations.")
