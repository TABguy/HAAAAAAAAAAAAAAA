"""Use case #1 (RAG) — retrieval + Bedrock LLM. Returns answer + cited sources.

Traceability sells: always return the source procedure ref / ATA chapter.
"""
from dataclasses import dataclass, field
from pathlib import Path

from src.ingest import CHROMA_COLLECTION


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)  # [{ref, ata, snippet, score}]


def _resolve_persist_dir(persist_dir: str) -> Path:
    path = Path(persist_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / persist_dir
    return path


def index_exists(persist_dir: str = "chroma_db") -> bool:
    """True if a persisted Chroma index appears to exist (app.py can call this)."""
    path = _resolve_persist_dir(persist_dir)
    if not path.exists():
        return False
    # chromadb's PersistentClient writes chroma.sqlite3 at the root of the dir.
    return (path / "chroma.sqlite3").exists() or any(path.iterdir())


def answer_question(question: str, persist_dir: str = "chroma_db", top_k: int = 4) -> RagAnswer:
    """Retrieve from Chroma, synthesize via Bedrock, attach citations."""
    from src.common import AWS_REGION, BEDROCK_MODEL_ID

    if not BEDROCK_MODEL_ID:
        raise RuntimeError(
            "BEDROCK_MODEL_ID is not set. Add BEDROCK_MODEL_ID=<bedrock-llm-model-id> "
            "to your .env before querying."
        )

    persist_path = _resolve_persist_dir(persist_dir)
    if not index_exists(persist_dir):
        raise RuntimeError(
            f"No Chroma index at {persist_path}. Run `python src/ingest.py` first."
        )

    from src.common import EMBED_MODEL_ID

    if not EMBED_MODEL_ID:
        raise RuntimeError(
            "EMBED_MODEL_ID is not set. The same embeddings model used at ingest time "
            "is required at query time. Add EMBED_MODEL_ID=<...> to your .env."
        )

    import chromadb
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.bedrock import BedrockEmbedding
    from llama_index.llms.bedrock_converse import BedrockConverse
    from llama_index.vector_stores.chroma import ChromaVectorStore

    embed_model = BedrockEmbedding(model_name=EMBED_MODEL_ID, region_name=AWS_REGION)
    llm = BedrockConverse(model=BEDROCK_MODEL_ID, region_name=AWS_REGION)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k)

    response = query_engine.query(question)

    sources = []
    for sn in getattr(response, "source_nodes", []) or []:
        meta = sn.node.metadata or {}
        text = sn.node.get_content() or ""
        sources.append(
            {
                "ref": meta.get("ref") or meta.get("source") or "",
                "ata": meta.get("ata", ""),
                "snippet": text[:200],
                "score": sn.score,
            }
        )

    return RagAnswer(answer=str(response), sources=sources)
