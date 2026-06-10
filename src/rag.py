"""Use case #1 (RAG) — retrieval + answer synthesis. Returns answer + cited sources.

Two backends, auto-selected by answer_question():
  - "bedrock":  Chroma retrieval + Bedrock LLM synthesis (needs AWS).
  - "offline":  sklearn TF-IDF retrieval + extractive answer (NO AWS) — de-risks the demo.

Traceability sells: always return the source procedure ref / ATA chapter.
"""
from dataclasses import dataclass, field
from pathlib import Path

from src.ingest import CHROMA_COLLECTION, TFIDF_FILENAME


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)  # [{ref, ata, snippet, score}]
    mode: str = ""  # "bedrock" or "offline"


def _resolve_persist_dir(persist_dir: str) -> Path:
    path = Path(persist_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / persist_dir
    return path


def _chroma_index_exists(persist_dir: str = "chroma_db") -> bool:
    """True if a persisted Chroma index appears to exist."""
    path = _resolve_persist_dir(persist_dir)
    if not path.exists():
        return False
    # chromadb's PersistentClient writes chroma.sqlite3 at the root of the dir.
    return (path / "chroma.sqlite3").exists() or any(path.iterdir())


def _tfidf_index_exists(persist_dir: str = "tfidf_index") -> bool:
    """True if a persisted offline TF-IDF index file exists."""
    return (_resolve_persist_dir(persist_dir) / TFIDF_FILENAME).exists()


def index_exists(persist_dir: str = "chroma_db", tfidf_dir: str = "tfidf_index") -> bool:
    """True if EITHER the Chroma index or the offline TF-IDF index exists.

    The UI can call this to show "ready" once either backend has been built.
    """
    return _chroma_index_exists(persist_dir) or _tfidf_index_exists(tfidf_dir)


# --------------------------------------------------------------------------- #
# Offline backend (TF-IDF, no AWS)
# --------------------------------------------------------------------------- #
def retrieve_offline(question: str, persist_dir: str = "tfidf_index", top_k: int = 4) -> list[dict]:
    """Load the joblib TF-IDF index, rank chunks by cosine similarity to the query.

    Returns top_k [{ref, ata, snippet, score}]. No AWS.
    """
    import joblib
    from sklearn.metrics.pairwise import linear_kernel

    index_path = _resolve_persist_dir(persist_dir) / TFIDF_FILENAME
    if not index_path.exists():
        raise RuntimeError(
            f"No TF-IDF index at {index_path}. Run `python src/ingest.py --offline` first."
        )

    store = joblib.load(index_path)
    vectorizer, matrix, chunks = store["vectorizer"], store["matrix"], store["chunks"]

    query_vec = vectorizer.transform([question])
    # TF-IDF rows are L2-normalized, so linear_kernel == cosine similarity here.
    scores = linear_kernel(query_vec, matrix).ravel()

    k = min(top_k, len(chunks))
    top_idx = scores.argsort()[::-1][:k]

    results = []
    for i in top_idx:
        c = chunks[i]
        text = c.get("text", "")
        results.append(
            {
                "ref": c.get("ref", ""),
                "ata": c.get("ata", ""),
                "snippet": text[:240],
                "score": float(scores[i]),
            }
        )
    return results


def answer_offline(question: str, persist_dir: str = "tfidf_index", top_k: int = 4) -> RagAnswer:
    """Extractive RAG answer from the offline TF-IDF index. No AWS.

    Builds a readable, cited answer: a lead sentence + bulleted source excerpts,
    each tagged with its ref / ATA chapter.
    """
    sources = retrieve_offline(question, persist_dir=persist_dir, top_k=top_k)

    relevant = [s for s in sources if s["score"] > 0.0]
    if not relevant:
        return RagAnswer(
            answer=(
                f"No matching content was found in the indexed corpus for: \"{question}\". "
                "Try rephrasing, or check that ingestion ran (`python src/ingest.py --offline`)."
            ),
            sources=sources,
            mode="offline",
        )

    lines = [
        f"Based on {len(relevant)} matching passage(s) in the maintenance corpus "
        f'(offline TF-IDF retrieval) for: "{question}"',
        "",
    ]
    for s in relevant:
        ata = f"ATA {s['ata']}" if s["ata"] else "ATA n/a"
        excerpt = " ".join(s["snippet"].split())
        lines.append(f"- [{s['ref']} | {ata}] {excerpt}")

    lines.append("")
    lines.append(
        "Sources above are extractive excerpts (verbatim from the cited documents); "
        "no generative model was used."
    )

    return RagAnswer(answer="\n".join(lines), sources=sources, mode="offline")


# --------------------------------------------------------------------------- #
# Bedrock backend (Chroma + LLM, needs AWS)
# --------------------------------------------------------------------------- #
def answer_bedrock(question: str, persist_dir: str = "chroma_db", top_k: int = 4) -> RagAnswer:
    """Retrieve from Chroma, synthesize via Bedrock, attach citations. Needs AWS."""
    from src.common import AWS_REGION, BEDROCK_MODEL_ID

    if not BEDROCK_MODEL_ID:
        raise RuntimeError(
            "BEDROCK_MODEL_ID is not set. Add BEDROCK_MODEL_ID=<bedrock-llm-model-id> "
            "to your .env before querying."
        )

    persist_path = _resolve_persist_dir(persist_dir)
    if not _chroma_index_exists(persist_dir):
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

    return RagAnswer(answer=str(response), sources=sources, mode="bedrock")


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def answer_question(question: str, prefer: str = "auto", top_k: int = 4) -> RagAnswer:
    """Answer a question, auto-selecting the best available backend.

    prefer="auto"    -> Bedrock if BEDROCK_MODEL_ID is set AND a Chroma index exists;
                        else offline if a TF-IDF index exists; else a clear RuntimeError.
    prefer="offline" -> force the offline TF-IDF path.
    prefer="bedrock" -> force the Bedrock/Chroma path (raises a clean guard if unavailable).
    """
    if prefer == "offline":
        return answer_offline(question, top_k=top_k)
    if prefer == "bedrock":
        return answer_bedrock(question, top_k=top_k)
    if prefer != "auto":
        raise ValueError(f"Unknown prefer={prefer!r}; expected 'auto', 'offline' or 'bedrock'.")

    from src.common import BEDROCK_MODEL_ID

    if BEDROCK_MODEL_ID and _chroma_index_exists():
        return answer_bedrock(question, top_k=top_k)
    if _tfidf_index_exists():
        return answer_offline(question, top_k=top_k)

    raise RuntimeError(
        "No usable RAG backend found. Either:\n"
        "  - build the offline index:  python src/ingest.py --offline   (no AWS), or\n"
        "  - set BEDROCK_MODEL_ID + EMBED_MODEL_ID in .env and run:  python src/ingest.py"
    )
