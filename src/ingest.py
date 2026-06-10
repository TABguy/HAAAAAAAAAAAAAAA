"""Use case #1 (RAG) — build the retrieval index from input/.

Two ingestion paths share one document loader (`load_documents`):
  - build_index():        pypdf -> Bedrock embeddings -> Chroma (needs AWS).
  - build_tfidf_index():  pypdf -> sklearn TF-IDF -> joblib (fully OFFLINE, no AWS).

Both read every PDF in input/ (workorders + ATA procedures) via pypdf and every
maintenance-log narrative from input/maintenance_logs.jsonl, attaching
traceability metadata (source filename / report_id + ATA chapter where known).

Run (Bedrock/Chroma):  python src/ingest.py
Run (offline TF-IDF):  python src/ingest.py --offline
"""
import json
import re
from pathlib import Path

CHROMA_COLLECTION = "haks_rag"


def _ata_from_filename(name: str) -> str:
    """Parse 'ATA<chapter>' out of a filename like procedure_ATA24.pdf -> '24'."""
    m = re.search(r"ATA[\s_-]?(\d{1,3})", name, re.IGNORECASE)
    return m.group(1) if m else ""


def _resolve_input_dir(input_dir) -> Path:
    """Resolve input_dir to an absolute path, anchored at the repo root if relative."""
    input_path = Path(input_dir)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent.parent / str(input_dir)
    return input_path


def load_documents(input_dir="input") -> list[dict]:
    """Return plain-dict documents from PDFs + log narratives.

    Each item is {"text", "ref", "ata", "doc_type"} (plus a few extra log fields).
    Backend-agnostic: both the Bedrock/Chroma and the offline TF-IDF paths build on
    top of this so they index exactly the same corpus.
    """
    from pypdf import PdfReader

    input_path = _resolve_input_dir(input_dir)
    docs: list[dict] = []

    # 1) PDFs (workorders + procedures)
    for pdf_path in sorted(input_path.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipped {pdf_path.name}: {e}")
            continue
        if not text.strip():
            print(f"  ! no extractable text in {pdf_path.name}")
            continue
        docs.append(
            {
                "text": text,
                "source": pdf_path.name,
                "ref": pdf_path.name,
                "ata": _ata_from_filename(pdf_path.name),
                "doc_type": "procedure" if "procedure" in pdf_path.name.lower() else "workorder",
            }
        )

    # 2) Maintenance-log narratives (one document per record)
    logs_path = input_path / "maintenance_logs.jsonl"
    if logs_path.exists():
        with logs_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                narrative = (rec.get("narrative") or "").strip()
                if not narrative:
                    continue
                report_id = rec.get("report_id", "")
                docs.append(
                    {
                        "text": narrative,
                        "source": report_id,
                        "ref": report_id,
                        "ata": str(rec.get("ata_chapter", "") or ""),
                        "doc_type": "maintenance_log",
                        "severity": rec.get("severity", ""),
                        "system": rec.get("system", ""),
                        "component": rec.get("component", ""),
                    }
                )

    return docs


def _to_llama_documents(doc_dicts: list[dict]):
    """Wrap plain-dict documents as llama_index Document objects (Bedrock path)."""
    from llama_index.core import Document

    documents = []
    for d in doc_dicts:
        meta = {k: v for k, v in d.items() if k != "text"}
        documents.append(Document(text=d["text"], metadata=meta))
    return documents


def build_index(input_dir: str = "input", persist_dir: str = "chroma_db") -> int:
    """Load docs -> Bedrock embeddings -> persist to local Chroma.

    Returns the number of nodes (chunks) indexed.
    Raises RuntimeError with a clear message if EMBED_MODEL_ID is not configured.
    """
    from src.common import AWS_REGION, EMBED_MODEL_ID

    if not EMBED_MODEL_ID:
        raise RuntimeError(
            "EMBED_MODEL_ID is not set. Add EMBED_MODEL_ID=<bedrock-embeddings-model-id> "
            "to your .env (region AWS_REGION) before running ingest."
        )

    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.bedrock import BedrockEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    input_path = _resolve_input_dir(input_dir)
    if not input_path.exists():
        raise RuntimeError(f"Input directory not found: {input_path}. Run src/gen_data.py first.")

    doc_dicts = load_documents(input_path)
    if not doc_dicts:
        raise RuntimeError(f"No documents found in {input_path} (no PDFs or log narratives).")
    print(f"Loaded {len(doc_dicts)} source documents from {input_path}")
    docs = _to_llama_documents(doc_dicts)

    embed_model = BedrockEmbedding(model_name=EMBED_MODEL_ID, region_name=AWS_REGION)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    persist_path = Path(persist_dir)
    if not persist_path.is_absolute():
        persist_path = Path(__file__).resolve().parent.parent / persist_dir
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    nodes = splitter.get_nodes_from_documents(docs)
    print(f"Split into {len(nodes)} nodes; embedding + indexing...")

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print(f"Indexed {len(nodes)} nodes into {persist_path} (collection '{CHROMA_COLLECTION}')")
    return len(nodes)


TFIDF_FILENAME = "tfidf.joblib"


def _resolve_persist_dir(persist_dir) -> Path:
    """Resolve persist_dir to an absolute path, anchored at the repo root if relative."""
    persist_path = Path(persist_dir)
    if not persist_path.is_absolute():
        persist_path = Path(__file__).resolve().parent.parent / str(persist_dir)
    return persist_path


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into ~chunk_size-char chunks with character overlap (no AWS deps)."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def build_tfidf_index(input_dir="input", persist_dir="tfidf_index") -> int:
    """Load docs -> chunk -> sklearn TF-IDF -> persist via joblib. NO AWS needed.

    Persists {vectorizer, matrix, chunks} to persist_dir/tfidf.joblib where each
    chunk is {"text", "ref", "ata"}. Returns the number of chunks indexed.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    input_path = _resolve_input_dir(input_dir)
    if not input_path.exists():
        raise RuntimeError(f"Input directory not found: {input_path}. Run src/gen_data.py first.")

    doc_dicts = load_documents(input_path)
    if not doc_dicts:
        raise RuntimeError(f"No documents found in {input_path} (no PDFs or log narratives).")
    print(f"Loaded {len(doc_dicts)} source documents from {input_path}")

    chunks: list[dict] = []
    for d in doc_dicts:
        for piece in _chunk_text(d["text"]):
            chunks.append({"text": piece, "ref": d.get("ref", ""), "ata": d.get("ata", "")})

    if not chunks:
        raise RuntimeError("No non-empty chunks produced from the corpus.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(c["text"] for c in chunks)

    persist_path = _resolve_persist_dir(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks},
        persist_path / TFIDF_FILENAME,
    )

    print(f"Indexed {len(chunks)} chunks into {persist_path / TFIDF_FILENAME} (TF-IDF, offline)")
    return len(chunks)


def tfidf_exists(persist_dir="tfidf_index") -> bool:
    """True if a persisted TF-IDF index file exists."""
    return (_resolve_persist_dir(persist_dir) / TFIDF_FILENAME).exists()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the RAG retrieval index.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Build the offline TF-IDF index (no AWS) instead of the Bedrock/Chroma index.",
    )
    args = parser.parse_args()

    if args.offline:
        count = build_tfidf_index()
        print(f"Done. {count} chunks indexed (offline TF-IDF).")
    else:
        count = build_index()
        print(f"Done. {count} nodes indexed.")
