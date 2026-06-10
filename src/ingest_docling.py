"""Use case #1 (RAG) — OPTIONAL Docling ingest path (the IBM angle).

Same metadata / return contract as src/ingest.py, but PDFs are parsed with
Docling for layout-aware extraction (tables, headings, reading order) instead
of plain pypdf. Docling is a heavy optional dependency (pulls torch + CUDA),
so the import is guarded and only happens inside build_index_docling().

Install on a capable machine:  pip install -r requirements-docling.txt
Run:  python src/ingest_docling.py
"""
import json
from pathlib import Path

from src.ingest import CHROMA_COLLECTION, _ata_from_filename


def _require_docling():
    """Import Docling or fail loudly with install guidance."""
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Docling is not installed (this is the optional layout-aware ingest path). "
            "Run `pip install -r requirements-docling.txt` on a machine with enough disk "
            "(it pulls torch + CUDA). Original import error: "
            f"{type(e).__name__}: {e}"
        ) from e
    from docling.document_converter import DocumentConverter

    return DocumentConverter


def _load_documents_docling(input_dir: Path):
    """Return llama_index Documents: Docling-parsed PDFs + log narratives."""
    from llama_index.core import Document

    DocumentConverter = _require_docling()
    converter = DocumentConverter()

    docs = []

    # 1) PDFs parsed via Docling (export to markdown to preserve structure)
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        try:
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipped {pdf_path.name}: {e}")
            continue
        if not text.strip():
            print(f"  ! no extractable text in {pdf_path.name}")
            continue
        docs.append(
            Document(
                text=text,
                metadata={
                    "source": pdf_path.name,
                    "ref": pdf_path.name,
                    "ata": _ata_from_filename(pdf_path.name),
                    "doc_type": "procedure" if "procedure" in pdf_path.name.lower() else "workorder",
                    "parser": "docling",
                },
            )
        )

    # 2) Maintenance-log narratives (identical to the pypdf path)
    logs_path = input_dir / "maintenance_logs.jsonl"
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
                    Document(
                        text=narrative,
                        metadata={
                            "source": report_id,
                            "ref": report_id,
                            "ata": str(rec.get("ata_chapter", "") or ""),
                            "doc_type": "maintenance_log",
                            "severity": rec.get("severity", ""),
                            "system": rec.get("system", ""),
                            "component": rec.get("component", ""),
                            "parser": "docling",
                        },
                    )
                )

    return docs


def build_index_docling(input_dir: str = "input", persist_dir: str = "chroma_db") -> int:
    """Docling-based variant of ingest.build_index. Returns nodes indexed.

    Raises RuntimeError if EMBED_MODEL_ID is unset or Docling is not installed.
    """
    from src.common import AWS_REGION, EMBED_MODEL_ID

    if not EMBED_MODEL_ID:
        raise RuntimeError(
            "EMBED_MODEL_ID is not set. Add EMBED_MODEL_ID=<bedrock-embeddings-model-id> "
            "to your .env before running ingest."
        )

    # Fail fast on missing Docling before any embedding work.
    _require_docling()

    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.bedrock import BedrockEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    input_path = Path(input_dir)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent.parent / input_dir
    if not input_path.exists():
        raise RuntimeError(f"Input directory not found: {input_path}. Run src/gen_data.py first.")

    docs = _load_documents_docling(input_path)
    if not docs:
        raise RuntimeError(f"No documents found in {input_path} (no PDFs or log narratives).")
    print(f"Loaded {len(docs)} source documents (Docling) from {input_path}")

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

    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    print(f"Indexed {len(nodes)} nodes into {persist_path} (collection '{CHROMA_COLLECTION}')")
    return len(nodes)


if __name__ == "__main__":
    count = build_index_docling()
    print(f"Done. {count} nodes indexed (Docling).")
