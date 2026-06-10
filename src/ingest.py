"""Use case #1 (RAG) — build the Chroma vector index from data/.

Run:  python src/ingest.py
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def build_index():
    """Load docs -> Bedrock embeddings -> persist to local Chroma.

    Sketch (wire after brief):
        from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
        from llama_index.embeddings.bedrock import BedrockEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
        ...
    """
    raise NotImplementedError("Fill after brief: ingest data/ into Chroma.")


if __name__ == "__main__":
    build_index()
