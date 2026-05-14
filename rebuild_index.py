"""Wipe and rebuild the ChromaDB vector store from scratch."""
import sys, json, pathlib
sys.path.insert(0, "src")

from embeddings import build_chunks
from vector_store import MSADSVectorStore

docs = json.loads(pathlib.Path("data/knowledge_base.json").read_text())
chunks = build_chunks(docs)
pathlib.Path("data/chunks.json").write_text(json.dumps(chunks, indent=2))
print(f"Created {len(chunks)} chunks from {len(docs)} pages.")

store = MSADSVectorStore()
store.reset()
store.build_from_chunks("data/chunks.json")
print("Index rebuilt.")
