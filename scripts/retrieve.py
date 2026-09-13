"""
Given a question, embeds it and searches the FAISS index for the closest chunks.

Usage:
    python scripts/retrieve.py "What does DataLoader's num_workers argument do?"
"""

import pickle
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # MUST match the model used in build_index.py --
                                        # embeddings from different models aren't comparable
TOP_K = 5


def load_index():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadata"]


def retrieve(question: str, index, chunks, metadata, model, top_k: int = TOP_K):
    query_vec = model.encode([question], convert_to_numpy=True)

    # index.search returns (distances, indices) -- both shape (num_queries, top_k)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:  # FAISS returns -1 if fewer than top_k results exist
            continue
        results.append({
            "rank": rank + 1,
            "distance": float(dist),  # lower = more similar (L2 distance)
            "text": chunks[idx],
            "metadata": metadata[idx],
        })
    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/retrieve.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    index, chunks, metadata = load_index()
    model = SentenceTransformer(EMBED_MODEL_NAME)  # load once, reuse across calls

    print(f"Question: {question}\n")
    results = retrieve(question, index, chunks, metadata, model)

    for r in results:
        meta = r["metadata"]
        print(f"--- Rank {r['rank']} (distance={r['distance']:.3f}) ---")
        print(f"Source: {meta['title']} ({meta['url']})")
        print(f"{r['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()