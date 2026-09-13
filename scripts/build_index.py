"""
Chunks the scraped markdown docs, embeds them, and builds a FAISS index.

Usage:
    python scripts/build_index.py
"""

import json
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for MVP
CHUNK_SIZE = 1500      # approx characters per chunk (not words -- see chunk_text)
CHUNK_OVERLAP = 200    # approx characters of overlap between consecutive chunks

# Tries paragraph breaks first, then sentences, then words -- falls back to a
# hard character cut only as a last resort. This avoids slicing mid-sentence
# or mid-code-block the way naive whitespace splitting would.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_metadata_for(md_path: Path) -> dict:
    """Try a couple of common sidecar naming conventions.

    NOTE: uses plain string slicing, not Path.with_suffix(), because
    slugs like "torch.nn.Conv2d" contain multiple dots -- with_suffix()
    only strips text after the LAST dot, which would mangle these names.
    """
    base_name = md_path.name[:-len(md_path.suffix)]  # strip trailing ".md" only
    for suffix in [".meta.json", ".metadata.json"]:
        candidate = md_path.parent / f"{base_name}{suffix}"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    # Fallback: no sidecar found, use filename as title
    return {"url": "", "title": base_name, "slug": base_name}


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return _splitter.split_text(text)


def build():
    md_files = sorted(RAW_DIR.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {RAW_DIR}. Run scrape_docs.py first.")
        return

    print(f"Found {len(md_files)} doc files.")

    all_chunks = []       # list of chunk text
    all_chunk_meta = []    # list of dicts: {url, title, slug, chunk_index}

    for md_path in md_files:
        meta = load_metadata_for(md_path)
        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"  {md_path.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_chunk_meta.append({
                "url": meta.get("url", ""),
                "title": meta.get("title", md_path.stem),
                "slug": meta.get("slug", md_path.stem),
                "chunk_index": i,
                "source_file": md_path.name,
            })

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks (this may take a minute)...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_chunk_meta}, f)

    print(f"\nSaved FAISS index + metadata to {INDEX_DIR}")
    print(f"  faiss.index  ({index.ntotal} vectors, dim={dim})")
    print(f"  chunks.pkl   (chunk text + metadata)")


if __name__ == "__main__":
    build()
