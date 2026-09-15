"""
Retrieves relevant chunks for a question, then asks an LLM (via Groq) to
answer ONLY using those chunks -- with citations, and permission to say
"I don't know" if the context is insufficient.

Setup:
    1. Get a free API key at https://console.groq.com
    2. echo "GROQ_API_KEY=your_key_here" > .env
    3. pip install groq python-dotenv

Usage:
    python scripts/generate_answer.py "What does DataLoader's num_workers argument do?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Reuse the retrieval logic instead of duplicating it
sys.path.insert(0, str(Path(__file__).parent))
from retrieve import load_index, retrieve  # noqa: E402

load_dotenv()
GROQ_MODEL = "openai/gpt-oss-20b"
# GROQ_MODEL = "llama-3.1-70b-versatile"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question
using ONLY the provided context chunks below. Follow these rules strictly:

1. If the context contains the answer, respond clearly and cite which source(s)
   you used by their [number].
2. If the context does NOT contain enough information to answer confidently,
   say so explicitly -- do NOT guess or make up information. Say something
   like: "I couldn't find enough information in the indexed docs to answer
   this confidently."
3. Do not use any knowledge outside the provided context, even if you know
   the answer from general training -- this is a grounded-retrieval demo,
   and answers must be traceable to the retrieved sources.
"""


def build_context_block(results: list[dict]) -> str:
    """Formats retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for r in results:
        meta = r["metadata"]
        blocks.append(
            f"[{r['rank']}] Source: {meta['title']} ({meta['url']})\n{r['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, results: list[dict], client: Groq) -> str:
    context_block = build_context_block(results)

    user_prompt = f"""Context:
{context_block}

Question: {question}

Answer the question using only the context above, citing sources by [number]."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature: favor grounded, consistent answers over creativity
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/generate_answer.py "your question here"')
        sys.exit(1)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in a .env file (see script docstring).")
        sys.exit(1)

    question = sys.argv[1]

    index, chunks, metadata = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    results = retrieve(question, index, chunks, metadata, embed_model)

    client = Groq(api_key=api_key)
    answer = generate_answer(question, results, client)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}\n")
    print("--- Sources used ---")
    for r in results:
        meta = r["metadata"]
        print(f"[{r['rank']}] {meta['title']} - {meta['url']}")


if __name__ == "__main__":
    main()