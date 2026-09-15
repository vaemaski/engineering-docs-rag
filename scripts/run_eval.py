"""
Runs the eval set against retrieval (and optionally generation) and reports:
  - Recall@k for in-scope questions (did the expected source appear in top-k?)
  - Correct refusal rate for out-of-scope questions (best-match distance above
    a "confident" threshold implies the system should decline to answer)

Usage:
    python scripts/run_eval.py                      # retrieval-only eval
    python scripts/run_eval.py --with-generation     # also calls the LLM and
                                                       # prints answers for
                                                       # manual/LLM-judge review
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import load_index, retrieve  # noqa: E402

load_dotenv()

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval_set.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# Distance above this is treated as "not a confident match" -- used to check
# whether out-of-scope questions correctly fail to retrieve anything relevant.
# NOTE: this threshold was picked by eyeballing distances from earlier manual
# testing (in-scope matches landed ~0.68-0.86). Recalibrate if your corpus
# or embedding model changes.
CONFIDENCE_THRESHOLD = 1.0


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(eval_items, index, chunks, metadata, model):
    results_log = []
    correct = 0
    total_in_scope = 0
    correct_refusals = 0
    total_out_of_scope = 0

    for item in eval_items:
        question = item["question"]
        expected_source = item.get("expected_source")
        category = item.get("category", "simple_fact")

        retrieved = retrieve(question, index, chunks, metadata, model, top_k=TOP_K)
        retrieved_sources = [r["metadata"]["source_file"] for r in retrieved]
        best_distance = retrieved[0]["distance"] if retrieved else float("inf")

        if expected_source is None:
            # Out-of-scope question: "pass" means nothing came back confidently
            total_out_of_scope += 1
            is_confident = best_distance < CONFIDENCE_THRESHOLD
            passed = not is_confident
            if passed:
                correct_refusals += 1
            results_log.append({
                "question": question, "category": category,
                "expected_source": None, "retrieved_top": retrieved_sources[0] if retrieved_sources else None,
                "best_distance": round(best_distance, 3), "passed": passed,
            })
        else:
            # In-scope question: "pass" means expected_source is in top-k
            total_in_scope += 1
            hit = expected_source in retrieved_sources
            if hit:
                correct += 1
            results_log.append({
                "question": question, "category": category,
                "expected_source": expected_source, "retrieved_sources": retrieved_sources,
                "best_distance": round(best_distance, 3), "passed": hit,
            })

    return {
        "recall_at_k": correct / total_in_scope if total_in_scope else None,
        "correct_refusal_rate": correct_refusals / total_out_of_scope if total_out_of_scope else None,
        "total_in_scope": total_in_scope,
        "total_out_of_scope": total_out_of_scope,
        "details": results_log,
    }


def run_generation_pass(eval_items, index, chunks, metadata, embed_model):
    from generate_answer import generate_answer
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY not set -- skipping generation pass.\n")
        return
    client = Groq(api_key=api_key)

    print("\n" + "=" * 70)
    print("GENERATION PASS (for manual / LLM-judge review)")
    print("=" * 70)
    for item in eval_items:
        question = item["question"]
        retrieved = retrieve(question, index, chunks, metadata, embed_model, top_k=TOP_K)
        answer = generate_answer(question, retrieved, client)
        print(f"\nQ: {question}")
        print(f"Category: {item.get('category', 'simple_fact')}")
        if item.get("expected_answer"):
            print(f"Expected: {item['expected_answer']}")
        print(f"Got: {answer}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generation", action="store_true",
                         help="Also run generation and print answers for manual review")
    args = parser.parse_args()

    eval_items = load_eval_set()
    index, chunks, metadata = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    report = evaluate_retrieval(eval_items, index, chunks, metadata, embed_model)

    print("=" * 70)
    print("RETRIEVAL EVAL RESULTS")
    print("=" * 70)
    print(f"Recall@{TOP_K} (in-scope, n={report['total_in_scope']}): "
          f"{report['recall_at_k']:.1%}" if report["recall_at_k"] is not None else "N/A")
    print(f"Correct refusal rate (out-of-scope, n={report['total_out_of_scope']}): "
          f"{report['correct_refusal_rate']:.1%}" if report["correct_refusal_rate"] is not None else "N/A")
    print()

    for r in report["details"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] ({r['category']}) {r['question']}")
        if r["expected_source"]:
            print(f"       expected={r['expected_source']} | retrieved_top5={r.get('retrieved_sources')} | dist={r['best_distance']}")
        else:
            print(f"       retrieved_top={r.get('retrieved_top')} | dist={r['best_distance']}")

    if args.with_generation:
        run_generation_pass(eval_items, index, chunks, metadata, embed_model)


if __name__ == "__main__":
    main()
