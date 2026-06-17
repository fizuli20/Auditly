#!/usr/bin/env python3
"""
Offline smoke test for Auditly.

Default behavior validates the sample document path without network access or
model downloads. Pass --with-model to also test sentence-transformers + FAISS.
"""

import argparse
import os
import socket
import sys
import time


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
socket.setdefaulttimeout(30)


def require_python_version():
    print("[0/4] Checking Python version")
    version = sys.version_info
    if version < (3, 10) or version >= (3, 13):
        raise RuntimeError(
            "Use Python 3.10, 3.11, or 3.12 for this project. "
            f"Current version is {version.major}.{version.minor}.{version.micro}."
        )
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")


def require_imports():
    print("[1/4] Checking dependencies")
    try:
        import numpy  # noqa: F401
        import faiss  # noqa: F401
        import pdfplumber  # noqa: F401
        import streamlit  # noqa: F401
        from rank_bm25 import BM25Okapi  # noqa: F401
    except ImportError as exc:
        print(f"FAILED: missing dependency: {exc}")
        print("Install dependencies with: pip install -r requirements.txt")
        raise
    print("OK: required packages import")


def test_sample_pipeline():
    print("[2/4] Testing built-in sample pipeline")
    from rank_bm25 import BM25Okapi

    from app import SAMPLE_AUDIT_PAGES, chunk_text, demo_search, generate_answer

    chunks = chunk_text(SAMPLE_AUDIT_PAGES)
    if not chunks:
        raise AssertionError("sample document produced no chunks")

    tokenized_corpus = [text.lower().split() for text, _ in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    retrieved = demo_search("List audit findings", chunks, bm25)
    answer, evidence, has_evidence = generate_answer("List audit findings", retrieved)

    if not has_evidence:
        raise AssertionError("sample query did not produce evidence")
    if not evidence:
        raise AssertionError("sample query returned no evidence excerpts")
    if "Finding" not in answer and "finding" not in answer:
        raise AssertionError("answer does not appear to reference audit findings")

    print(f"OK: sample produced {len(chunks)} chunks and an evidence-backed answer")


def test_model_pipeline():
    print("[3/4] Testing embedding model and FAISS index")
    from sentence_transformers import SentenceTransformer

    from app import PASSAGE_PREFIX, SAMPLE_AUDIT_PAGES, build_faiss_index, chunk_text

    start = time.time()
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    chunks = chunk_text(SAMPLE_AUDIT_PAGES)
    texts = [PASSAGE_PREFIX + text for text, _ in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    index = build_faiss_index(embeddings)

    if index.ntotal != len(chunks):
        raise AssertionError("FAISS index size does not match chunk count")

    print(f"OK: model + FAISS pipeline passed in {time.time() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Also load intfloat/multilingual-e5-small. This may download the model on first run.",
    )
    args = parser.parse_args()

    print("AUDITLY - SMOKE TEST")
    print("=" * 36)

    try:
        require_python_version()
        require_imports()
        test_sample_pipeline()
        if args.with_model:
            test_model_pipeline()
        else:
            print("[3/4] Skipping model test. Use --with-model to enable it.")
        print("[4/4] Complete")
    except Exception as exc:
        print(f"\nSmoke test failed: {exc}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
