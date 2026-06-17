# Auditly

Auditly is a local-first Streamlit workspace for audit document question answering. Upload an audit PDF, ask natural-language questions, and get short answers with page-level evidence from the source document.

The project is designed for demos and portfolio review: it includes a built-in sample audit report, an offline smoke test, evidence cards, and exportable answer memos.

## What It Shows

- Local document processing: PDF text extraction happens on the user's machine.
- Evidence-grounded answers: responses are assembled from retrieved document chunks and shown with page numbers.
- Hybrid retrieval for uploaded PDFs: multilingual E5 embeddings plus BM25 keyword search.
- Zero-API demo path: the bundled sample report uses BM25 only, so it does not need a model download.
- Reviewer-ready workflow: one command to run the app and one smoke test to validate the sample path.

## Quick Start

Use Python 3.10, 3.11, or 3.12. Some ML dependencies may not publish wheels for newer Python versions immediately.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If you use `uv`, this is the shortest setup:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```

Open the local URL that Streamlit prints, usually `http://localhost:8501`.

For a no-PDF demo, click `Load sample document (demo)`, choose one of the preset questions, and run `Analyze`.

## Smoke Test

Validate the offline sample workflow:

```bash
python3 test_sample_load.py
```

Optionally validate the embedding model and FAISS pipeline:

```bash
python3 test_sample_load.py --with-model
```

The optional model test may download `intfloat/multilingual-e5-small` the first time it runs. After the model is cached, uploaded-PDF search can run locally.

## Deployment Notes

Use the GitHub repository for the showcase: <https://github.com/fizuli20/Auditly>. It makes the project easy to review, clone, and connect to deployment platforms.

For the current Streamlit app, prefer a Python app host such as Streamlit Community Cloud, Render, Railway, Fly.io, or a small Docker host. Vercel is excellent for frontend and serverless apps, but this project is a long-running Streamlit workspace with heavier ML dependencies. Deploy to Vercel only after refactoring the product into a supported web frontend plus a separate ASGI/WSGI API, or use Vercel for a marketing/demo page that links to the hosted Streamlit app.

## How It Works

1. PDF text is extracted with `pdfplumber`.
2. Text is split into page-preserving chunks.
3. Demo mode builds a BM25 index over the built-in sample report.
4. Uploaded PDFs use multilingual E5 embeddings, FAISS vector search, and BM25 keyword scoring.
5. The app creates an extractive answer from the highest-scoring chunks and displays the evidence trail.

## Project Structure

```text
app.py              Streamlit application and retrieval pipeline
test_sample_load.py Offline smoke test with optional model validation
requirements.txt    Runtime dependencies
pyproject.toml      Project metadata and supported Python range
.python-version     Suggested Python version for local tools
.streamlit/         Local-only Streamlit runtime config
RUN_AND_DEMO.md     Short demo script
```

## Limitations

- Scanned PDFs without embedded text are not supported yet. OCR would be the next major upgrade.
- The app is an evidence assistant, not a compliance decision engine.
- Uploaded-PDF semantic search needs the E5 model available locally. The first uncached run may need internet access.
- Answers are extractive and intentionally conservative; they do not synthesize beyond retrieved text.

## Roadmap

- Add OCR for scanned audit reports.
- Add a small evaluation set with expected evidence pages.
- Package the app with Docker for one-command review.
- Add document history with local-only encrypted storage.
- Add richer exports for audit workpapers and review notes.
