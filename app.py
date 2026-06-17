import html
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
import pdfplumber
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

# Network timeout configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent parallel tokenizer processes
os.environ["HF_DATASETS_OFFLINE"] = "0"  # Allow HF downloads but with timeout
socket.setdefaulttimeout(30)  # Global socket timeout

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "Auditly"
APP_TAGLINE = "Audit evidence workspace"
EXPORT_FILENAME = "auditly-answer-memo.md"
CHUNK_MIN_TOKENS = 500
CHUNK_MAX_TOKENS = 800
TOP_K = 5
HYBRID_VECTOR_WEIGHT = 0.5
HYBRID_BM25_WEIGHT = 0.5
CONFIDENCE_THRESHOLD = 0.15  # Below this, return safe refusal

# E5 prefix for multilingual model (query vs passage)
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

DOCUMENT_STATE_KEYS = [
    "pdf_path", "chunks", "embeddings", "faiss_index", "processed", "ephemeral",
    "demo_mode", "bm25", "model", "uploaded_signature", "search_mode",
]
RESULT_STATE_KEYS = [
    "last_answer", "last_evidence", "last_has_evidence", "last_retrieved", "last_query",
]

PRESET_QUERIES = [
    "List audit findings",
    "Where are internal control weaknesses mentioned?",
    "What compliance issues were identified?",
    "What financial risks are discussed?",
]

# Sample audit document for demo (no PDF upload required). Text is long enough to produce chunks.
SAMPLE_AUDIT_PAGES: List[Tuple[int, str]] = [
    (1, """
    Internal Audit Report – Annual Financial Controls. Fiscal Year 2024 | Confidential.

    Executive Summary. This report presents the results of our internal audit of financial
    controls and compliance processes. We identified several audit findings that require
    management attention. The audit focused on procurement, segregation of duties, and
    compliance with company policies. Scope of work included testing of transaction samples,
    review of authorization matrices, and interviews with process owners. The audit was
    conducted in accordance with internal audit standards. Management has provided
    responses to the findings which are reflected in this report. We recommend that the
    audit committee review the findings and monitor implementation of corrective actions.
    """ * 3),  # Repeat to reach chunk size
    (2, """
    Audit Findings.

    Finding 1 – Procurement approval process. The procurement approval process lacks
    segregation of duties. A single individual can both initiate and approve purchase orders
    up to $50,000. This creates a material weakness in internal controls. We recommend
    implementing dual approval for all orders above $10,000. The control owner has agreed
    to implement this by end of Q2. Finding 2 – Access controls. User access reviews were
    not performed quarterly as required by policy. Several terminated employees retained
    system access for more than 48 hours. This compliance issue was identified in the IT
    general controls review. IT has committed to automating access certification and
    implementing a 24-hour deprovisioning SLA. Additional findings related to change
    management and backup testing were reported separately to IT management.
    """ * 3),
    (3, """
    Internal Control Weaknesses.

    Material weakness identified in procurement controls. The procurement approval process
    lacks segregation of duties and has not been updated to reflect the current authorization
    matrix. Management has acknowledged this weakness and plans to implement a two-step
    approval workflow by Q2. Additional control weaknesses were noted in the revenue
    recognition process. Manual journal entries are not consistently reviewed by a second
    party, which may lead to financial misstatement risk. We recommend that all manual
    entries above a threshold be reviewed and approved by the controller. The company
    has documented its remediation plan and will report progress to the audit committee.
    """ * 3),
    (4, """
    Compliance Issues Identified.

    The following compliance issues were identified during the audit: Late filing of vendor
    tax documentation (W-9 forms) in 12% of sampled contracts. Expense reimbursement
    policies were not consistently applied; three instances of non-compliant reimbursements
    were noted. The code of conduct acknowledgment was missing from personnel files for
    5% of staff. These compliance issues should be remediated and monitored by the
    compliance officer. Legal and HR have been engaged to update procedures and conduct
    training. A follow-up review is planned for next quarter to verify remediation.
    """ * 3),
    (5, """
    Financial Risks Discussed.

    Financial risks discussed in this audit include: Currency exposure on overseas
    contracts – the company has no hedging program in place. Concentration risk with
    the top three customers representing 60% of revenue. Credit risk on receivables
    past 90 days, which have increased by 15% year-over-year. Management has been
    advised to document risk acceptance or mitigation for each of these areas. Treasury
    will present hedging options to the board. Sales has been asked to diversify the
    customer base and credit has tightened payment terms for at-risk accounts.
    """ * 3),
]


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_path: str, progress=None, status=None) -> List[Tuple[int, str]]:
    """
    Extract text from PDF with page numbers using pdfplumber.
    Returns list of (page_number, text) tuples.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages) or 1
        for i, page in enumerate(pdf.pages, start=1):
            if status and (i == 1 or i % 10 == 0 or i == total_pages):
                status.info(f"Extracting text from page {i} of {total_pages}...")
            if progress:
                progress.progress(min(i / total_pages * 0.45, 0.45))
            text = page.extract_text()
            if text and text.strip():
                pages_text.append((i, text.strip()))
    return pages_text


# ---------------------------------------------------------------------------
# Chunking (500–800 tokens using word heuristic: ~1.3 words per token)
# ---------------------------------------------------------------------------
def _token_approx(text: str) -> int:
    """Approximate token count (words + punctuation)."""
    return len(text.split()) + len(re.findall(r'[^\w\s]', text))


def chunk_text(pages_text: List[Tuple[int, str]]) -> List[Tuple[str, int]]:
    """
    Chunk text into 500–800 token segments. Preserves page number for each chunk.
    Returns list of (chunk_text, page_number).
    """
    chunks = []
    for page_num, text in pages_text:
        words = text.split()
        start = 0
        while start < len(words):
            # Take a segment aiming for CHUNK_MAX_TOKENS
            segment = []
            tok_count = 0
            i = start
            while i < len(words) and tok_count < CHUNK_MAX_TOKENS:
                segment.append(words[i])
                tok_count = _token_approx(" ".join(segment))
                i += 1
            chunk_str = " ".join(segment).strip()
            if chunk_str and _token_approx(chunk_str) >= CHUNK_MIN_TOKENS // 2:
                chunks.append((chunk_str, page_num))
            start = i
    return chunks


# ---------------------------------------------------------------------------
# Embedding model (cached for same session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """Load multilingual E5 model once. Fully offline after first download."""
    from sentence_transformers import SentenceTransformer
    
    try:
        model = SentenceTransformer("intfloat/multilingual-e5-small")
        return model
    except Exception as e:
        st.error(
            "Embedding model unavailable.\n\n"
            f"Details: {str(e)}\n\n"
            "Use the sample document for the no-download demo, or cache the model before "
            "processing uploaded PDFs offline."
        )
        raise


def build_embeddings(chunks: List[Tuple[str, int]], model) -> np.ndarray:
    """Encode chunk texts with E5 passage prefix. Returns (n_chunks, dim) array."""
    texts = [PASSAGE_PREFIX + text for text, _ in chunks]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS index for inner-product similarity (embeddings already normalized)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def keyword_score(chunks: List[Tuple[str, int]], query: str) -> np.ndarray:
    """BM25 scores for query against chunk texts. Returns array of length len(chunks)."""
    tokenized_corpus = [text.lower().split() for text, _ in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    return np.array(scores, dtype=np.float64)


def demo_search(
    query: str,
    chunks: List[Tuple[str, int]],
    bm25: BM25Okapi,
    top_k: int = TOP_K,
) -> List[Tuple[int, float, str, int]]:
    """
    BM25-only search for demo mode. No embeddings, no network.
    Returns same format as hybrid_search: (chunk_index, score, chunk_text, page_num).
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    scores = np.array(scores, dtype=np.float64)
    # Normalize to [0, 1] so confidence threshold works
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (int(idx), float(scores[idx]), chunks[idx][0], chunks[idx][1])
        for idx in top_indices
    ]


def hybrid_search(
    query: str,
    chunks: List[Tuple[str, int]],
    embeddings: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    model,
    top_k: int = TOP_K,
) -> List[Tuple[int, float, str, int]]:
    """
    Combine vector similarity and BM25. Returns list of
    (chunk_index, combined_score, chunk_text, page_num).
    """
    # Vector: encode query with E5 query prefix
    q_emb = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    q_emb = q_emb.astype(np.float32)
    vector_k = min(top_k * 4, len(chunks))  # Get more candidates for fusion
    scores_vec, indices_vec = faiss_index.search(q_emb, vector_k)
    scores_vec = scores_vec[0]
    indices_vec = indices_vec[0]

    # BM25 on full corpus
    bm25_scores = keyword_score(chunks, query)

    # Normalize and combine
    def _norm(x):
        x = np.array(x, dtype=np.float64)
        if x.max() > x.min():
            x = (x - x.min()) / (x.max() - x.min())
        return x

    n = len(chunks)
    vector_scores_norm = np.zeros(n)
    for i, idx in enumerate(indices_vec):
        if 0 <= idx < n:
            vector_scores_norm[idx] = scores_vec[i]

    bm25_norm = _norm(bm25_scores)
    vector_norm = _norm(vector_scores_norm)
    combined = HYBRID_VECTOR_WEIGHT * vector_norm + HYBRID_BM25_WEIGHT * bm25_norm

    top_indices = np.argsort(combined)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if idx < len(chunks):
            text, page = chunks[idx]
            results.append((int(idx), float(combined[idx]), text, page))
    return results


def _sentences_from(text: str) -> List[str]:
    """Split text into sentences (conservative)."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _query_terms(query: str) -> set:
    """Lowercased query terms for relevance scoring."""
    return set(re.findall(r'\w+', query.lower()))


def generate_answer(
    query: str,
    retrieved: List[Tuple[int, float, str, int]],
) -> Tuple[str, List[Tuple[int, str]], bool]:
    """
    Build answer strictly from retrieved chunks. No hallucination.
    Uses multiple top chunks: picks sentences that mention query terms when possible,
    then fills with leading sentences. Returns (answer_text, evidence_list, has_sufficient_evidence).
    """
    if not retrieved:
        return (
            "No sufficient evidence found in the document.",
            [],
            False,
        )

    best_score = retrieved[0][1]
    if best_score < CONFIDENCE_THRESHOLD:
        return (
            "No sufficient evidence found in the document.",
            [],
            False,
        )

    evidence = [(page, excerpt) for _, _, excerpt, page in retrieved]
    query_terms_set = _query_terms(query)
    max_answer_chars = 600

    # Collect candidate sentences from top chunks (by score order), preferring query-relevant ones
    seen_sentences: set = set()
    answer_sentences: List[str] = []
    total_len = 0

    for _, _, chunk_text, page in retrieved:
        if total_len >= max_answer_chars:
            break
        sentences = _sentences_from(chunk_text)
        # Prefer sentences that contain any query term
        with_terms = [s for s in sentences if query_terms_set and any(t in s.lower() for t in query_terms_set)]
        without_terms = [s for s in sentences if s not in with_terms]
        ordered = with_terms + without_terms
        for s in ordered:
            if total_len + len(s) > max_answer_chars and answer_sentences:
                break
            key = s[:80]  # dedupe by start of sentence
            if key not in seen_sentences:
                seen_sentences.add(key)
                answer_sentences.append(s)
                total_len += len(s)

    answer = " ".join(answer_sentences).strip() if answer_sentences else retrieved[0][2][:500].strip()
    return answer, evidence, True


def load_sample_document() -> bool:
    """
    Build chunks and BM25 index from built-in sample audit text. Fully offline – no model
    download, no network. Demo uses keyword (BM25) search only.
    """
    try:
        clear_analysis_state()
        chunks = chunk_text(SAMPLE_AUDIT_PAGES)
        if not chunks:
            return False
        tokenized_corpus = [text.lower().split() for text, _ in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        st.session_state.chunks = chunks
        st.session_state.bm25 = bm25
        st.session_state.processed = True
        st.session_state.pdf_path = None
        st.session_state.demo_mode = True
        st.session_state.uploaded_signature = "sample:audit"
        st.session_state.search_mode = "Keyword BM25"
        # Demo does not use embeddings/FAISS/model – no download
        return True
    except Exception as e:
        st.error(f"Failed to load sample: {str(e)}")
        return False


def clear_analysis_state():
    """Clear the most recent answer/evidence while preserving the indexed document."""
    for key in RESULT_STATE_KEYS:
        st.session_state.pop(key, None)


def clear_document_state():
    """Clear indexed document artifacts and any answer derived from them."""
    for key in DOCUMENT_STATE_KEYS:
        st.session_state.pop(key, None)
    clear_analysis_state()


def clear_session(temp_dir: Optional[str] = None):
    """Clear session state and optionally delete temp files."""
    clear_document_state()
    if temp_dir and os.path.isdir(temp_dir):
        try:
            for f in Path(temp_dir).iterdir():
                if f.is_file():
                    f.unlink()
        except Exception:
            pass


def document_stats(chunks: List[Tuple[str, int]]) -> Tuple[int, int]:
    """Return chunk count and distinct page count for display."""
    return len(chunks), len({page for _, page in chunks})


def build_answer_report(
    query: str,
    answer: str,
    retrieved: List[Tuple[int, float, str, int]],
) -> str:
    """Create a small markdown memo that can be downloaded after an analysis."""
    lines = [
        f"# {APP_NAME} Answer Memo",
        "",
        f"Question: {query}",
        "",
        "## Answer",
        answer,
        "",
        "## Evidence",
    ]
    for rank, (_, score, excerpt, page) in enumerate(retrieved, start=1):
        compact_excerpt = re.sub(r"\s+", " ", excerpt).strip()
        lines.extend([
            "",
            f"{rank}. Page {page} | score {score:.2f}",
            f"   {compact_excerpt}",
        ])
    lines.extend([
        "",
        "---",
        f"Generated locally by {APP_NAME}. No document text was sent to an external API.",
    ])
    return "\n".join(lines)


def compact_excerpt(text: str, limit: int = 520) -> str:
    """Normalize whitespace and trim long evidence excerpts."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def inject_styles():
    """Apply a polished workstation-style layer over Streamlit defaults."""
    st.markdown(
        """
        <style>
        :root {
            --pl-ink: #101418;
            --pl-ink-soft: #25313d;
            --pl-muted: #65707b;
            --pl-paper: #f7f8f4;
            --pl-panel: #ffffff;
            --pl-line: #d8ded5;
            --pl-green: #0b7563;
            --pl-blue: #1d5f8a;
            --pl-amber: #b46d1f;
            --pl-red: #9d3f36;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(247,248,244,.98) 0%, rgba(255,255,255,1) 48%),
                linear-gradient(90deg, rgba(29,95,138,.05) 0 1px, transparent 1px 32px),
                linear-gradient(0deg, rgba(11,117,99,.045) 0 1px, transparent 1px 32px);
            color: var(--pl-ink);
        }
        header[data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stMainMenu"],
        [data-testid="stDeployButton"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        [data-testid="stAppViewContainer"] {
            margin-top: 0 !important;
        }
        .block-container {
            max-width: 1220px;
            padding-top: .85rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #101820;
            border-right: 1px solid rgba(255,255,255,.12);
        }
        [data-testid="stSidebar"] * {
            color: #f8faf6;
        }
        [data-testid="stSidebar"] .stButton button {
            border-color: rgba(255,255,255,.22);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #f8faf6;
        }
        [data-testid="stSidebar"] div.stButton > button {
            background: #202b35;
            color: #f8faf6 !important;
            border: 1px solid rgba(255,255,255,.22);
        }
        [data-testid="stSidebar"] div.stButton > button:hover {
            background: #283746;
            border-color: rgba(255,255,255,.4);
            color: #ffffff !important;
        }
        .pl-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1.25rem;
            align-items: end;
            border-bottom: 1px solid var(--pl-line);
            padding: .15rem 0 1.2rem;
            margin-bottom: 1rem;
        }
        .pl-brand-row {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin-bottom: .5rem;
        }
        .pl-mark {
            display: inline-grid;
            place-items: center;
            width: 2rem;
            height: 2rem;
            border-radius: 7px;
            background: linear-gradient(135deg, var(--pl-ink) 0%, var(--pl-blue) 100%);
            color: #fff;
            font-size: .74rem;
            font-weight: 900;
            letter-spacing: .04em;
        }
        .pl-kicker {
            color: var(--pl-green);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .pl-title {
            font-family: Charter, Georgia, serif;
            font-size: clamp(2.4rem, 5vw, 4.7rem);
            line-height: .9;
            margin: 0;
            color: var(--pl-ink);
        }
        .pl-subtitle {
            max-width: 760px;
            color: var(--pl-muted);
            margin-top: .85rem;
            font-size: 1.02rem;
        }
        .pl-rail {
            border-left: 3px solid var(--pl-amber);
            padding-left: .85rem;
            color: var(--pl-muted);
            font-size: .88rem;
            line-height: 1.35;
            max-width: 280px;
        }
        .pl-status-card {
            background: rgba(255,255,255,.93);
            border: 1px solid var(--pl-line);
            border-radius: 8px;
            padding: .8rem .9rem;
            min-height: 88px;
            box-shadow: 0 10px 24px rgba(16, 20, 24, .045);
        }
        .pl-status-label {
            color: var(--pl-muted);
            font-size: .76rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 800;
        }
        .pl-status-value {
            color: var(--pl-ink);
            font-size: 1.15rem;
            font-weight: 800;
            margin-top: .35rem;
        }
        .pl-answer {
            background: var(--pl-panel);
            border: 1px solid var(--pl-line);
            border-left: 5px solid var(--pl-green);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            line-height: 1.58;
            box-shadow: 0 10px 26px rgba(16, 20, 24, .05);
        }
        .pl-evidence {
            background: rgba(255,255,255,.92);
            border: 1px solid var(--pl-line);
            border-radius: 8px;
            padding: .85rem .95rem;
            margin-bottom: .7rem;
        }
        .pl-evidence-meta {
            display: flex;
            gap: .55rem;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: .45rem;
            font-size: .78rem;
            font-weight: 800;
            color: var(--pl-muted);
            text-transform: uppercase;
            letter-spacing: .05em;
        }
        .pl-pill {
            display: inline-flex;
            border: 1px solid var(--pl-line);
            background: var(--pl-paper);
            border-radius: 999px;
            padding: .16rem .55rem;
            color: var(--pl-ink);
        }
        .pl-evidence-text {
            color: var(--pl-ink);
            line-height: 1.52;
        }
        .pl-section-label {
            margin-top: .65rem;
            color: var(--pl-ink-soft);
            font-size: .82rem;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        div.stButton > button,
        div.stDownloadButton > button {
            min-height: 2.8rem;
            border-radius: 8px;
            border: 1px solid #c9d2ca;
            background: #ffffff !important;
            color: var(--pl-ink) !important;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(16, 20, 24, .045);
            white-space: normal;
        }
        div.stButton > button[kind="primary"] {
            background: var(--pl-green) !important;
            border-color: var(--pl-green) !important;
            color: #ffffff !important;
        }
        div.stButton > button[data-testid="stBaseButton-primary"] {
            background: var(--pl-green) !important;
            border-color: var(--pl-green) !important;
            color: #ffffff !important;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            border-color: var(--pl-green);
            background: #eef7f3 !important;
            color: var(--pl-ink) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #095f51 !important;
            border-color: #095f51 !important;
            color: #ffffff !important;
        }
        div.stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: #095f51 !important;
            border-color: #095f51 !important;
            color: #ffffff !important;
        }
        div.stButton > button:disabled,
        div.stButton > button[disabled] {
            background: #e9eee9 !important;
            border-color: #d2dbd4 !important;
            color: #68746d !important;
            opacity: 1;
            box-shadow: none;
            cursor: not-allowed;
        }
        div.stTextInput label,
        div.stFileUploader label {
            color: var(--pl-ink-soft) !important;
            font-weight: 800;
        }
        div.stTextInput > div > div > input {
            border-radius: 8px;
            background: #ffffff !important;
            border: 1px solid #c9d2ca;
            color: var(--pl-ink) !important;
            min-height: 3rem;
        }
        div.stTextInput > div > div > input::placeholder {
            color: #758179;
            opacity: 1;
        }
        div.stAlert {
            color: var(--pl-ink);
        }
        section[data-testid="stSidebar"] div.stButton > button {
            background: #202b35 !important;
            color: #f8faf6 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: #283746 !important;
            color: #ffffff !important;
        }
        @media (max-width: 760px) {
            .pl-header {
                grid-template-columns: 1fr;
            }
            .pl-rail {
                max-width: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the main project header."""
    st.markdown(
        f"""
        <section class="pl-header">
            <div>
                <div class="pl-brand-row">
                    <span class="pl-mark">AU</span>
                    <span class="pl-kicker">{APP_TAGLINE}</span>
                </div>
                <h1 class="pl-title">{APP_NAME}</h1>
                <div class="pl-subtitle">
                    Ask audit questions, trace every answer back to source pages, and export
                    a review-ready memo from a local document session.
                </div>
            </div>
            <div class="pl-rail">
                Private by default. Evidence first. Built for fast audit review.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_cards():
    """Render document and runtime status as compact dashboard cards."""
    processed = st.session_state.get("processed", False)
    chunks = st.session_state.get("chunks", [])
    chunk_count, page_count = document_stats(chunks) if processed else (0, 0)
    source = "Sample audit report" if st.session_state.get("demo_mode") else "Uploaded PDF"
    search_mode = st.session_state.get("search_mode")
    if not search_mode:
        search_mode = "Keyword BM25" if st.session_state.get("demo_mode") else "Hybrid vector + BM25"
    if not processed:
        source = "No document"
        search_mode = "Waiting"

    values = [
        ("Document", source),
        ("Indexed pages", str(page_count)),
        ("Chunks", str(chunk_count)),
        ("Search", search_mode),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                f"""
                <div class="pl-status-card">
                    <div class="pl-status-label">{label}</div>
                    <div class="pl-status-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_evidence_cards(retrieved: List[Tuple[int, float, str, int]]):
    """Render retrieved chunks with page and score metadata."""
    for rank, (_, score, excerpt, page) in enumerate(retrieved, start=1):
        safe_excerpt = html.escape(compact_excerpt(excerpt))
        st.markdown(
            f"""
            <div class="pl-evidence">
                <div class="pl-evidence-meta">
                    <span class="pl-pill">Evidence {rank}</span>
                    <span>Page {page}</span>
                    <span>Score {score:.2f}</span>
                </div>
                <div class="pl-evidence-text">{safe_excerpt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title=f"{APP_NAME} | Audit Evidence Workspace",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    render_header()

    # Sidebar
    with st.sidebar:
        st.header(APP_NAME)
        st.caption("Local audit review")
        st.divider()
        st.subheader("Document")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="uploader")
        semantic_search = st.toggle(
            "Semantic search",
            value=False,
            help="Slower first run. Downloads/loads the local E5 model and builds embeddings.",
        )
        if st.button("Load sample document (demo)", help="Use built-in sample audit report to try the app without uploading a PDF"):
            with st.spinner("Loading sample audit document..."):
                if load_sample_document():
                    st.success("Sample loaded. Ask a question below.")
                    st.rerun()
                else:
                    st.error("Failed to load sample.")
        ephemeral = st.toggle("Ephemeral mode", value=False, key="ephemeral_toggle")
        if st.button("Clear document", type="secondary"):
            temp_dir = st.session_state.get("temp_dir")
            clear_session(temp_dir)
            if temp_dir and os.path.isdir(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
            st.session_state.pop("temp_dir", None)
            st.rerun()
        st.divider()
        st.caption("For large reports, leave Semantic search off first. You can still ask evidence-backed questions with the fast keyword index.")

    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="audit_ai_")

    temp_dir = st.session_state.temp_dir

    if uploaded_file is not None:
        uploaded_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("uploaded_signature") != uploaded_signature:
            clear_document_state()
            st.session_state.uploaded_signature = uploaded_signature
            st.session_state.processed = False

    # Process PDF when uploaded
    if uploaded_file is not None and not st.session_state.get("processed"):
        status_box = st.empty()
        progress_bar = st.progress(0)
        try:
            status_box.info(f"Saving {uploaded_file.name} ({uploaded_file.size / (1024 * 1024):.1f} MB)...")
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state.pdf_path = path
            progress_bar.progress(0.08)

            pages_text = extract_pdf_text(path, progress=progress_bar, status=status_box)
            if not pages_text:
                st.error("Could not extract text from this PDF. It may be scanned or image-only. OCR is not supported yet.")
                st.session_state.pdf_path = None
                progress_bar.empty()
            else:
                status_box.info(f"Chunking extracted text from {len(pages_text)} text pages...")
                chunks = chunk_text(pages_text)
                progress_bar.progress(0.58)
                if not chunks:
                    st.warning("No text chunks produced. PDF may be too short or mostly non-text.")
                    st.session_state.pdf_path = None
                    progress_bar.empty()
                else:
                    tokenized_corpus = [text.lower().split() for text, _ in chunks]
                    st.session_state.bm25 = BM25Okapi(tokenized_corpus)
                    st.session_state.chunks = chunks
                    st.session_state.demo_mode = False

                    if semantic_search:
                        status_box.info("Loading embedding model and building semantic index...")
                        progress_bar.progress(0.68)
                        model = load_embedding_model()
                        embeddings = build_embeddings(chunks, model)
                        progress_bar.progress(0.9)
                        faiss_index = build_faiss_index(embeddings)
                        st.session_state.embeddings = embeddings
                        st.session_state.faiss_index = faiss_index
                        st.session_state.model = model
                        st.session_state.search_mode = "Hybrid vector + BM25"
                    else:
                        st.session_state.pop("embeddings", None)
                        st.session_state.pop("faiss_index", None)
                        st.session_state.pop("model", None)
                        st.session_state.search_mode = "Keyword BM25"

                    st.session_state.processed = True
                    if ephemeral:
                        st.session_state.ephemeral = True
                    progress_bar.progress(1.0)
                    status_box.success(f"Indexed {len(chunks)} chunks from {len(pages_text)} text pages.")
                    st.rerun()
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")
            st.session_state.pdf_path = None
            progress_bar.empty()

    render_status_cards()

    # Show status
    if st.session_state.get("processed"):
        chunks = st.session_state.chunks
        if st.session_state.get("demo_mode"):
            st.success(f"Sample audit document loaded. {len(chunks)} chunks indexed.")
        else:
            st.success(f"PDF processed. {len(chunks)} chunks indexed.")
    else:
        if uploaded_file is not None and not st.session_state.get("pdf_path"):
            pass  # Error already shown
        else:
            st.info("Load the sample audit report or upload a PDF.")

    # Question input and preset buttons
    st.markdown('<div class="pl-section-label">Inquiry</div>', unsafe_allow_html=True)
    st.subheader("Ask the document")
    cols = st.columns(len(PRESET_QUERIES))
    for i, (col, q) in enumerate(zip(cols, PRESET_QUERIES)):
        with col:
            if st.button(q, key=f"preset_{i}"):
                st.session_state.query = q
                st.rerun()

    query_input = st.text_input("Question", key="query", placeholder="Ask about findings, controls, compliance, risks...")
    analyze_clicked = st.button("Analyze", type="primary", disabled=not st.session_state.get("processed"))

    if analyze_clicked:
        if not query_input or not query_input.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.get("processed"):
            st.warning("Please upload and process a PDF first.")
        else:
            with st.spinner("Searching and analyzing..."):
                try:
                    chunks = st.session_state.chunks
                    if st.session_state.get("search_mode") != "Hybrid vector + BM25":
                        # Fast mode: BM25 only, no network, no model.
                        retrieved = demo_search(
                            query_input.strip(),
                            chunks,
                            st.session_state.bm25,
                            top_k=TOP_K,
                        )
                    else:
                        retrieved = hybrid_search(
                            query_input.strip(),
                            chunks,
                            st.session_state.embeddings,
                            st.session_state.faiss_index,
                            st.session_state.model,
                            top_k=TOP_K,
                        )
                    answer, evidence, has_evidence = generate_answer(query_input.strip(), retrieved)
                    st.session_state.last_answer = answer
                    st.session_state.last_evidence = evidence
                    st.session_state.last_has_evidence = has_evidence
                    st.session_state.last_retrieved = retrieved
                    st.session_state.last_query = query_input.strip()
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.session_state.last_has_evidence = False

    # Display last result
    if st.session_state.get("last_has_evidence") is not None:
        st.markdown('<div class="pl-section-label">Output</div>', unsafe_allow_html=True)
        st.subheader("Answer")
        if st.session_state.last_has_evidence:
            safe_answer = html.escape(st.session_state.last_answer)
            st.markdown(f'<div class="pl-answer">{safe_answer}</div>', unsafe_allow_html=True)
            st.download_button(
                "Export answer memo",
                data=build_answer_report(
                    st.session_state.get("last_query", query_input.strip()),
                    st.session_state.last_answer,
                    st.session_state.get("last_retrieved", []),
                ),
                file_name=EXPORT_FILENAME,
                mime="text/markdown",
            )
            with st.expander("Evidence trail", expanded=True):
                render_evidence_cards(st.session_state.get("last_retrieved", []))
        else:
            st.info(st.session_state.last_answer)

    # Ephemeral: delete file when session ends is implicit (temp dir); optional explicit note
    if st.session_state.get("ephemeral") and st.session_state.get("pdf_path"):
        st.sidebar.caption("Ephemeral mode: file will be removed when you clear the document.")


if __name__ == "__main__":
    main()
