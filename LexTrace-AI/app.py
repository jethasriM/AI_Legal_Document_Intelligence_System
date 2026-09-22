
import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st


# ============================================================
# LexTrace AI — Evidence-grounded legal document intelligence
# UI-only orchestration layer
#
# This app intentionally uses the existing pipeline as the
# backend. It does not duplicate ingestion/retrieval/generation.
# ============================================================

st.set_page_config(
    page_title="LexTrace AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WORK_UPLOADS = ROOT / ".lextrace_uploads"
REVIEWS = OUTPUTS / "reviews"

OUTPUTS.mkdir(exist_ok=True)
WORK_UPLOADS.mkdir(exist_ok=True)
REVIEWS.mkdir(exist_ok=True)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #17211f;
        --muted: #68736f;
        --paper: #f7f5ef;
        --paper-2: #eeece4;
        --card: #ffffff;
        --line: #d9ddd7;
        --green: #236b58;
        --green-soft: #e3f0ea;
        --gold: #b78332;
        --gold-soft: #f7ecd8;
        --red: #a94c45;
        --red-soft: #f8e5e2;
        --blue: #315c73;
        --blue-soft: #e6eff3;
    }

    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    .stApp {
        background:
            linear-gradient(180deg, #f7f5ef 0%, #f3f1ea 48%, #eeece4 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #17211f;
        border-right: 1px solid #26312e;
    }

    section[data-testid="stSidebar"] * {
        color: #e9eee9 !important;
    }

    .side-brand {
        padding: 0.8rem 0.2rem 1.4rem;
        border-bottom: 1px solid #35413d;
        margin-bottom: 1.4rem;
    }

    .side-brand-title {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .side-brand-sub {
        color: #aebbb5 !important;
        font-size: 0.75rem;
        line-height: 1.5;
        margin-top: 0.3rem;
    }

    .side-label {
        color: #8da098 !important;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.45rem;
    }

    .side-capability {
        font-size: 0.78rem;
        padding: 0.28rem 0;
        color: #cbd5d0 !important;
    }

    .system-card {
        margin-top: 1.5rem;
        padding: 0.8rem;
        border: 1px solid #40514a;
        border-radius: 12px;
        background: #1d2b27;
    }

    .system-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        background: #65c49e;
        border-radius: 50%;
        margin-right: 0.4rem;
    }

    /* ---------- Hero ---------- */

    .hero {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 2.2rem 2.4rem;
        box-shadow: 0 12px 36px rgba(23, 33, 31, 0.06);
        margin-bottom: 1.8rem;
    }

    .eyebrow {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--green);
        font-weight: 500;
        margin-bottom: 0.65rem;
    }

    .hero h1 {
        font-size: clamp(2.4rem, 5vw, 4.7rem);
        line-height: 0.98;
        letter-spacing: -0.065em;
        margin: 0;
        color: var(--ink);
        font-weight: 700;
    }

    .hero h1 span {
        color: var(--green);
    }

    .hero-copy {
        max-width: 720px;
        color: #59645f;
        font-size: 1rem;
        line-height: 1.65;
        margin-top: 1rem;
    }

    .hero-note {
        margin-top: 1.2rem;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        background: var(--green-soft);
        color: var(--green);
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ---------- Section ---------- */

    .section-kicker {
        color: var(--green);
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 1.8rem;
        margin-bottom: 0.25rem;
    }

    .section-heading {
        font-size: 1.8rem;
        letter-spacing: -0.04em;
        font-weight: 700;
        margin: 0 0 0.25rem;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }

    /* ---------- Cards ---------- */

    .metric-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        min-height: 105px;
        box-shadow: 0 5px 20px rgba(23, 33, 31, 0.035);
    }

    .metric-label {
        color: #78827e;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.5rem;
        letter-spacing: -0.035em;
    }

    .metric-detail {
        color: #8a938f;
        font-size: 0.68rem;
        margin-top: 0.15rem;
    }

    .panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 1.1rem;
        box-shadow: 0 5px 20px rgba(23, 33, 31, 0.035);
    }

    .panel-title {
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .mono {
        font-family: "IBM Plex Mono", monospace;
    }

    /* ---------- Status badges ---------- */

    .badge {
        display: inline-block;
        padding: 0.2rem 0.48rem;
        border-radius: 999px;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.61rem;
        font-weight: 500;
        margin-left: 0.35rem;
    }

    .badge-verified {
        color: var(--green);
        background: var(--green-soft);
    }

    .badge-uncertain {
        color: #8b651f;
        background: var(--gold-soft);
    }

    .badge-missing {
        color: var(--red);
        background: var(--red-soft);
    }

    .field-name {
        font-size: 0.82rem;
        color: #5f6a65;
        margin-bottom: 0.35rem;
    }

    .field-value {
        background: #f6f7f4;
        border: 1px solid #e3e6e1;
        border-radius: 10px;
        padding: 0.7rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.74rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    /* ---------- Source viewer ---------- */

    .source-viewer {
        background: #202725;
        color: #e9eee9;
        border-radius: 13px;
        padding: 1rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.73rem;
        line-height: 1.65;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        max-height: 620px;
        overflow-y: auto;
    }

    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #aab7b1;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.65rem;
        margin-bottom: 0.65rem;
    }

    /* ---------- Draft ---------- */

    .draft-shell {
        background: #fffdf8;
        border: 1px solid #dcd9cf;
        border-radius: 15px;
        padding: 1.25rem 1.35rem;
    }

    .draft-section {
        padding: 0.9rem 0;
        border-bottom: 1px solid #ebe8df;
    }

    .draft-section:last-child {
        border-bottom: none;
    }

    .draft-section h4 {
        font-size: 1rem;
        margin: 0 0 0.45rem;
        color: var(--ink);
    }

    .draft-content {
        color: #46514d;
        font-size: 0.84rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    .citation-chip {
        display: inline-block;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        background: var(--blue-soft);
        color: var(--blue);
        padding: 0.16rem 0.35rem;
        border-radius: 5px;
        margin: 0.12rem 0.12rem 0.12rem 0;
    }

    /* ---------- Evidence ---------- */

    .evidence-card {
        background: #f8f8f5;
        border: 1px solid #e1e4de;
        border-radius: 11px;
        padding: 0.75rem;
        margin-bottom: 0.65rem;
    }

    .evidence-meta {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        color: var(--green);
        margin-bottom: 0.35rem;
    }

    .evidence-text {
        color: #5b6661;
        font-size: 0.74rem;
        line-height: 1.55;
    }

    /* ---------- Upload ---------- */

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #9eaaa4;
        border-radius: 15px;
        padding: 0.4rem;
    }

    div[data-testid="stFileUploader"] button {
        color: #17211f !important;
        background: #ffffff !important;
        border: 1px solid #bfc8c2 !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        color: #17211f !important;
        background: #eeece4 !important;
    }

    div[data-testid="stFileUploader"] button:focus,
    div[data-testid="stFileUploader"] button:active {
        color: #17211f !important;
    }

    /* ---------- Streamlit visibility fixes ---------- */

    /* Tabs: keep labels readable on the warm ivory background. */
    div[data-baseweb="tab-list"] {
        gap: 0.2rem;
    }
    
    button[data-baseweb="tab"] {
    color: #0f0f0f !important;
    background: transparent !important;
    border: none !important;
    }

    button[data-baseweb="tab"] * {
    color: #0f0f0f !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff4b4b !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] * {
    color: #ff4b4b !important;
    }

    /* Streamlit metrics: dark text instead of the theme's light default. */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #68736f !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] > div,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
        color: #17211f !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #68736f !important;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #bfc8c2;
        font-weight: 600;
        min-height: 2.5rem;
    }

    .stButton > button[kind="primary"] {
        background: var(--green);
        border-color: var(--green);
        color: white;
    }

    /* ---------- Expanders ---------- */

    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 11px;
        background: #ffffff;
    }

    /* ---------- Footer ---------- */

    .footer {
        border-top: 1px solid #d5d8d3;
        margin-top: 3rem;
        padding-top: 1rem;
        color: #7a8580;
        font-size: 0.7rem;
        text-align: center;
    }

    @media (max-width: 900px) {
        .hero {
            padding: 1.4rem;
        }
        .hero h1 {
            font-size: 2.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_json_load(path: Path, default=None):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def newest_file(pattern: str):
    files = list(OUTPUTS.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def human_doc_type(value):
    mapping = {
        "case_intake": "Case Intake",
        "legal_notice": "Legal Notice",
        "title_deed": "Title Deed",
    }
    return mapping.get(value, str(value).replace("_", " ").title())


def display_value(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def status_for_field(field_name, value, field_evidence):
    evidence = field_evidence.get(field_name)

    if isinstance(evidence, dict):
        status = evidence.get("status")
        if status in {"verified", "uncertain", "missing"}:
            return status

    text = display_value(value).lower()

    if not text or text in {"none", "[]", "{}"}:
        return "missing"

    uncertainty_terms = [
        "illegible",
        "unclear",
        "damaged",
        "missing",
        "unknown",
        "not found",
        "???",
        "??",
    ]

    if any(term in text for term in uncertainty_terms):
        return "uncertain"

    return "verified"


def status_badge(status):
    labels = {
        "verified": ("✓ verified", "badge-verified"),
        "uncertain": ("? uncertain", "badge-uncertain"),
        "missing": ("× missing", "badge-missing"),
    }

    label, cls = labels.get(status, ("? review", "badge-uncertain"))
    return f'<span class="badge {cls}">{label}</span>'


def parse_markdown_draft(path: Path):
    if not path or not path.exists():
        return {
            "title": "No draft available",
            "generated": "",
            "grounding": None,
            "sections": [],
            "raw": "",
        }

    raw = path.read_text(encoding="utf-8", errors="replace")

    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    generated_match = re.search(
        r"\*Generated:\s*(.*?)\s*\|\s*Grounding:\s*(\d+)%\*",
        raw,
    )

    sections = []
    matches = list(
        re.finditer(
            r"^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)",
            raw,
            re.MULTILINE | re.DOTALL,
        )
    )

    for match in matches:
        name = match.group(1).strip()
        content = match.group(2).strip()

        evidence = []
        for ev in re.finditer(
            r"-\s+`([^`]+)`\s+\(score\s+([0-9.]+)\):\s+\*\"(.*?)\"\*",
            content,
            re.DOTALL,
        ):
            evidence.append(
                {
                    "passage_id": ev.group(1),
                    "score": float(ev.group(2)),
                    "excerpt": ev.group(3).replace("\n", " "),
                }
            )

        # Remove the rendered evidence block from the main section text.
        clean_content = re.split(r"\n\*\*Evidence:\*\*", content, maxsplit=1)[0].strip()

        sections.append(
            {
                "name": name,
                "content": clean_content,
                "evidence": evidence,
            }
        )

    return {
        "title": title_match.group(1).strip() if title_match else path.stem,
        "generated": generated_match.group(1) if generated_match else "",
        "grounding": int(generated_match.group(2)) if generated_match else None,
        "sections": sections,
        "raw": raw,
    }


def find_draft_for_doc(doc_id):
    candidates = list(OUTPUTS.glob(f"{doc_id}*draft.md"))
    if not candidates:
        # Current naming convention is draft_<id>_draft.md, so search contents.
        for path in OUTPUTS.glob("*_draft.md"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                if doc_id in text:
                    candidates.append(path)
            except Exception:
                pass

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_pipeline(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".txt"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", uploaded_file.name)
    input_path = WORK_UPLOADS / safe_name

    input_path.write_bytes(uploaded_file.getbuffer())

    cmd = [
        sys.executable,
        str(ROOT / "src" / "pipeline.py"),
        "--input",
        str(WORK_UPLOADS),
    ]

    try:
        # Force UTF-8 for the child Python process and for captured output.
        # This prevents Windows cp1252 from crashing on characters such as
        # checkmarks, arrows, OCR symbols, and other Unicode emitted by the
        # LexTrace pipeline.
        pipeline_env = os.environ.copy()
        pipeline_env["PYTHONIOENCODING"] = "utf-8"
        pipeline_env["PYTHONUTF8"] = "1"

        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=pipeline_env,
            timeout=300,
        )

        log = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0:
            return False, log

        return True, log

    except subprocess.TimeoutExpired:
        return False, "Pipeline timed out after 300 seconds."
    except Exception as exc:
        return False, str(exc)


def latest_processed_document():
    files = list(OUTPUTS.glob("*_extracted.json"))
    if not files:
        return None

    latest = max(files, key=lambda p: p.stat().st_mtime)
    return safe_json_load(latest, {})


def render_metric(label, value, detail=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_field(field_name, value, evidence):
    status = status_for_field(field_name, value, evidence)

    st.markdown(
        f"""
        <div class="field-name">
            {field_name.replace("_", " ").title()}
            {status_badge(status)}
        </div>
        <div class="field-value">{display_value(value)}</div>
        """,
        unsafe_allow_html=True,
    )

    field_ev = evidence.get(field_name)

    if field_ev:
        with st.expander("View source evidence", expanded=False):
            if isinstance(field_ev, dict):
                page = field_ev.get("page")
                excerpt = field_ev.get("excerpt")
                reason = field_ev.get("reason")

                if page is not None:
                    st.caption(f"Page {page}")

                if excerpt:
                    st.markdown(f"> {excerpt}")

                if reason:
                    st.caption(reason)
            else:
                st.json(field_ev)


def render_draft(draft):
    st.markdown('<div class="draft-shell">', unsafe_allow_html=True)

    if draft["grounding"] is not None:
        st.caption(f"Model-reported grounding assessment: {draft['grounding']}%")

    for section in draft["sections"]:
        st.markdown(
            f'<div class="draft-section"><h4>{section["name"]}</h4>',
            unsafe_allow_html=True,
        )

        content = section["content"] or "[Section not generated]"
        st.markdown(
            f'<div class="draft-content">{content}</div>',
            unsafe_allow_html=True,
        )

        if section["evidence"]:
            st.markdown("**Source evidence**")

            for item in section["evidence"]:
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-meta">
                            {item["passage_id"]} · score {item["score"]:.2f}
                        </div>
                        <div class="evidence-text">
                            {html.escape(str(item["excerpt"]))}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def load_evaluation():
    return safe_json_load(OUTPUTS / "evaluation_report.json", {})


# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = ""

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="side-brand-title">⚖ LexTrace AI</div>
            <div class="side-brand-sub">
                Evidence-grounded legal document intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Workflow</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Analyze",
            "Documents",
            "Review",
            "Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)

    capabilities = [
        "OCR-aware ingestion",
        "Structured extraction",
        "Hybrid retrieval",
        "Grounded generation",
        "Page-level evidence",
        "Human review",
    ]

    for item in capabilities:
        st.markdown(
            f'<div class="side-capability">✓ {item}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="system-card">
            <div style="font-size:0.75rem;font-weight:600;">
                <span class="system-dot"></span>System ready
            </div>
            <div style="font-size:0.65rem;color:#8fa09a;margin-top:0.3rem;">
                Evidence tracing enabled
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Evidence-grounded document intelligence</div>
        <h1>LexTrace <span>AI</span></h1>
        <div class="hero-copy">
            Understand messy legal documents, extract structured facts,
            retrieve supporting evidence, and generate reviewable drafts
            without losing the connection to the source.
        </div>
        <div class="hero-note">
            ● Source evidence remains visible throughout the workflow
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ANALYZE
# ============================================================

if page == "Analyze":

    st.markdown('<div class="section-kicker">01 · Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Start an analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Upload a legal-style document and let the existing LexTrace pipeline process it.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload document",
        type=["txt", "pdf", "png", "jpg", "jpeg", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        left, right = st.columns([3, 1])

        with left:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">Selected document</div>
                    <div class="mono" style="font-size:0.8rem;">
                        {uploaded.name}
                    </div>
                    <div style="color:#7a8580;font-size:0.72rem;margin-top:0.3rem;">
                        {uploaded.size / 1024:.1f} KB · Ready for analysis
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            process = st.button(
                "Analyze document",
                type="primary",
                use_container_width=True,
            )

        if process:
            with st.spinner("Running ingestion → retrieval → grounded generation..."):
                ok, log = run_pipeline(uploaded)

            st.session_state.processed = ok
            st.session_state.pipeline_log = log

            if ok:
                st.success("Analysis completed successfully.")
                st.rerun()
            else:
                st.error("The pipeline returned an error.")
                with st.expander("Pipeline log"):
                    st.code(log)

    doc = latest_processed_document()

    if doc:
        st.markdown('<div class="section-kicker">02 · Intelligence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-heading">Document intelligence</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-copy">A structured view of what was extracted, what remains uncertain, and what the draft used as evidence.</div>',
            unsafe_allow_html=True,
        )

        structured = doc.get("structured_fields", {})
        field_evidence = doc.get("field_evidence", {})
        warnings = doc.get("warnings", [])

        doc_type = human_doc_type(doc.get("doc_type", "unknown"))
        quality = doc.get("confidence_flags", {}).get("quality_score", 0)
        pages = doc.get("page_count", 1)

        verified = sum(
            status_for_field(k, v, field_evidence) == "verified"
            for k, v in structured.items()
        )
        uncertain = sum(
            status_for_field(k, v, field_evidence) == "uncertain"
            for k, v in structured.items()
        )

        metrics = st.columns(4)

        with metrics[0]:
            render_metric("Document", doc_type, "Detected document type")

        with metrics[1]:
            render_metric("Source quality", f"{quality}/100", "Extraction quality")

        with metrics[2]:
            render_metric("Pages", pages, "Detected pages")

        with metrics[3]:
            render_metric(
                "Fields",
                len(structured),
                f"{verified} verified · {uncertain} uncertain",
            )

        # --------------------------------------------------------
        # SOURCE VIEWER
        # --------------------------------------------------------

        st.markdown("### Source document")

        source_col, info_col = st.columns([1.35, 0.65], gap="large")

        with source_col:
            page_texts = doc.get("page_texts") or []
            source_text = doc.get("cleaned_text", "")

            if page_texts:
                page_choice = st.selectbox(
                    "Source page",
                    range(1, len(page_texts) + 1),
                    format_func=lambda x: f"Page {x}",
                )
                source_text = page_texts[page_choice - 1]

            st.markdown(
                f"""
                <div class="source-viewer">
                    <div class="source-header">
                        <span>{doc.get("source_path", "uploaded document")}</span>
                        <span>source text</span>
                    </div>
                    {html.escape(str(source_text))}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with info_col:
            st.markdown(
                '<div class="panel-title">Extraction health</div>',
                unsafe_allow_html=True,
            )

            render_metric("Verified", f"{verified}/{len(structured)}", "Fields with source support")
            render_metric("Uncertain", uncertain, "Fields requiring attention")
            render_metric("Warnings", len(warnings), "Extraction warnings")

            if warnings:
                with st.expander("View extraction warnings"):
                    for warning in warnings:
                        st.warning(warning)

        # --------------------------------------------------------
        # EXTRACTION + DRAFT
        # --------------------------------------------------------

        st.markdown("### Working view")

        extraction_tab, draft_tab, evidence_tab = st.tabs(
            ["Extracted intelligence", "Grounded draft", "Evidence trail"]
        )

        with extraction_tab:
            if not structured:
                st.info("No structured fields were extracted.")
            else:
                field_items = list(structured.items())

                for start in range(0, len(field_items), 2):
                    cols = st.columns(2, gap="large")

                    for idx, col in enumerate(cols):
                        if start + idx >= len(field_items):
                            continue

                        key, value = field_items[start + idx]

                        with col:
                            render_field(key, value, field_evidence)

        draft_path = find_draft_for_doc(doc.get("doc_id", ""))
        draft = parse_markdown_draft(draft_path)

        with draft_tab:
            if draft_path:
                render_draft(draft)

                st.markdown("### Review this draft")

                edited = st.text_area(
                    "Editable draft",
                    value=draft["raw"],
                    height=430,
                    label_visibility="collapsed",
                )

                save_col, reset_col = st.columns([1, 1])

                with save_col:
                    if st.button("Save reviewed draft", type="primary"):
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        review_path = REVIEWS / f"{doc.get('doc_id', 'document')}_review_{stamp}.md"
                        review_path.write_text(edited, encoding="utf-8")

                        st.success(
                            f"Review saved to {review_path.relative_to(ROOT)}"
                        )

                with reset_col:
                    if st.button("Restore generated draft"):
                        st.rerun()

            else:
                st.info("No generated draft was found for this document.")

        with evidence_tab:
            if field_evidence:
                for key, evidence in field_evidence.items():
                    with st.expander(
                        key.replace("_", " ").title(),
                        expanded=False,
                    ):
                        if isinstance(evidence, dict):
                            status = evidence.get("status", "unknown")
                            st.markdown(
                                f"Status: {status_badge(status)}",
                                unsafe_allow_html=True,
                            )
                            st.write("Value:", evidence.get("value"))
                            if evidence.get("page") is not None:
                                st.write("Page:", evidence.get("page"))
                            if evidence.get("excerpt"):
                                st.markdown(
                                    f"> {evidence['excerpt']}"
                                )
                            if evidence.get("reason"):
                                st.caption(evidence["reason"])
                        else:
                            st.json(evidence)
            else:
                st.info("No field-level evidence map was found.")

# ============================================================
# DOCUMENTS
# ============================================================

elif page == "Documents":

    st.markdown('<div class="section-kicker">03 · Corpus</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Processed documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Browse the extraction artifacts produced by the LexTrace pipeline.</div>',
        unsafe_allow_html=True,
    )

    extracted_files = sorted(
        OUTPUTS.glob("*_extracted.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not extracted_files:
        st.info("No processed documents yet. Analyze a document first.")
    else:
        for path in extracted_files:
            item = safe_json_load(path, {})
            if not item:
                continue

            quality = item.get("confidence_flags", {}).get("quality_score", 0)
            doc_type = human_doc_type(item.get("doc_type", "unknown"))
            field_count = len(item.get("structured_fields", {}))
            warning_count = len(item.get("warnings", []))

            with st.container(border=True):
                a, b, c, d = st.columns([2.2, 1.1, 1.1, 1.1])

                with a:
                    st.markdown(f"**{doc_type}**")
                    st.caption(item.get("source_path", path.name))

                with b:
                    st.metric("Quality", f"{quality}/100")

                with c:
                    st.metric("Fields", field_count)

                with d:
                    st.metric("Warnings", warning_count)

# ============================================================
# REVIEW
# ============================================================

elif page == "Review":

    st.markdown('<div class="section-kicker">04 · Human review</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Review history</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Operator edits are stored separately so they can be compared with generated drafts and used by the improvement layer.</div>',
        unsafe_allow_html=True,
    )

    reviews = sorted(
        REVIEWS.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not reviews:
        st.info("No reviewed drafts have been saved yet.")
    else:
        for review in reviews:
            with st.expander(review.name):
                st.text_area(
                    "Saved review",
                    review.read_text(encoding="utf-8", errors="replace"),
                    height=350,
                    key=str(review),
                )

# ============================================================
# EVALUATION
# ============================================================

elif page == "Evaluation":

    st.markdown('<div class="section-kicker">05 · Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">System evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Development-time metrics for extraction, retrieval, grounding, citation quality, and improvement.</div>',
        unsafe_allow_html=True,
    )

    report = load_evaluation()

    if not report:
        st.info(
            "No evaluation report found. Run `python src\\evaluation.py` "
            "from the project root first."
        )
    else:
        # Support both nested and flat report formats.
        metrics = report.get("metrics", report)

        candidates = {
            "Extraction": metrics.get("extraction"),
            "Retrieval": metrics.get("retrieval"),
            "Grounding": metrics.get("grounding"),
            "Citation coverage": metrics.get("citation_coverage"),
            "Citation validity": metrics.get("citation_validity"),
            "Section evidence": metrics.get("section_evidence"),
            "Improvement": metrics.get("improvement"),
        }

        visible = [
            (name, value)
            for name, value in candidates.items()
            if value is not None
        ]

        if visible:
            cols = st.columns(min(4, len(visible)))

            for idx, (name, value) in enumerate(visible):
                if isinstance(value, dict):
                    value = value.get("score", value.get("percentage", 0))

                try:
                    numeric = float(value)
                    if numeric <= 1:
                        numeric *= 100
                    formatted = f"{numeric:.1f}%"
                except Exception:
                    formatted = str(value)

                with cols[idx % len(cols)]:
                    render_metric(
                        name,
                        formatted,
                        "Development metric",
                    )

        st.markdown("### Evaluation report")

        with st.expander("View raw evaluation JSON"):
            st.json(report)

        st.caption(
            "These metrics describe the current development/evaluation set; "
            "they should not be presented as production accuracy."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        LexTrace AI · Evidence-grounded legal document intelligence ·
        Human verification required before professional use
    </div>
    """,
    unsafe_allow_html=True,
)
