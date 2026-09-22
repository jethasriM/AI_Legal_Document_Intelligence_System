# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE FLOW                            │
│                                                                 │
│  [Messy Input]                                                  │
│  .txt / .pdf                                                    │
│  (OCR noise, handwriting,                                       │
│   partial illegibility)                                         │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │  INGESTION   │  ← clean_ocr_noise()                          │
│  │              │  ← detect_doc_type()                          │
│  │ ExtractedDoc │  ← extract_*_fields()                         │
│  │  + warnings  │  ← analyze_confidence()                       │
│  └──────┬───────┘                                               │
│         │ ExtractedDocument                                     │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  RETRIEVAL   │  ← chunk_text()                               │
│  │              │  ← TFIDFIndex (BM25-lite)                     │
│  │RetrievalStore│ ← retrieve(query, top_k)                      │
│  │  20 passages │  ← retrieve_for_draft(section_queries)        │
│  └──────┬───────┘                                               │
│         │ RetrievedEvidence[]                                   │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  GENERATION  │  ← build_generation_prompt()                  │
│  │              │  ← Claude API (claude-sonnet-4)               │
│  │GeneratedDraft│ ← _parse_llm_sections()                       │
│  │  + citations │  ← _build_evidence_refs()                     │
│         │ GeneratedDraft                                        │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  OPERATOR    │  Human reviews, edits draft                   │
│  │   REVIEW     │  (or simulated edit for demo)                 │
│  └──────┬───────┘                                               │
│         │ edited_markdown                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ IMPROVEMENT  │  ← compute_diff_summary()                     │
│  │              │  ← extract_rules_from_edit()                  │
│  │ImprovementStore ← rules persisted to JSON                    │
│  │  (rules grow)│  ← injected into next generation              │
│  └──────────────┘                                               │
│         │                                                       │
│         └──────────────────────────────────────────────────────┤|
│                     (loops back to GENERATION)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Design Decisions

### 1. Ingestion (`ingestion.py`)

The ingestion layer makes a clear distinction between **raw text** and **cleaned text**, preserving both. This matters because:

- Raw text is kept for audit/debugging
- Cleaned text is what retrieval and generation see
- Confidence flags tell downstream stages how much to trust the content

OCR fixes are applied as regex substitutions with a curated set of common legal-document errors (OCR confusion of `0`/`O`, `1`/`l`, smudged pincodes, etc.). Each fix is logged as a warning so operators can inspect what was corrected.

Document type detection uses a keyword scoring approach — simple, transparent, and debuggable. Structured field extraction is type-specific (one extractor per doc type) rather than a generic NER approach, which gives more reliable output for this narrow domain at zero ML overhead.

### 2. Retrieval (`retrieval.py`)

The retrieval layer uses a **BM25-lite TF-IDF index** built from scratch with stdlib. The deliberate choice to avoid FAISS/ChromaDB/sentence-transformers was about:

- Zero dependency overhead
- Full transparency (every scoring step is readable)
- Adequate performance for legal doc workloads (documents are short, vocabulary is specialized)

The `retrieve_for_draft()` method accepts a dict of `{section_name: query}` and runs a retrieval per section. This means each section of the draft is grounded in passages that are relevant to *that specific section's concerns*, not just the document overall.

Passage provenance is maintained through the entire stack — every `RetrievedEvidence` object carries its `passage_id`, `doc_id`, and source path, making citations traceable to exact character offsets in the original document.

### 3. Generation (`generation.py`)

The generation prompt is structured to enforce grounding through three mechanisms:

1. **Explicit rules in the prompt**: "Every factual claim MUST be supported by a specific evidence passage."
2. **Evidence passages included in context**: The retrieved passages are included in the prompt so the model has access to the source material.
3. **Uncertainty acknowledgment**: The model is instructed to say "NOT FOUND IN DOCUMENTS" rather than hallucinate.

The prompt also requests a `GROUNDING_ASSESSMENT: X/10` from the model itself — this is a cheap self-evaluation signal that correlates reasonably with actual citation coverage.

Operator instructions from the improvement loop are injected as a clearly-labeled block in the prompt with the label "OPERATOR STYLE INSTRUCTIONS (learned from prior edits)". This framing encourages the model to follow them consistently.

### 4. Improvement (`improvement.py`)

The improvement loop is the most deliberately designed part of the system. The key insight is that **operator edits encode implicit preferences** that are hard to specify upfront but easy to demonstrate. Rather than asking operators to write style guides, we extract rules from what they actually do.

The rule extractors are named, single-purpose functions (not a black box). This means:
- Each rule type is independently testable
- New patterns can be added without breaking existing ones
- Operators can inspect exactly what rules were learned

Rules are persisted per-document-type (not per-document), so rules learned from reviewing one title deed apply to all future title deed drafts. Rules accumulate over time and deduplicate.

**What this is not**: It's not fine-tuning. It's not RL from human feedback in the ML sense. It's structured prompt modification based on observed behavioral differences. This is appropriate for the scope, transparent, and fast.

---

## Tradeoffs

| Decision | What was traded off |
|----------|---------------------|
| TF-IDF over dense embeddings | Less semantic retrieval quality, but zero setup, transparent scoring, no GPU |
| Regex field extraction over NER | More brittle to format variations, but accurate for known doc types, fast, debuggable |
| Rule extraction over fine-tuning | Slower convergence, but interpretable, no training data needed, works day 1 |
| stdlib over frameworks | No pip install needed; limits some capabilities (no streaming, etc.) |
| Single JSON improvement store | Not suitable for multi-user concurrent writes; fine for single-operator use |

---

## Extension Points

- **OCR**: Drop `pytesseract` + `pdf2image` into the `ingest_document` function's PDF branch
- **Embeddings**: Replace `TFIDFIndex` with a `SentenceTransformer` + `numpy` dot-product store
- **Multi-user**: Replace `improvement_store.json` with SQLite or Postgres
- **Streaming**: Replace `urllib` API call with the `anthropic` Python SDK's stream API
- **UI**: The `pipeline.py` `run_pipeline()` function is already usable as a FastAPI endpoint — add a thin wrapper
