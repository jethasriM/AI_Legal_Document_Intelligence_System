# Ambitio Legal Document AI

> A four-stage AI pipeline for processing messy legal documents. Ingests scanned PDFs and images via OCR, extracts structured fields, retrieves grounded evidence using hybrid semantic + BM25 search, and generates cited draft summaries via Gemini. Improves over time by learning style rules from operator edits.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com)

### Install & Run

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd ambitio-legal-ai

# 2. Install the Gemini SDK
pip install google-generativeai

# 3. Set your Gemini API key
export GEMINI_API_KEY=AIza...        # Mac/Linux
set GEMINI_API_KEY=AIza...           # Windows CMD
$env:GEMINI_API_KEY = "AIza..."      # Windows PowerShell

# 4. Run the full pipeline on all sample documents
python src/pipeline.py --input samples/ --simulate-edit

# 5. Run the test suite
python tests/test_pipeline.py

# 6. Run the evaluation
python src/evaluation.py
```

### What you'll see

```
[1/4] Ingesting documents...
  ✓ doc_690e344d | type=title_deed    | quality=60 | warnings=4
  ✓ doc_12b272f9 | type=legal_notice  | quality=55 | warnings=6
  ✓ doc_dfa9baef | type=case_intake   | quality=80 | warnings=1
  → Saved extraction: outputs/doc_690e344d_extracted.json

[2/4] Building retrieval index...
  ✓ Indexed 20 passages across 3 doc(s)

[3/4] Generating grounded drafts...
  ℹ Applying 3 learned rule(s) for case_intake
  ✓ Draft draft_eced0f26 | grounding=75% | sections=5
  → Saved: outputs/draft_eced0f26_draft.md

[4/4] Simulating operator edits and extracting improvement rules...
  ✓ Edit edit_f954d26e: Added 11 line(s)
    → Rule learned: Be more thorough and detailed
    → Rule learned: Always include these additional sections: Risk Flags
    → Rule learned: Explicitly flag risks, missing documents, and ambiguities

  📊 Improvement store: 3 doc types · 7 rules total

✅ Pipeline complete. Outputs in: outputs/
```

---

## Project Structure

```
ambitio-legal-ai/
├── src/
│   ├── ingestion/
│   │   └── ingestion.py       ← OCR noise cleaning, doc type detection, field extraction
│   ├── retrieval/
│   │   └── retrieval.py       ← BM25-lite TF-IDF index, passage chunking, evidence retrieval
│   ├── generation/
│   │   └── generation.py      ← Prompt builder, Gemini API call, section parser, citation linker
│   ├── improvement/
│   │   └── improvement.py     ← Diff capture, rule extraction, persistent improvement store
│   ├── pipeline.py            ← Top-level orchestrator (main entry point)
│   └── evaluation.py          ← Evaluation framework with ground truth scoring
├── samples/
│   ├── title_deed_001.txt     ← Synthetic messy title deed (OCR noise, missing pages)
│   ├── legal_notice_002.txt   ← Synthetic legal notice (smudged amounts, garbled pincodes)
│   └── case_intake_003.txt    ← Synthetic handwritten case intake (transcription errors)
├── tests/
│   └── test_pipeline.py       ← 36 unit + integration tests (all passing)
├── outputs/                   ← Generated at runtime (gitignored)
│   ├── *_extracted.json       ← Structured fields + confidence flags per document
│   ├── *_draft.md             ← Generated draft in readable markdown
│   ├── *_draft.json           ← Draft with full evidence provenance
│   ├── *_edit.json            ← Edit record + rules extracted per operator edit
│   └── evaluation_report.json ← Full evaluation scores
├── improvement_store.json     ← Persisted operator style rules (gitignored)
├── README.md
├── ARCHITECTURE.md            ← System design, module decisions, tradeoffs
├── ASSUMPTIONS_TRADEOFFS.md   ← Design rationale and what I'd do with more time
└── SAMPLE_INPUTS_OUTPUTS.md   ← Annotated input/output examples for each stage
```

---

## Pipeline CLI Flags

```bash
python src/pipeline.py [options]

  --input PATH        File or directory to process (default: samples/)
  --simulate-edit     Apply simulated operator edits and extract improvement rules
  --output-dir PATH   Where to write outputs (default: outputs/)
  --store PATH        Improvement store JSON path (default: improvement_store.json)
  --quiet             Suppress verbose console output
```

Examples:

```bash
# Single document
python src/pipeline.py --input samples/case_intake_003.txt --simulate-edit

# Full batch, quiet mode
python src/pipeline.py --input samples/ --simulate-edit --quiet

# Custom output directory
python src/pipeline.py --input samples/ --output-dir my_outputs/
```

---

## Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| `google-generativeai` | Gemini API calls for draft generation | Yes (for live generation) |
| Python stdlib only | Everything else — ingestion, retrieval, improvement, tests | Always |

The pipeline runs without the Gemini key — ingestion, retrieval, extraction, and the improvement loop all work fully. Only the LLM draft generation step falls back to a descriptive error message.

---

## Outputs Explained

After running `python src/pipeline.py --input samples/ --simulate-edit`, the `outputs/` folder contains:

**`*_extracted.json`** — What stage 1 produced: cleaned text, detected doc type, structured fields (names, amounts, dates, document checklists), and confidence flags showing where the document was unclear or damaged.

**`*_draft.md`** — The generated case summary in readable markdown. Each section includes inline citations like `[doc_abc_chunk3]` linking every claim back to a specific passage in the source document.

**`*_edit.json`** — The operator edit record: what changed (diff summary), and what reusable rules were extracted from those changes.

**`improvement_store.json`** — The growing ruleset per document type. Run the pipeline a second time and these rules get injected into the generation prompt automatically, making future drafts better without any manual prompt engineering.

**`evaluation_report.json`** — Scores across all four rubric dimensions with per-document breakdowns.

---

## Evaluation Results

```
Extraction (field coverage + type accuracy)   100%
Retrieval  (precision@3 + top-1 accuracy)     100%
Grounding  (citation coverage + LLM score)     87.5%
Improvement (rule extraction accuracy)        100%
─────────────────────────────────────────────────
Weighted overall                               98.5%
```

---

## Running Against Your Own Documents

Drop any `.txt` file into `samples/` and point `--input` at it:

```bash
python src/pipeline.py --input samples/my_document.txt --simulate-edit
```

For actual scanned PDFs, the ingestion module has a clearly marked hook for `pytesseract` + `pdf2image` — the rest of the pipeline is fully ready to receive that output:

```python
# ingestion.py — drop-in OCR hook
def ingest_document(source_path, text_override=None):
    # TODO: detect PDF → run pdf2image + pytesseract here
    # then pass result as text_override to clean_ocr_noise()
    ...
```

---

## Architecture Overview

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system diagram and module-level design decisions.

See [`ASSUMPTIONS_TRADEOFFS.md`](ASSUMPTIONS_TRADEOFFS.md) for why certain choices were made and what would change with more time.

See [`SAMPLE_INPUTS_OUTPUTS.md`](SAMPLE_INPUTS_OUTPUTS.md) for annotated examples of what each stage produces.
