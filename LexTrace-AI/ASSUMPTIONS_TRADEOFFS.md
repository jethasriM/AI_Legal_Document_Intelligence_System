# Assumptions & Tradeoffs

## What I assumed going in

**On the document domain**: I scoped the system to three concrete legal document types found commonly in Indian property/tenancy law contexts: title deeds, legal notices, and case intake forms. These cover the three most common "messy legal document" scenarios in a property law practice. The system has an `unknown` fallback for anything else.

**On "messy inputs"**: I interpreted this as primarily OCR-induced noise in already-digitized or scanned text — character substitutions (0/O, 1/l), garbled pincodes, smudged amounts, partial page damage, and handwriting transcription inaccuracies. For an actual production system, this would be the pre-processing stage before real Tesseract OCR; here I simulate what that output would look like.

**On the improvement loop**: I read "real improvement loop, not a side-by-side version diff" as meaning the system needs to *extract reusable behavioral patterns* from edits, not just store the edited version. So instead of "here's the new draft to use as a template", the system asks "what did the operator consistently do, and can I turn that into a rule?" This is a softer form of learning but it's interpretable and doesn't require training data.

**On legal correctness**: As stated in the brief — not evaluating this. The system intentionally refuses to generate legal conclusions that aren't in the source material.

---

## Key tradeoffs made

### TF-IDF vs. dense semantic search

I went with a BM25-lite TF-IDF index rather than sentence embeddings. For legal documents, this is a reasonable call: legal language is precise and keyword-heavy, so lexical search works well. Semantic search would add meaningful value for paraphrastic queries ("how much is owed" vs "outstanding dues"), but requires a model, a GPU or API call, and more complexity. The tradeoff: slightly lower recall on semantically-similar queries, but fully transparent scoring and zero setup.

### Regex extraction vs. NER/LLM extraction

Field extraction is done with handwritten regex per document type. This is deliberately fragile in one dimension (it breaks if format changes) but robust in another (no false positives, fast, auditable). An LLM-based extractor would generalize better but would add latency and cost to every ingestion run. Since legal documents from a given practice tend to follow consistent templates, regex is the right first choice.

### Rule-based improvement vs. fine-tuning

The improvement system extracts explicit style rules from diffs rather than anything that touches model weights. This has real limitations — it can't capture nuanced stylistic changes that are hard to verbalize as a rule — but it's honest, interpretable, and doesn't require data collection over time before it starts working. A fine-tuned model trained on `(original_draft, edited_draft)` pairs would eventually outperform this, but that's a month-2 project, not a day-1 prototype.

### Single-file JSON store vs. database

The improvement store is a flat JSON file. This works fine for a single-operator workflow where edits happen sequentially. It would break under concurrent multi-user writes (race condition on the file). Replacing it with SQLite is a 20-line change. I chose JSON for zero setup and easy inspection.

### No streaming

The LLM call is synchronous. For a production API this would be unacceptable UX (5-10 second wait). The fix is a one-line change to the `anthropic` SDK's streaming API. I left it synchronous to keep the code readable and the HTTP layer dependency-free.

---

## What I'd do with more time

1. **Real OCR integration** — Wire in `pytesseract` + `pdf2image` for actual scanned PDF input. The rest of the pipeline is already ready for it.

2. **Embedding-based retrieval** — Add a `SentenceTransformerIndex` as an optional drop-in alongside the TF-IDF index, with a flag to switch between them for comparison.

3. **UI** — A minimal FastAPI + React frontend where operators can upload documents, see the generated draft, make inline edits, and submit — with the edit automatically processed by the improvement loop.

4. **Multi-pass generation** — For long documents, retrieve per-section, generate per-section, then do a final consistency pass. The current single-pass approach works but can produce inconsistencies across sections for complex documents.

5. **Confidence-aware generation** — Currently the `[ILLEGIBLE]` markers in cleaned text flow into the LLM context and the model is instructed to flag them. A better approach would suppress those passages from retrieval entirely and instead generate a "data gap" section that explicitly lists what was unreadable.

6. **Rule versioning** — Track which rule was introduced by which edit, so rules can be individually reverted if an operator changes their mind.
