# LexTrace AI — Evidence-Grounded Legal Document Intelligence

LexTrace AI is an AI-powered legal document intelligence system that transforms messy, scanned, and semi-structured legal documents into structured information and evidence-grounded drafts.

The system combines OCR, NLP, hybrid retrieval, and Gemini-powered generation while preserving links between generated content and its source evidence.

> **Note:** LexTrace AI is an assistive document-intelligence system. Generated outputs are drafts and require professional legal verification. The system does not provide legal advice.

## 🌐 Live Demo

**Frontend:** [https://lextraceai.vercel.app/]

**Backend API:** [https://lextrace-ai-api.onrender.com]

**API Documentation:** [https://lextrace-ai-api.onrender.com/docs]

---

## 🚀 Key Features

- **Legal Document Ingestion** — Processes PDF and text-based legal documents with OCR, preprocessing, metadata extraction, and page tracking.
- **Structured Information Extraction** — Extracts parties, dates, financial amounts, case references, disputes, missing documents, and other relevant fields.
- **Uncertainty Detection** — Flags illegible, damaged, ambiguous, or missing information instead of silently generating values.
- **Hybrid Retrieval** — Combines BM25 lexical retrieval with Sentence Transformer semantic search.
- **Evidence-Grounded Generation** — Uses Gemini to generate structured drafts grounded in retrieved source evidence.
- **Source Citations** — Links retrieved evidence to source documents, pages, and chunks.
- **Human-in-the-Loop Review** — Captures operator corrections and converts them into reusable generation rules.
- **Evaluation** — Provides development-time evaluation of retrieval, extraction, grounding, and generation.
- **Full-Stack Application** — React frontend with a FastAPI backend.

---

## 🏗️ System Architecture

```text
Document Input
      ↓
Document Ingestion
(PDF / OCR / Preprocessing)
      ↓
Legal Information Extraction
(Entities / Dates / Parties / Fields)
      ↓
Hybrid Retrieval
(BM25 + Semantic Search)
      ↓
Evidence-Grounded Generation
(Gemini API)
      ↓
Human Review
(Review / Corrections)
      ↓
Evaluation & Improvement
```

---

## 🛠️ Tech Stack

### Backend
Python | FastAPI | REST APIs

### AI / NLP
Gemini API | NLP | OCR | RAG | BM25 | Sentence Transformers | Hybrid Retrieval

### Document Processing
PDF Processing | Structured Information Extraction | Page-aware Chunking | Evidence Mapping

### Frontend
React | Vite | JavaScript

### Deployment & Development
Git | GitHub | Vercel | Render

---

## 📂 Project Structure

```text
LexTrace-AI/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── package-lock.json
│
├── samples/
│
├── src/
│   ├── generation/
│   │   └── generation.py
│   ├── improvement/
│   │   └── improvement.py
│   ├── ingestion/
│   │   └── ingestion.py
│   ├── retrieval/
│   │   └── retrieval.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── tests/
│   └── test_pipeline.py
│
├── ARCHITECTURE.md
├── ASSUMPTIONS_TRADEOFFS.md
├── README.md
├── SAMPLE_INPUTS_OUTPUTS.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jethasriM/AI_Legal_Document_Intelligence_System.git
cd AI_Legal_Document_Intelligence_System/LexTrace-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file inside the `LexTrace-AI` directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Do not commit your API key to GitHub.

---

## ▶️ Run the Backend

From the `LexTrace-AI` directory:

```bash
python -m uvicorn main:app --reload --app-dir app
```

Backend:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 💻 Run the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

For local development, configure:

```env
VITE_API_URL=http://127.0.0.1:8000
```

For production, set `VITE_API_URL` to your deployed FastAPI backend URL.

---

## 🔄 Document Processing Pipeline

### 1. Ingestion

Documents are loaded and normalized before their contents are extracted.

The ingestion layer tracks:

- Document type
- Source path
- Raw and cleaned text
- Page count
- OCR usage
- Warnings
- Structured fields
- Field-level evidence

### 2. Legal Information Extraction

Relevant legal information is extracted into structured fields, including:

```text
Case Reference
Parties
Dates
Financial Amounts
Nature of Dispute
Missing Documents
Legal Notice Demands
Property Details
```

Unclear information is explicitly marked instead of being treated as verified.

### 3. Hybrid Retrieval

Documents are divided into page-aware passages.

```text
BM25 Retrieval
      +
Semantic Retrieval
      ↓
Hybrid Ranking
```

Semantic retrieval uses Sentence Transformers when available, with BM25 serving as a lexical retrieval fallback.

### 4. Grounded Generation

Retrieved evidence and structured extraction results are provided to Gemini to generate the requested draft.

The generation process is designed to:

- Ground factual claims in retrieved evidence.
- Preserve uncertain values.
- Identify missing information.
- Avoid unsupported legal claims.
- Maintain the requested document structure.

### 5. Human Review

Generated drafts can be reviewed and corrected by an operator.

Operator corrections can be converted into reusable generation rules that influence subsequent drafts.

### 6. Evaluation

The pipeline records retrieval, grounding, extraction, and generation information for development-time evaluation.

---

## 📊 Demonstration Dataset

The project includes sample legal documents representing:

| Document Type | Example |
|---|---|
| Case Intake | Apartment purchase dispute |
| Legal Notice | Rental / tenancy dispute |
| Title Deed | Property transaction |

The demonstration dataset contains **3 document types** and the retrieval pipeline can index **20 passages** across the sample documents.

---

## 🔎 Evidence & Uncertainty Handling

A core design principle of LexTrace AI is distinguishing between verified and uncertain information.

```text
Source Evidence
      ↓
Field Extraction
      ↓
Evidence Matching
      ↓
┌───────────────┬────────────────┐
│   Verified    │    Uncertain   │
│               │                │
│ Source found  │ Illegible /    │
│ and linked    │ damaged /      │
│               │ ambiguous      │
└───────────────┴────────────────┘
```

Evidence records can contain:

```text
Status
Value
Page
Source Excerpt
Reason
```

This allows reviewers to identify information that requires additional verification.

---

## 📌 Example Output

Generated drafts can contain structured sections such as:

```text
CASE SUMMARY

PARTIES

NATURE OF DISPUTE

KEY FACTS

FINANCIAL SUMMARY

MISSING DOCUMENTS

NEXT ACTIONS
```

Supporting evidence can be referenced using source citations such as:

```text
[Source: legal_notice_002.txt, page 1, chunk 3]
```

---

## 🔐 Environment Variables

### Backend

```env
GEMINI_API_KEY=your_gemini_api_key
```

### Frontend

```env
VITE_API_URL=https://your-backend-url
```

`VITE_API_URL` is a public configuration value.

API credentials such as `GEMINI_API_KEY` must remain on the backend and should never be exposed through frontend code.

---

## 🚀 Deployment

LexTrace AI uses a split deployment architecture:

```text
React + Vite
      │
      ▼
   Vercel
      │
      │ REST API
      ▼
FastAPI Backend
      │
      ▼
   Render
```

### Frontend — Vercel

Deploy the `frontend` directory to Vercel.

Set:

```text
VITE_API_URL=https://your-render-backend-url
```

### Backend — Render

Set the root directory to:

```text
LexTrace-AI
```

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port $PORT --app-dir app
```

Environment variable:

```text
GEMINI_API_KEY=your_secret_key
```

---

## 🧪 Testing

Run the pipeline:

```bash
python -m src.pipeline
```

Run tests:

```bash
pytest
```

---

## 📚 Documentation

Additional technical documentation:

- `ARCHITECTURE.md` — System architecture and component design
- `ASSUMPTIONS_TRADEOFFS.md` — Engineering assumptions and design trade-offs
- `SAMPLE_INPUTS_OUTPUTS.md` — Sample inputs and generated outputs

---

## ⚠️ Limitations

- OCR quality depends on source document quality.
- Poorly scanned or damaged documents may produce uncertain extraction.
- Retrieval quality depends on document content and available semantic models.
- Generated drafts require human verification.
- Development-time grounding scores should not be interpreted as production accuracy benchmarks.
- The system does not provide legal advice or replace professional legal review.

---

## 🔮 Future Improvements

- Legal-domain embedding models
- Improved handwritten-document OCR
- Advanced entity and clause extraction
- Automated citation verification
- Larger evaluation datasets
- Authentication and role-based access
- Persistent document storage
- Asynchronous processing for large document collections
- Advanced human-review workflows
- Production monitoring and observability

---

## 👩‍💻 Author

**Jethasri Muvvala**

AI/ML Engineer | Generative AI | NLP | Computer Vision | Backend Development

[GitHub](https://github.com/jethasriM)

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.
