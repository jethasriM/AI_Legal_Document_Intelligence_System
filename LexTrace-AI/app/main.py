from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import run_pipeline


# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

app = FastAPI(
    title="LexTrace AI",
    description="Evidence-grounded legal document intelligence API",
    version="1.0.0",
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / ".lextrace_uploads"
OUTPUT_DIR = ROOT / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "LexTrace AI",
        "version": "1.0.0",
    }


# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "LexTrace AI",
        "description": "Evidence-grounded legal document intelligence",
        "status": "running",
        "docs": "/docs",
    }


# ------------------------------------------------------------
# Analyze document
# ------------------------------------------------------------
@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    allowed_extensions = {
        ".txt",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
    }

    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    # Create a unique temporary input directory.
    with tempfile.TemporaryDirectory(
        dir=str(UPLOAD_DIR)
    ) as temp_dir:

        temp_path = Path(temp_dir)
        input_file = temp_path / Path(file.filename).name

        with input_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = run_pipeline(
                input_path=str(input_file),
                output_dir=str(OUTPUT_DIR),
                verbose=False,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline execution failed: {exc}",
            ) from exc

    # Convert pipeline outputs to JSON-serializable dictionaries.
    documents = []

    for doc in result.get("docs", []):
        if hasattr(doc, "to_dict"):
            documents.append(doc.to_dict())
        else:
            documents.append(doc)

    drafts = []

    for draft in result.get("drafts", []):
        if hasattr(draft, "to_dict"):
            drafts.append(draft.to_dict())
        else:
            drafts.append(draft)

    # Make sure we actually have a generated draft.
    if not drafts:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed, but no draft was generated.",
        )

    # The current pipeline generates one draft per document.
    draft = drafts[0]

    return {
        "success": True,
        "document": documents[0] if documents else None,
        "draft": draft,
        "sections": draft.get("sections", []),
        "grounding": draft.get("overall_grounding_score", 0),
    }