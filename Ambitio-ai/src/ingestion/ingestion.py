"""
ingestion.py — Document ingestion and text extraction pipeline.

Handles messy inputs: scanned PDFs, image files, noisy OCR text, partially
illegible files, and plain text documents.

PDF/image pipeline:
  1. pdf2image converts PDF pages → PIL images
  2. Pillow pre-processes each image (deskew, denoise, contrast boost)
  3. pytesseract runs OCR → raw text per page
  4. Pages are joined and passed through OCR noise cleaning
  5. Structured fields are extracted

Plain text pipeline:
  1. Read file as UTF-8
  2. Apply OCR noise cleaning (handles pre-extracted noisy text)
  3. Extract structured fields

Outputs a clean ExtractedDocument ready for retrieval and drafting.
"""

import re
import json
import hashlib
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedDocument:
    doc_id: str
    source_path: str
    doc_type: str                    # "title_deed" | "legal_notice" | "case_intake" | "unknown"
    raw_text: str
    cleaned_text: str
    structured_fields: dict
    confidence_flags: dict
    ingested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    char_count: int = 0
    warnings: list = field(default_factory=list)
    ocr_used: bool = False           # True if pytesseract was used
    page_count: int = 1

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Image pre-processing for better OCR accuracy
# ---------------------------------------------------------------------------

def _preprocess_image_for_ocr(img):
    """
    Apply image enhancement steps that significantly improve Tesseract accuracy
    on scanned legal documents:
      - Convert to grayscale
      - Boost contrast (helps with faded ink)
      - Binarize with a threshold (removes background noise)
      - Slight sharpening (helps with blurry scans)
    """
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps

        # Grayscale
        img = img.convert("L")

        # Contrast boost — critical for low-quality scans
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Sharpness boost
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)

        # Binarize — convert to pure black/white
        img = img.point(lambda x: 0 if x < 140 else 255, "1")

        # Convert back to RGB for Tesseract
        img = img.convert("RGB")

        return img
    except Exception:
        return img  # return original if enhancement fails


# ---------------------------------------------------------------------------
# OCR extraction from PDF or image
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> tuple[str, int, list]:
    """
    Convert a PDF to images and run Tesseract OCR on each page.
    Returns (full_text, page_count, warnings).
    """
    warnings = []
    pages_text = []

    try:
        from pdf2image import convert_from_path
        import pytesseract

        # Convert PDF pages to PIL images at 300 DPI (optimal for OCR)
        images = convert_from_path(pdf_path, dpi=300)
        warnings.append(f"PDF converted to {len(images)} page image(s) at 300 DPI")

        for i, img in enumerate(images):
            # Pre-process for better accuracy
            processed = _preprocess_image_for_ocr(img)

            # Run Tesseract with legal document config:
            # --oem 3 = LSTM engine (most accurate)
            # --psm 6 = assume a uniform block of text
            config = "--oem 3 --psm 6"
            page_text = pytesseract.image_to_string(processed, config=config, lang="eng")

            if not page_text.strip():
                warnings.append(f"Page {i+1}: OCR returned empty text — page may be blank or image quality too low")
            else:
                pages_text.append(f"[PAGE {i+1}]\n{page_text.strip()}")

        full_text = "\n\n".join(pages_text)
        return full_text, len(images), warnings

    except ImportError as e:
        warnings.append(f"OCR dependency missing: {e}. Install with: pip install pdf2image pytesseract")
        return "", 0, warnings
    except Exception as e:
        warnings.append(f"PDF OCR failed: {str(e)[:120]}")
        return "", 0, warnings


def extract_text_from_image(image_path: str) -> tuple[str, list]:
    """
    Run Tesseract OCR directly on an image file (PNG, JPG, TIFF, etc).
    Returns (text, warnings).
    """
    warnings = []
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(image_path)
        processed = _preprocess_image_for_ocr(img)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, config=config, lang="eng")

        if not text.strip():
            warnings.append("Image OCR returned empty text — check image quality")

        return text, warnings

    except ImportError as e:
        warnings.append(f"OCR dependency missing: {e}")
        return "", warnings
    except Exception as e:
        warnings.append(f"Image OCR failed: {str(e)[:120]}")
        return "", warnings


# ---------------------------------------------------------------------------
# OCR noise cleaning (applies to both pre-extracted text and OCR output)
# ---------------------------------------------------------------------------

OCR_FIXES = [
    (r'\b0(?=[a-zA-Z])', 'O'),
    (r'(?<=[a-zA-Z])0\b', 'o'),
    (r'\bl(?=\d)', '1'),
    (r'(?<=\d)l\b', '1'),
    (r'5OOO', '5000'),
    (r'(?<!\d)5O(?=\d{2}\b)', '50'),
    (r'2O(?=\d{2})', '20'),
    (r'\bRs\.\s+(\d)', r'Rs. \1'),
    (r'\bdt\.?\s*:', 'Date:'),
    (r'\[Page \d+ partially.*?\]', '[PAGE_DAMAGED]'),
    (r'\[.*?illegible.*?\]', '[ILLEGIBLE]'),
    (r'\[.*?unclear.*?\]', '[UNCLEAR]'),
    (r'\[.*?missing.*?\]', '[MISSING]'),
    (r'\?\?\?', '[ILLEGIBLE]'),
    # Common Tesseract errors in legal docs
    (r'\bS[/\\]o\b', 'S/o'),          # "S\o" → "S/o" (son of)
    (r'\bW[/\\]o\b', 'W/o'),          # "W\o" → "W/o" (wife of)
    (r'\bRs\s*\.\s*(\d)', r'Rs. \1'), # "Rs ." spacing
    (r'(?<=\d),(?=\d{3})\s', r','),   # fix spaces in numbers like "42, 00,000"
    (r'\bNot\b(?=\s*:\s*\d)', 'No.'), # "Plot Not 112" → "Plot No. 112"
    (r'(?<=[A-Z])[|](?=[A-Z])', 'I'), # pipe char mistaken for I
]


def clean_ocr_noise(text: str) -> tuple[str, list]:
    warnings = []
    cleaned = text

    for pattern, replacement in OCR_FIXES:
        matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
        if matches:
            warnings.append(f"OCR fix: '{pattern}' corrected {len(matches)} time(s)")
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    uncertain = re.findall(r'\[(?:ILLEGIBLE|UNCLEAR|MISSING|PAGE_DAMAGED)\]', cleaned)
    if uncertain:
        warnings.append(f"Document has {len(uncertain)} uncertain section(s): {set(uncertain)}")

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = cleaned.strip()

    return cleaned, warnings


# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------

DOC_TYPE_SIGNALS = {
    "title_deed": [
        "deed of conveyance", "vendor", "purchaser", "sale consideration",
        "plot no", "survey no", "encumbrance", "sub-registrar", "conveyance"
    ],
    "legal_notice": [
        "legal notice", "instructed by", "notice for", "dues outstanding",
        "vacate", "tenant", "lease agreement", "demand", "failure to comply"
    ],
    "case_intake": [
        "case ref", "intake", "opposing party", "nature of dispute",
        "client information", "next steps", "preliminary legal", "rera"
    ],
}


def detect_doc_type(text: str) -> str:
    text_lower = text.lower()
    scores = {dtype: 0 for dtype in DOC_TYPE_SIGNALS}
    for dtype, signals in DOC_TYPE_SIGNALS.items():
        for signal in signals:
            if signal in text_lower:
                scores[dtype] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "unknown"


# ---------------------------------------------------------------------------
# Structured field extractors (per doc type)
# ---------------------------------------------------------------------------

def _extract_money(text: str) -> list:
    return re.findall(r'Rs\.?\s*[\d,]+(?:/\-)?(?:\s*\([^)]+\))?', text)


def _extract_dates(text: str) -> list:
    patterns = [
        r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
        r'\d{1,2}(?:st|nd|rd|th)?\s+\w+[,\s]+\d{4}',
        r'\w+\s+\d{4}',
    ]
    dates = []
    for p in patterns:
        dates.extend(re.findall(p, text))
    return list(set(dates))[:10]


def extract_title_deed_fields(text: str) -> dict:
    fields = {}
    vendor = re.search(r'(?:VENDOR[^:]*:|between[^,\n]*,)\s*([A-Z][a-zA-Z\s]+),\s*S/o', text)
    if vendor:
        fields["vendor_name"] = vendor.group(1).strip()
    purchaser = re.search(r'(?:PURCHASER[^:]*:.*?\n.*?|AND\s+)([A-Z][a-zA-Z\s]+),\s*W/o', text)
    if purchaser:
        fields["purchaser_name"] = purchaser.group(1).strip()
    plot = re.search(r'Plot No\.?:?\s*([^\n,]+)', text)
    if plot:
        fields["plot_number"] = plot.group(1).strip()
    survey = re.search(r'Survey No\.?:?\s*([^\n,]+)', text)
    if survey:
        fields["survey_number"] = survey.group(1).strip()
    area = re.search(r'Area:?\s*([^\n]+)', text)
    if area:
        fields["area"] = area.group(1).strip()
    consideration = re.search(r'Total Sale Consideration:?\s*(Rs\.?[^\n]+)', text)
    if consideration:
        fields["sale_consideration"] = consideration.group(1).strip()
    fields["encumbrances_present"] = bool(
        re.search(r'EXCEPT|encumbrance|dispute|arrears|outstanding', text, re.IGNORECASE)
    )
    fields["missing_documents"] = re.findall(r'([A-Za-z\s]+)\s*[-–]\s*NOT ATTACHED', text)
    fields["attached_documents"] = re.findall(r'([A-Za-z\s\(\)]+)\s*[-–]\s*ATTACHED', text)
    fields["all_amounts"] = _extract_money(text)
    fields["dates_found"] = _extract_dates(text)
    return fields


def extract_legal_notice_fields(text: str) -> dict:
    fields = {}
    to_match = re.search(r'To,\s*\n([^\n]+)\n([^\n]+)', text)
    if to_match:
        fields["notice_recipient"] = to_match.group(1).strip()
    client_match = re.search(r'behalf of my client\s+([A-Z][a-zA-Z\s\.]+),', text)
    if client_match:
        fields["client_name"] = client_match.group(1).strip()
    rent_match = re.search(r'Monthly Rent:?\s*(Rs\.?[^\n]+)', text)
    if rent_match:
        fields["monthly_rent"] = rent_match.group(1).strip()
    deposit_match = re.search(r'Security Deposit[^:]*:?\s*(Rs\.?[^\n]+)', text)
    if deposit_match:
        fields["security_deposit"] = deposit_match.group(1).strip()
    total_match = re.search(r'TOTAL[^:]*:?\s*(Rs\.?[^\n]+)', text)
    if total_match:
        fields["total_dues"] = total_match.group(1).strip()
    demands = re.findall(r'\([a-z]\)\s+([^\n]+)', text)
    if demands:
        fields["demands"] = demands
    violations = re.findall(r'\d+\.\s+([^\n]+)', text)
    if violations:
        fields["violations_alleged"] = violations[:5]
    fields["all_amounts"] = _extract_money(text)
    fields["dates_found"] = _extract_dates(text)
    return fields


def extract_case_intake_fields(text: str) -> dict:
    fields = {}
    client_match = re.search(r'Name:\s*([^\n]+)', text)
    if client_match:
        fields["client_name"] = client_match.group(1).strip()
    opposing_match = re.search(r'OPPOSING PARTY:\s*\n([^\n]+)', text)
    if opposing_match:
        fields["opposing_party"] = opposing_match.group(1).strip()
    nature_match = re.search(r'NATURE OF DISPUTE:\s*\n([^\n]+)', text)
    if nature_match:
        fields["nature_of_dispute"] = nature_match.group(1).strip()
    case_ref = re.search(r'Case Ref:?\s*([^\n]+)', text)
    if case_ref:
        fields["case_ref"] = case_ref.group(1).strip()
    present = re.findall(r'\[x\]\s+([^\n]+)', text)
    missing = re.findall(r'\[ \]\s+([^\n]+)', text)
    fields["documents_present"] = present
    fields["documents_missing"] = missing
    fields["all_amounts"] = _extract_money(text)
    fields["dates_found"] = _extract_dates(text)
    next_steps = re.findall(r'\d+\.\s+((?:Obtain|Verify|Draft|Consider)[^\n]+)', text)
    if next_steps:
        fields["next_steps"] = next_steps
    return fields


EXTRACTORS = {
    "title_deed": extract_title_deed_fields,
    "legal_notice": extract_legal_notice_fields,
    "case_intake": extract_case_intake_fields,
    "unknown": lambda text: {
        "raw_amounts": _extract_money(text),
        "dates": _extract_dates(text),
    },
}


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

def analyze_confidence(text: str, ocr_used: bool = False) -> dict:
    flags = {}
    flags["has_ocr_artifacts"] = bool(re.search(r'[0O]{2,}|[1l]{2,}(?=\d)', text))
    flags["has_damage_markers"] = bool(
        re.search(r'\[(?:PAGE_DAMAGED|ILLEGIBLE|UNCLEAR|MISSING)\]', text)
    )
    flags["has_partial_amounts"] = bool(re.search(r'Rs\.?\s*[\d,]*\?+', text))
    flags["missing_attachments"] = bool(
        re.search(r'NOT ATTACHED|MISSING', text, re.IGNORECASE)
    )
    flags["ocr_used"] = ocr_used

    penalty = sum([
        20 if flags["has_ocr_artifacts"] else 0,
        15 if flags["has_damage_markers"] else 0,
        10 if flags["has_partial_amounts"] else 0,
        5  if flags["missing_attachments"] else 0,
        5  if ocr_used else 0,       # small penalty for OCR — inherently less reliable
    ])
    flags["quality_score"] = max(0, 100 - penalty)
    return flags


# ---------------------------------------------------------------------------
# File type routing
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS   = {".pdf"}
TEXT_EXTENSIONS  = {".txt", ".md", ".text"}


def _detect_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TEXT_EXTENSIONS:
        return "text"
    # Try to sniff PDF magic bytes for files without extension
    try:
        with open(path, "rb") as f:
            if f.read(4) == b"%PDF":
                return "pdf"
    except Exception:
        pass
    return "text"  # fallback


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_document(
    source_path: str,
    text_override: Optional[str] = None,
) -> ExtractedDocument:
    """
    Ingest a document from path.

    Routing:
      .pdf            → pdf2image + pytesseract OCR → noise cleaning → extraction
      .png/.jpg/etc   → pytesseract OCR → noise cleaning → extraction
      .txt/.md        → read as text → noise cleaning → extraction
      text_override   → use provided text directly (for testing)

    Returns a fully populated ExtractedDocument.
    """
    path = Path(source_path)
    ocr_used = False
    page_count = 1
    all_warnings = []

    if text_override:
        raw_text = text_override
        file_type = "text"
    else:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source_path}")

        file_type = _detect_file_type(path)

        if file_type == "pdf":
            raw_text, page_count, ocr_warnings = extract_text_from_pdf(str(path))
            all_warnings.extend(ocr_warnings)
            ocr_used = True
            if not raw_text.strip():
                # PDF might have embedded text — try reading directly as fallback
                try:
                    raw_text = path.read_text(encoding="utf-8", errors="replace")
                    ocr_used = False
                    all_warnings.append("PDF OCR produced no text; fell back to embedded text extraction")
                except Exception:
                    raw_text = ""

        elif file_type == "image":
            raw_text, ocr_warnings = extract_text_from_image(str(path))
            all_warnings.extend(ocr_warnings)
            ocr_used = True

        else:
            # Plain text
            raw_text = path.read_text(encoding="utf-8", errors="replace")

    # Clean OCR noise (applies to all paths)
    cleaned_text, clean_warnings = clean_ocr_noise(raw_text)
    all_warnings.extend(clean_warnings)

    # Detect type
    doc_type = detect_doc_type(cleaned_text)

    # Extract structured fields
    structured_fields = EXTRACTORS[doc_type](cleaned_text)

    # Confidence analysis
    confidence_flags = analyze_confidence(cleaned_text, ocr_used=ocr_used)

    # Stable doc_id from content hash
    doc_id = "doc_" + hashlib.md5(raw_text.encode()).hexdigest()[:8]

    return ExtractedDocument(
        doc_id=doc_id,
        source_path=str(source_path),
        doc_type=doc_type,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        structured_fields=structured_fields,
        confidence_flags=confidence_flags,
        ingested_at=datetime.utcnow().isoformat(),
        char_count=len(cleaned_text),
        warnings=all_warnings,
        ocr_used=ocr_used,
        page_count=page_count,
    )


def ingest_directory(dir_path: str) -> list[ExtractedDocument]:
    """
    Ingest all supported files in a directory.
    Supported: .txt, .pdf, .png, .jpg, .jpeg, .tiff, .tif, .bmp
    """
    supported = TEXT_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS
    docs = []
    for p in sorted(Path(dir_path).iterdir()):
        if p.suffix.lower() in supported:
            try:
                docs.append(ingest_document(str(p)))
            except Exception as e:
                print(f"[ingestion] Warning: failed to process {p.name}: {e}")
    return docs


if __name__ == "__main__":
    samples_dir = Path(__file__).parent.parent.parent / "samples"
    docs = ingest_directory(str(samples_dir))
    for d in docs:
        print(f"\n{'='*60}")
        print(f"ID: {d.doc_id}  Type: {d.doc_type}  OCR: {d.ocr_used}  "
              f"Pages: {d.page_count}  Quality: {d.confidence_flags['quality_score']}")
        print(f"Warnings: {d.warnings}")
        print(f"Fields: {json.dumps(d.structured_fields, indent=2)}")
