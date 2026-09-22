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
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedDocument:
    doc_id: str
    source_path: str
    doc_type: str
    raw_text: str
    cleaned_text: str
    structured_fields: dict
    confidence_flags: dict
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    char_count: int = 0
    warnings: list = field(default_factory=list)
    ocr_used: bool = False
    page_count: int = 1

    # Page-aware text used by retrieval and evidence citation.
    page_texts: list[str] = field(default_factory=list)
    
    
    # Source evidence for every extracted structured field.
    field_evidence: dict = field(default_factory=dict)

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
    """
    Extract monetary values and normalize common OCR substitutions.

    Examples:
        Rs. 28,OOO/-  -> Rs. 28,000/-
        Rs. 5O,OOO/-  -> Rs. 50,000/-
        Rs. 4,3??/-   -> Rs. 4,3??/-
    """

    pattern = (
        r'Rs\.?\s*'
        r'[0-9Oo][0-9Oo,?]*'
        r'(?:/\-)?'
        r'(?:\s*\([^)]+\))?'
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    cleaned = []

    for value in matches:
        original_value = value.strip()

        value = re.sub(
            r'^(Rs\.?\s*)([0-9Oo,?]+)',
            lambda m: (
                m.group(1)
                + m.group(2).replace("O", "0").replace("o", "0")
            ),
            original_value,
            flags=re.IGNORECASE
        )

        numeric_part = re.search(
            r'Rs\.?\s*([0-9,?]+)',
            value,
            flags=re.IGNORECASE
        )

        if not numeric_part:
            continue

        numeric_value = numeric_part.group(1)

        
        if not re.search(r'\d', numeric_value):
            continue

        
        original_numeric = re.search(
            r'Rs\.?\s*([0-9Oo,?]+)',
            original_value,
            flags=re.IGNORECASE
        )

        if original_numeric:
            original_token = original_numeric.group(1)

            if (
                'o' in original_token.lower()
                and re.fullmatch(r'0+', numeric_value)
            ):
                continue

        cleaned.append(value)

    return list(dict.fromkeys(cleaned))


def _extract_dates(text: str) -> list:
    """
    Extract likely legal-document dates while rejecting common OCR/number
    artifacts such as:
        "for 1450"
        "possibly 2023"
        "6-3-347"
        "4-7/89"

    Supported formats:
        12/05/2005
        14-03-2019
        01.02.2019
        3rd Jan 2024
        22nd September, 2023
        December 2022
        Dec 2023

    Also tolerates the OCR typo "Septmber".
    """

    month_names = (
        r'(?:'
        r'Jan(?:uary)?|'
        r'Feb(?:ruary)?|'
        r'Mar(?:ch)?|'
        r'Apr(?:il)?|'
        r'May|'
        r'Jun(?:e)?|'
        r'Jul(?:y)?|'
        r'Aug(?:ust)?|'
        r'Sep(?:t(?:ember)?)?|'
        r'Septmber|'
        r'Oct(?:ober)?|'
        r'Nov(?:ember)?|'
        r'Dec(?:ember)?'
        r')'
    )

    patterns = [
        # Numeric dates:
        # 12/05/2005
        # 14-03-2019
        # 01.02.2019
        r'\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{4}\b',

        # Dates with ordinal day:
        # 3rd Jan 2024
        # 22nd September, 2023
        (rf'\b\d{{1,2}}(?:st|nd|rd|th)?\s+'
        rf'{month_names}[,\s]+\d{{4}}\b'),

        # Month + year:
        # December 2022
        # Dec 2023
        rf'\b{month_names}\s+\d{{4}}\b',
    ]

    dates = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:
            value = match.strip()

            if value and value not in dates:
                dates.append(value)
            
            filtered_dates = []

    for date in dates:
        if any(
            date.lower() != other.lower()
            and date.lower() in other.lower()
            for other in dates
        ):
            continue

        if date not in filtered_dates:
            filtered_dates.append(date)

    return filtered_dates[:15]


def _normalize_ocr_number(value: str) -> str:
    """
    Normalize common OCR substitutions inside numeric values.

    Examples:
        28,OOO  -> 28,000
        5O,OOO  -> 50,000
        4,3??   -> 4,3??
    """

    value = value.strip()
    
    if re.search(r'\d', value):
        value = re.sub(r'[Oo](?=[\d,Oo?/-])', '0', value)
        value = re.sub(r'(?<=[\d,])([Oo])', '0', value)

    return value


def extract_title_deed_fields(text: str) -> dict:
    fields = {}

    # ---------------------------------------------------------
    # Parties
    # ---------------------------------------------------------

    vendor = re.search(
        r'VENDOR[^:]*:\s*\n?\s*'
        r'([A-Z][a-zA-Z\s]+),\s*S/o',
        text,
        re.IGNORECASE
    )

    if vendor:
        fields["vendor_name"] = vendor.group(1).strip()

    purchaser = re.search(
        r'PURCHASER[^:]*:\s*\n?\s*'
        r'([A-Z][a-zA-Z\s]+),\s*W/o',
        text,
        re.IGNORECASE
    )

    if purchaser:
        fields["purchaser_name"] = purchaser.group(1).strip()

    # ---------------------------------------------------------
    # Property section
    # ---------------------------------------------------------
    # IMPORTANT:
    # Restrict extraction to PROPERTY DETAILS so that
    # "Plot No. 22" in the purchaser's residential address
    # cannot be mistaken for the property plot number.

    property_match = re.search(
        r'PROPERTY DETAILS:(.*?)(?=\n[A-Z][A-Z &]+:|\Z)',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    property_text = property_match.group(1) if property_match else ""

    plot = re.search(
        r'Plot\s+No\.?\s*:\s*([^\n,]+)',
        property_text,
        flags=re.IGNORECASE
    )

    if plot:
        fields["plot_number"] = plot.group(1).strip()

    survey = re.search(
        r'Survey\s+No\.?\s*:\s*([^\n,]+)',
        property_text,
        flags=re.IGNORECASE
    )

    if survey:
        fields["survey_number"] = survey.group(1).strip()

    area = re.search(
        r'Area\s*:\s*([^\n]+)',
        property_text,
        flags=re.IGNORECASE
    )

    if area:
        fields["area"] = area.group(1).strip()

    locality = re.search(
        r'Locality\s*:\s*([^\n]+)',
        property_text,
        flags=re.IGNORECASE
    )

    if locality:
        fields["locality"] = locality.group(1).strip()

    registration_district = re.search(
        r'Registration District\s*:\s*([^\n]+)',
        property_text,
        flags=re.IGNORECASE
    )

    if registration_district:
        fields["registration_district"] = registration_district.group(1).strip()

    sub_registrar = re.search(
        r'Sub-Registrar Office\s*:\s*([^\n]+)',
        property_text,
        flags=re.IGNORECASE
    )

    if sub_registrar:
        fields["sub_registrar_office"] = sub_registrar.group(1).strip()

    # ---------------------------------------------------------
    # Financial information
    # ---------------------------------------------------------

    consideration = re.search(
        r'Total Sale Consideration\s*:\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if consideration:
        fields["sale_consideration"] = (
            _normalize_ocr_number(consideration.group(1).strip())
        )

    advance = re.search(
        r'Advance paid[^:]*:\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if advance:
        fields["advance_paid"] = _normalize_ocr_number(
            advance.group(1).strip()
        )

    balance = re.search(
        r'Balance paid[^:]*:\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if balance:
        fields["balance_paid"] = _normalize_ocr_number(
            balance.group(1).strip()
        )

    # ---------------------------------------------------------
    # Encumbrances / risks
    # ---------------------------------------------------------

    fields["encumbrances_present"] = bool(
        re.search(
            r'EXCEPT|encumbrance|dispute|arrears|outstanding',
            text,
            re.IGNORECASE
        )
    )

    dispute = re.search(
        r'boundary dispute[^\\n]*',
        text,
        flags=re.IGNORECASE
    )

    if dispute:
        fields["boundary_dispute"] = dispute.group(0).strip()

    tax_arrears = re.search(
        r'Municipal tax arrears[^:]*:\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if tax_arrears:
        fields["municipal_tax_arrears"] = _normalize_ocr_number(
            tax_arrears.group(1).strip()
        )

    # ---------------------------------------------------------
    # Documents
    # ---------------------------------------------------------

    fields["missing_documents"] = re.findall(
        r'([A-Za-z][A-Za-z\s()]+?)\s*[-–]\s*NOT ATTACHED',
        text,
        flags=re.IGNORECASE
    )

    attached_matches = re.findall(
        r'^\s*[-•]\s*(.+?)\s*[-–]\s*ATTACHED\s*$',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    fields["attached_documents"] = [
        item.strip()
        for item in attached_matches
        if len(item.strip()) > 2
    ]

    # ---------------------------------------------------------
    # General information
    # ---------------------------------------------------------

    fields["all_amounts"] = _extract_money(text)
    fields["dates_found"] = _extract_dates(text)

    return fields


def extract_legal_notice_fields(text: str) -> dict:
    fields = {}

    # Recipient
    to_match = re.search(
        r'To,\s*\n([^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if to_match:
        fields["notice_recipient"] = to_match.group(1).strip()

    # Client
    client_match = re.search(
        r'behalf of my client\s+([A-Z][a-zA-Z\s.]+?)(?:,|\n)',
        text,
        flags=re.IGNORECASE
    )

    if client_match:
        fields["client_name"] = client_match.group(1).strip()

    # Monthly rent
    rent_match = re.search(
        r'Monthly Rent\s*:\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if rent_match:
        fields["monthly_rent"] = _normalize_ocr_number(
            rent_match.group(1).strip()
        )

    # Security deposit
    deposit_match = re.search(
        r'Security Deposit Paid\s*:\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if deposit_match:
        fields["security_deposit"] = _normalize_ocr_number(
            deposit_match.group(1).strip()
        )

    # Total dues
    total_match = re.search(
        r'TOTAL APPROX\s*:\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if total_match:
        total_value = _normalize_ocr_number(
            total_match.group(1).strip()
        )

        fields["total_dues"] = total_value

        # Explicitly record whether the total contains uncertainty.
        fields["total_dues_uncertain"] = "?" in total_value

    # Individual outstanding amounts
    fields["outstanding_items"] = []

    for match in re.finditer(
        r'-\s*([^:\n]+):\s*(Rs\.?[^\n]+)',
        text,
        flags=re.IGNORECASE
    ):
        description = match.group(1).strip()
        amount = _normalize_ocr_number(match.group(2).strip())

        fields["outstanding_items"].append({
            "description": description,
            "amount": amount,
            "uncertain": "?" in amount
        })

    # Demands
    demands = re.findall(
        r'\([a-z]\)\s+([^\n]+)',
        text,
        flags=re.IGNORECASE
    )

    if demands:
        fields["demands"] = demands

    # Violations
    violations = re.findall(
        r'\d+\.\s+([^\n]+)',
        text
    )

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
    flags["uncertain_values"] = len(
        re.findall(r'Rs\.?\s*[\dO,]*\?+', text, re.IGNORECASE))
    flags["ocr_used"] = ocr_used

    penalty = sum([
        20 if flags["has_ocr_artifacts"] else 0,
        15 if flags["has_damage_markers"] else 0,
        10 if flags["has_partial_amounts"] else 0,
        5 if flags["missing_attachments"] else 0,
        5 if ocr_used else 0,
        min(flags["uncertain_values"] * 5, 15),
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


def split_into_pages(
    cleaned_text: str,
) -> list[str]:
    """
    Split cleaned document text into page-level text blocks.

    OCR-generated PDFs contain explicit markers such as:

        [PAGE 1]
        ...
        [PAGE 2]
        ...

    Plain-text documents do not normally contain page markers,
    so they are treated as a single-page document.
    """

    if not cleaned_text or not cleaned_text.strip():
        return [""]

    # Detect OCR page markers.
    matches = list(
        re.finditer(
            r"\[PAGE\s+(\d+)\]",
            cleaned_text,
            flags=re.IGNORECASE,
        )
    )

    # No page markers → treat as one page.
    if not matches:
        return [cleaned_text.strip()]

    pages = []

    for index, match in enumerate(matches):

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(cleaned_text)

        page_text = cleaned_text[
            start:end
        ].strip()

        pages.append(page_text)

    # Protect against completely empty pages.
    if not pages:
        return [cleaned_text.strip()]

    return pages

def _normalize_evidence_text(value: str) -> str:
    """
    Normalize text for approximate evidence matching.

    This helps match OCR variations such as:
        Rs. 28,OOO/- 
        Rs. 28,000/-

    without changing the actual extracted value.
    """

    value = str(value).lower()

    value = value.replace("o", "0")

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def _find_field_evidence(
    field_name: str,
    field_value,
    page_texts: list[str],
) -> dict:
    """
    Locate the source evidence for an extracted field.

    Returns a structured evidence record containing:

        status:
            verified
            uncertain
            missing

        value:
            extracted value

        page:
            source page

        excerpt:
            supporting text

        reason:
            explanation for uncertainty
    """

    value_string = str(field_value).strip()

    # ------------------------------------------------------------
    # Empty / missing values
    # ------------------------------------------------------------

    if (
        not value_string
        or value_string.lower()
        in {"none", "null", "[]", "{}"}
    ):
        return {
            "status": "missing",
            "value": field_value,
            "page": None,
            "excerpt": None,
            "reason": "No value was extracted.",
        }
        
        # ------------------------------------------------------------
    # Boolean / derived fields
    # ------------------------------------------------------------

    if isinstance(field_value, bool):
        return {
            "status": "verified",
            "value": field_value,
            "page": None,
            "excerpt": None,
            "reason": (
                "Derived field; source evidence should be "
                "linked to the underlying trigger text."
            ),
        }
        
    # ------------------------------------------------------------
    # Lists
    # ------------------------------------------------------------

    if isinstance(field_value, list):

        evidence_items = []

        for item in field_value:

            # ----------------------------------------------------
            # Structured list item
            # ----------------------------------------------------

            if isinstance(item, dict):

                item_records = {}

                for key, value in item.items():

                    # Boolean flags such as "uncertain"
                    # don't need independent source matching.
                    if isinstance(value, bool):
                        continue

                    item_records[key] = _find_field_evidence(
                        key,
                        value,
                        page_texts,
                    )
                statuses = [
                    record.get("status")
                    for record in item_records.values()
                ]

                if "uncertain" in statuses:
                    item_status = "uncertain"
                elif "verified" in statuses:
                    item_status = "verified"
                else:
                    item_status = "missing"

                evidence_items.append({
                    "status": item_status,
                    "value": item,
                    "items": item_records,
                })

            else:

                evidence_items.append(
                    _find_field_evidence(
                        field_name,
                        item,
                        page_texts,
                    )
                )

        statuses = [
            item.get("status")
            for item in evidence_items
        ]

        if "uncertain" in statuses:
            overall_status = "uncertain"
        elif "verified" in statuses:
            overall_status = "verified"
        else:
            overall_status = "missing"

        return {
            "status": overall_status,
            "value": field_value,
            "items": evidence_items,
        }

    # ------------------------------------------------------------
    # Dictionary values
    # ------------------------------------------------------------

    if isinstance(field_value, dict):

        evidence_items = {}

        for key, value in field_value.items():

            evidence_items[key] = (
                _find_field_evidence(
                    key,
                    value,
                    page_texts,
                )
            )

        return {
            "status": "verified",
            "value": field_value,
            "items": evidence_items,
        }

    # ------------------------------------------------------------
    # Detect known uncertainty markers
    # ------------------------------------------------------------

    uncertain_marker = bool(
        re.search(
            r"\?+|"
            r"\[ILLEGIBLE\]|"
            r"\[UNCLEAR\]|"
            r"\[PAGE_DAMAGED\]|"
            r"\[MISSING\]|"
            r"\bsmudged\b|"
            r"\bunclear\b|"
            r"\bpartially\b",
            value_string,
            flags=re.IGNORECASE,
        )
    )

    # ------------------------------------------------------------
    # Search each page
    # ------------------------------------------------------------

    normalized_value = (
        _normalize_evidence_text(
            value_string
        )
    )

    for page_number, page_text in enumerate(
        page_texts,
        start=1,
    ):

        normalized_page = (
            _normalize_evidence_text(
                page_text
            )
        )

        # --------------------------------------------------------
        # Exact normalized match
        # --------------------------------------------------------

        if normalized_value in normalized_page:

            index = normalized_page.find(
                normalized_value
            )

            # Use original page text for the excerpt.
            start = max(
                0,
                index - 120,
            )

            end = min(
                len(page_text),
                index + len(value_string) + 120,
            )

            excerpt = page_text[
                start:end
            ].replace(
                "\n",
                " ",
            )

            return {
                "status": (
                    "uncertain"
                    if uncertain_marker
                    else "verified"
                ),
                "value": field_value,
                "page": page_number,
                "excerpt": excerpt,
                "reason": (
                    "Value contains OCR uncertainty markers."
                    if uncertain_marker
                    else "Value matched source text."
                ),
            }

    # ------------------------------------------------------------
    # Fallback: search for important tokens
    #
    # Useful when the extractor normalized or shortened
    # the value.
    # ------------------------------------------------------------

    tokens = [
        token
        for token in re.findall(
            r"[a-zA-Z0-9]+",
            value_string,
        )
        if len(token) >= 3
    ]

    if tokens:

        best_page = None
        best_count = 0

        for page_number, page_text in enumerate(
            page_texts,
            start=1,
        ):

            normalized_page = (
                _normalize_evidence_text(
                    page_text
                )
            )

            count = sum(
                1
                for token in tokens
                if _normalize_evidence_text(token)
                in normalized_page
            )

            if count > best_count:
                best_count = count
                best_page = page_number

        # Require at least one meaningful token.
        if best_page is not None and best_count > 0:

            page_text = page_texts[
                best_page - 1
            ]

            return {
                "status": "uncertain",
                "value": field_value,
                "page": best_page,
                "excerpt": page_text[:300].replace(
                    "\n",
                    " ",
                ),
                "reason": (
                    "Related source text was found, "
                    "but the extracted value did not "
                    "match exactly."
                ),
            }

    # ------------------------------------------------------------
    # Nothing found
    # ------------------------------------------------------------

    return {
        "status": "missing",
        "value": field_value,
        "page": None,
        "excerpt": None,
        "reason": (
            "Extracted value could not be located "
            "in the source pages."
        ),
    }


def build_field_evidence(
    structured_fields: dict,
    page_texts: list[str],
) -> dict:
    """
    Build source evidence for all structured fields.
    """

    evidence = {}

    for field_name, field_value in (
        structured_fields.items()
    ):

        evidence[field_name] = (
            _find_field_evidence(
                field_name,
                field_value,
                page_texts,
            )
        )

    return evidence


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
    
    # Preserve page-level text for evidence tracing.
    page_texts = split_into_pages(cleaned_text)

    # Keep page count consistent with the actual
    # page-level representation.
    if page_texts:
        page_count = len(page_texts)

    # Detect type
    doc_type = detect_doc_type(cleaned_text)

    # Extract structured fields
    structured_fields = EXTRACTORS[doc_type](cleaned_text)
    
    # Locate every extracted field in the source document.
    field_evidence = build_field_evidence(
    structured_fields,
    page_texts,
    )

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
        ingested_at=datetime.now(timezone.utc).isoformat(),
        char_count=len(cleaned_text),
        warnings=all_warnings,
        ocr_used=ocr_used,
        page_count=page_count,
        page_texts=page_texts,
        field_evidence=field_evidence,
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
