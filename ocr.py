import os
import fitz  # PyMuPDF
from config import OCR_DPI

# ── Tesseract detection ───────────────────────────────────────────────────────
try:
    from PIL import Image
    import pytesseract
    import io
    import platform

    if platform.system() == "Windows":
        _tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(_tess):
            pytesseract.pytesseract.tesseract_cmd = _tess

    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True

except Exception:
    OCR_AVAILABLE = False


def extract_page_text(page) -> str:
    """
    Return text for a single PyMuPDF page.
    Falls back to Tesseract OCR if native text is too short (scanned page).
    """
    text = page.get_text().strip()
    if len(text) > 50:
        return text

    if not OCR_AVAILABLE:
        return ""

    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def extract_pdf_text(pdf_path: str) -> tuple[str, bool]:
    """
    Extract full text from a PDF file.
    Returns (full_text, ocr_was_used).
    """
    doc = fitz.open(pdf_path)
    pages, ocr_used = [], False

    for page in doc:
        if len(page.get_text().strip()) <= 50 and OCR_AVAILABLE:
            ocr_used = True
        pages.append(extract_page_text(page))

    return "\n".join(pages), ocr_used


def ocr_status_message() -> str:
    """Return a UI-ready status string indicating OCR availability."""
    if OCR_AVAILABLE:
        return "✅ OCR ready (Tesseract found)"
    return (
        "⚠️ **OCR off** — scanned pages will be skipped. "
        "Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) "
        "then `pip install pytesseract Pillow` and restart."
    )