# src/ocr.py
# PDF text extraction with OCR fallback

import pymupdf
import os


def extract_text_with_ocr(file_path: str, page_num: int = 0) -> str:
    """
    Extract text from a specific page of a PDF file.
    Falls back to OCR if PDF is image-based.
    
    Args:
        file_path: Path to the PDF file
        page_num: Page number (0-indexed). Default is first page.
        
    Returns:
        Extracted text as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no text could be extracted or page doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = pymupdf.open(file_path)
    
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} doesn't exist. PDF has {len(doc)} pages.")
    
    page = doc[page_num]

    # Try standard text extraction first
    text = page.get_text()
    
    # If empty, use OCR
    if not text.strip():
        text = page.get_text("text", textpage=page.get_textpage_ocr())
    
    if not text.strip():
        raise ValueError(f"Could not extract text from page {page_num}")
    
    return text


def get_page_count(file_path: str) -> int:
    """Return total number of pages in PDF."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    doc = pymupdf.open(file_path)
    return len(doc)


def extract_pages_range(file_path: str, start_page: int, end_page: int) -> list[str]:
    """
    Extract text from a range of pages.
    
    Args:
        file_path: Path to the PDF file
        start_page: First page (0-indexed)
        end_page: Last page (inclusive)
        
    Returns:
        List of extracted text strings, one per page
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    doc = pymupdf.open(file_path)
    texts = []
    
    for page_num in range(start_page, min(end_page + 1, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        
        if not text.strip():
            text = page.get_text("text", textpage=page.get_textpage_ocr())
        
        texts.append(text)
    
    return texts