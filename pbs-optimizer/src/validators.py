# src/validators.py
# Regex validators and cross-check logic

import re
from models.schemas import BidPacketPage
from pydantic import ValidationError


def count_sequences(text: str) -> int:
    """Count SEQ headers in OCR text."""
    pattern = r"SEQ\s+(\d+)"
    matches = re.findall(pattern, text)
    return len(matches)


def count_ttl(text: str) -> int:
    """Count TTL lines in OCR text."""
    pattern = r"TTL\s+([\d.:]+)"
    matches = re.findall(pattern, text)
    return len(matches)


def count_rpt(text: str) -> int:
    """Count RPT times in OCR text."""
    pattern = r"RPT\s+(\d{4})/(\d{4})"
    matches = re.findall(pattern, text)
    return len(matches)


def count_rls(text: str) -> int:
    """Count RLS times in OCR text."""
    pattern = r"RLS\s+(\d{4})/(\d{4})"
    matches = re.findall(pattern, text)
    return len(matches)


def validate_llm_output(llm_result: dict, ocr_text: str) -> dict:
    """
    Cross-check LLM output against regex counts.
    
    Returns:
        dict with 'valid' bool and 'errors' list
    """
    errors = []
    warnings = []
    
    # Count from regex
    expected_seq = count_sequences(ocr_text)
    expected_ttl = count_ttl(ocr_text)
    
    # Count from LLM output
    actual_seq = len(llm_result.get("sequences", []))
    actual_ttl = len([s for s in llm_result.get("sequences", []) if s.get("ttl")])
    
    # Cross-check
    if actual_seq != expected_seq:
        errors.append(f"Sequence count mismatch: LLM returned {actual_seq}, regex found {expected_seq}")
    
    if actual_ttl != expected_ttl:
        warnings.append(f"TTL count mismatch: LLM returned {actual_ttl}, regex found {expected_ttl}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "sequences_expected": expected_seq,
            "sequences_actual": actual_seq,
            "ttl_expected": expected_ttl,
            "ttl_actual": actual_ttl
        }
    }


def validate_schema(llm_result: dict) -> dict:
    """
    Validate LLM output against Pydantic schema.
    
    Returns:
        dict with 'valid' bool and 'errors' list
    """
    try:
        BidPacketPage(**llm_result) # Dictionary unpacking 
        return {"valid": True, "errors": []}
    except ValidationError as e:
        return {"valid": False, "errors": e.errors()}