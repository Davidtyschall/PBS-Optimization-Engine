# src/engine.py
# The "brain" of the parsing pipeline
# Orchestrates: OCR → LLM → Validation
# Returns structured result with confidence flags

import time
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
import json
import sys

from ocr import extract_text_with_ocr
from llm.client import parse_full_page
from validators import validate_llm_output
from models.schemas import BidPacketPage
from config import DEFAULT_BID_PACKET
from calendar_resolver import extract_calendar_dates_from_text, extract_sequence_text


# =============================================================================
# RESULT DATA STRUCTURES (Pydantic for type safety)
# =============================================================================

class CostMetrics(BaseModel):
    """Tracks LLM API costs for business visibility."""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    def calculate_cost(self):
        """Calculate cost based on GPT-4o-mini pricing."""
        input_cost = (self.input_tokens / 1_000_000) * 0.15
        output_cost = (self.output_tokens / 1_000_000) * 0.60
        self.estimated_cost_usd = round(input_cost + output_cost, 6)


class TimingMetrics(BaseModel):
    """Tracks performance for UX optimization."""
    ocr_seconds: float = 0.0
    llm_seconds: float = 0.0
    validation_seconds: float = 0.0
    total_seconds: float = 0.0


class ParseResult(BaseModel):
    """
    Container for parsing results.
    
    Uses Pydantic for type safety - wrong types fail immediately.
    """
    success: bool
    data: Optional[dict] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requires_review: bool = False
    confidence: str = "low"
    cost: CostMetrics = Field(default_factory=CostMetrics)
    timing: TimingMetrics = Field(default_factory=TimingMetrics)


# =============================================================================
# MAIN ENGINE FUNCTION
# =============================================================================

def parse_bid_packet(file_path: str, page_num: int = 0, max_retries: int = 1) -> ParseResult:
    """
    Full parsing pipeline: PDF → Structured Data
    
    Args:
        file_path: Path to bid packet PDF
        max_retries: How many times to retry LLM if it fails
        
    Returns:
        ParseResult with data, errors, warnings, and review flags
    """
    
    errors = []
    warnings = []
    requires_review = False
    timing = TimingMetrics()
    cost = CostMetrics()
    total_start = time.time()
    
    # -------------------------------------------------------------------------
    # STEP 1: Extract text from PDF
    # -------------------------------------------------------------------------
    
    print(f"[ENGINE] Starting parse: {file_path}")
    ocr_start = time.time()
    
    try:
        text = extract_text_with_ocr(file_path, page_num=page_num) # Calls the OCR module
        timing.ocr_seconds = round(time.time() - ocr_start, 2)
        print(f"[ENGINE] OCR complete (page {page_num}): {len(text)} chars in {timing.ocr_seconds}s")
        
    except FileNotFoundError:
        return ParseResult(
            success=False,
            errors=[f"File not found: {file_path}"]
        )
    except ValueError as e:
        return ParseResult(
            success=False,
            errors=[f"Could not extract text: {e}"]
        )
    
    # -------------------------------------------------------------------------
    # STEP 2: Parse with LLM
    # -------------------------------------------------------------------------
    
    llm_result = None
    llm_usage = None
    attempts = 0
    llm_start = time.time()
    
    while attempts <= max_retries:
        attempts += 1
        print(f"[ENGINE] LLM attempt {attempts}...")
        
        try:
            llm_result, llm_usage = parse_full_page(text)
            
            # Track tokens
            if llm_usage:
                cost.input_tokens += llm_usage.get("input_tokens", 0)
                cost.output_tokens += llm_usage.get("output_tokens", 0)
            
            # Check for empty response
            if not llm_result:
                warnings.append(f"LLM returned empty (attempt {attempts})")
                continue
            
            # Check for required structure
            if "sequences" not in llm_result:
                warnings.append(f"Missing 'sequences' key (attempt {attempts})")
                llm_result = None
                continue
            
            break  # Success
            
        except Exception as e:
            warnings.append(f"LLM failed (attempt {attempts}): {e}")
            continue
    
    timing.llm_seconds = round(time.time() - llm_start, 2)
    cost.calculate_cost()
    print(f"[ENGINE] LLM complete: {timing.llm_seconds}s, ${cost.estimated_cost_usd}")
    
    # Halt if no result after retries
    if llm_result is None:
        timing.total_seconds = round(time.time() - total_start, 2)
        return ParseResult(
            success=False,
            errors=["LLM parsing failed after all retries"],
            warnings=warnings,
            cost=cost,
            timing=timing
        )
    
    # -------------------------------------------------------------------------
    # STEP 3: Cross-check with Regex
    # -------------------------------------------------------------------------
    
    validation_start = time.time()
    
    cross_check = validate_llm_output(llm_result, text)
    
    if not cross_check["valid"]:
        requires_review = True
        errors.extend(cross_check["errors"])
        print(f"[ENGINE] Cross-check failed: {cross_check['errors']}")
    
    if cross_check["warnings"]:
        warnings.extend(cross_check["warnings"])
        
    # -------------------------------------------------------------------------
    # STEP 3.5: Override calendar_start_dates with regex extraction
    # -------------------------------------------------------------------------
    
    for seq in llm_result.get("sequences", []):
        seq_num = seq.get("seq_num")
        ops_count = seq.get("ops_count", 0)
    
        # Extract this sequence's text from OCR
        seq_text = extract_sequence_text(text, seq_num)
        
        # Get dates via regex
        calendar_result = extract_calendar_dates_from_text(seq_text, ops_count)
        
        # Override LLM's dates with regex dates
        if calendar_result["dates"]:
            seq["calendar_start_dates"] = calendar_result["dates"]
         
    # -------------------------------------------------------------------------
    # STEP 4: Transform with Pydantic (not just validate)
    # -------------------------------------------------------------------------
    
    validated_data = None
    
    try:
        # Transform: dict → typed object → clean dict
        validated_page = BidPacketPage(**llm_result)
        validated_data = validated_page.model_dump()
        print("[ENGINE] Schema validation passed")
        
    except ValidationError as e:
        requires_review = True
        for err in e.errors():
            loc = " → ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "invalid")
            warnings.append(f"Schema: {loc} - {msg}")
        print(f"[ENGINE] Schema validation failed: {len(e.errors())} errors")
        validated_data = llm_result  # Keep raw data
    
    timing.validation_seconds = round(time.time() - validation_start, 2)
    
    # -------------------------------------------------------------------------
    # STEP 5: Determine Confidence
    # -------------------------------------------------------------------------
    
    if requires_review:
        confidence = "low" if errors else "medium"
    else:
        confidence = "high"
    
    # -------------------------------------------------------------------------
    # STEP 6: Return Result
    # -------------------------------------------------------------------------
    
    timing.total_seconds = round(time.time() - total_start, 2)
    print(f"[ENGINE] Complete: {timing.total_seconds}s - {confidence} confidence")
    
    return ParseResult(
        success=True,
        data=validated_data,
        errors=errors,
        warnings=warnings,
        requires_review=requires_review,
        confidence=confidence,
        cost=cost,
        timing=timing
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments FIRST
    page = 0
    if "--page" in sys.argv:
        idx = sys.argv.index("--page")
        page = int(sys.argv[idx + 1])
    
    print("=" * 60)
    print(f"PBS PARSER ENGINE - TEST RUN (Page {page})")
    print("=" * 60)
    
    # Run parser with page number
    result = parse_bid_packet(str(DEFAULT_BID_PACKET), page_num=page)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nSuccess: {result.success}")
    print(f"Confidence: {result.confidence}")
    print(f"Requires Review: {result.requires_review}")
    
    print(f"\n--- Timing ---")
    print(f"  OCR: {result.timing.ocr_seconds}s")
    print(f"  LLM: {result.timing.llm_seconds}s")
    print(f"  Validation: {result.timing.validation_seconds}s")
    print(f"  Total: {result.timing.total_seconds}s")
    
    print(f"\n--- Cost ---")
    print(f"  Tokens: {result.cost.input_tokens} in / {result.cost.output_tokens} out")
    print(f"  Cost: ${result.cost.estimated_cost_usd}")
    
    if result.errors:
        print(f"\n--- Errors ({len(result.errors)}) ---")
        for err in result.errors:
            print(f"  ✗ {err}")
    
    if result.warnings:
        print(f"\n--- Warnings ({len(result.warnings)}) ---")
        for warn in result.warnings:
            print(f"  ⚠ {warn}")
    
    if result.data:
        seqs = result.data.get("sequences", [])
        print(f"\n--- Data ({len(seqs)} sequences) ---")
        for seq in seqs:
            num = seq.get("seq_num", "?")
            ops = seq.get("ops_count", "?")
            dps = seq.get("duty_periods", [])
            legs = sum(len(dp.get("legs", [])) for dp in dps)
            print(f"  SEQ {num}: {ops} OPS, {legs} legs")
    
    # Save output if requested
    if "--save" in sys.argv and result.data:
        output_path = DEFAULT_BID_PACKET.parent.parent / "parsed" / "engine_output.json"
        with open(output_path, "w") as f:
            json.dump(result.data, f, indent=2)
        print(f"\n--- Saved to {output_path} ---")