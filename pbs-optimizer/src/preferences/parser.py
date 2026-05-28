"""
NL Preference Parser
====================
Converts natural language pilot preferences to structured data using Claude API.

Architecture:
1. Preprocessing - Clean input
2. LLM Extraction - Call Claude with prompt
3. Validation - Check against schema
4. Normalization - Expand synonyms, convert formats
5. Conflict Detection - Flag impossible combinations
6. Output - Return structured preferences with confidence
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from anthropic import Anthropic

from .prompt_template import get_system_message, get_user_message
from .schema import (
    PilotPreferences,
    validate_airport_code,
    validate_date,
    parse_time_to_minutes,
    VALID_AIRPORT_CODES,
    DayOfWeek,
    PairingType,
    Region,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_BID_MONTH = "2026-01"


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ParseResult:
    """Result of parsing natural language preferences."""
    success: bool
    preferences: Optional[PilotPreferences] = None
    raw_extraction: Optional[Dict[str, Any]] = None
    confidence: str = "low"  # "high", "medium", "low"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    clarification_needed: Optional[str] = None
    original_input: str = ""
    tokens_used: int = 0
    cached: bool = False


# =============================================================================
# PREPROCESSING (Layer 1)
# =============================================================================

def preprocess_input(text: str) -> str:
    """
    Clean and normalize user input.
    
    Args:
        text: Raw user input
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Normalize whitespace (multiple spaces to single)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text


# =============================================================================
# LLM EXTRACTION (Layer 2)
# =============================================================================

def call_llm(
    user_input: str,
    bid_month: str = DEFAULT_BID_MONTH,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> tuple[Dict[str, Any], int, bool]:
    """
    Call Claude API to extract preferences.
    
    Args:
        user_input: Preprocessed user input
        bid_month: Current bid month (YYYY-MM)
        model: Model to use
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        (extracted_dict, tokens_used, was_cached)
    """
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    
    system_message = get_system_message(bid_month)
    user_message = get_user_message(user_input)
    
    # Call with cache control
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    # Extract response text
    response_text = response.content[0].text
    
    # Check if cache was used
    cached = getattr(response.usage, 'cache_read_input_tokens', 0) > 0
    
    # Calculate tokens used
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    
    # Parse JSON response
    try:
        # Clean potential markdown code blocks
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        extracted = json.loads(cleaned)
        return extracted, tokens_used, cached
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}", "raw": response_text}, tokens_used, cached


# =============================================================================
# VALIDATION (Layer 3)
# =============================================================================

def validate_extraction(extracted: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate the LLM's extracted data against schema rules.
    
    Args:
        extracted: Raw extraction from LLM
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for LLM error response
    if "error" in extracted:
        errors.append(f"LLM Error: {extracted['error']}")
        return False, errors
    
    # Check for ambiguous response
    if extracted.get("ambiguous"):
        # Not an error - just needs clarification
        return True, []
    
    # Validate days_off
    if "days_off" in extracted:
        days_off = extracted["days_off"]
        valid_days = {d.value for d in DayOfWeek}
        
        for day in days_off.get("days_off", []):
            if day not in valid_days:
                errors.append(f"Invalid day: {day}")
        
        for date in days_off.get("must_off_dates", []) + days_off.get("prefer_off_dates", []):
            is_valid, error = validate_date(date)
            if not is_valid:
                errors.append(error)
    
    # Validate pairing
    if "pairing" in extracted:
        pairing = extracted["pairing"]
        valid_types = {t.value for t in PairingType}
        
        for pt in pairing.get("prefer_pairing_type", []) + pairing.get("avoid_pairing_type", []):
            if pt not in valid_types:
                errors.append(f"Invalid pairing type: {pt}")
        
        # Validate ranges
        if pairing.get("max_pairing_length") and not (1 <= pairing["max_pairing_length"] <= 6):
            errors.append(f"Invalid max_pairing_length: {pairing['max_pairing_length']}")
    
    # Validate times
    if "times" in extracted:
        times = extracted["times"]
        
        for time_field in ["report_after_minutes", "report_before_minutes",
                          "release_after_minutes", "release_before_minutes"]:
            if time_field in times:
                val = times[time_field]
                if not (0 <= val <= 1439):
                    errors.append(f"Invalid {time_field}: {val}")
        
        if times.get("commuter_base"):
            is_valid, error = validate_airport_code(times["commuter_base"])
            if not is_valid:
                errors.append(error)
    
    # Validate layovers
    if "layovers" in extracted:
        layovers = extracted["layovers"]
        valid_regions = {r.value for r in Region}
        
        for city in layovers.get("prefer_layover_city", []) + layovers.get("avoid_layover_city", []):
            is_valid, error = validate_airport_code(city)
            if not is_valid:
                errors.append(error)
        
        for region in layovers.get("prefer_region", []) + layovers.get("avoid_region", []):
            if region not in valid_regions:
                errors.append(f"Invalid region: {region}")
    
    return len(errors) == 0, errors


# =============================================================================
# NORMALIZATION (Layer 4)
# =============================================================================

def normalize_extraction(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize extracted data (uppercase codes, expand synonyms, etc.)
    
    Args:
        extracted: Validated extraction
        
    Returns:
        Normalized extraction
    """
    normalized = json.loads(json.dumps(extracted))  # Deep copy
    
    # Uppercase airport codes
    if "layovers" in normalized:
        for key in ["prefer_layover_city", "avoid_layover_city"]:
            if key in normalized["layovers"]:
                normalized["layovers"][key] = [
                    code.upper() for code in normalized["layovers"][key]
                ]
    
    if "times" in normalized and normalized["times"].get("commuter_base"):
        normalized["times"]["commuter_base"] = normalized["times"]["commuter_base"].upper()
    
    return normalized


# =============================================================================
# CONFLICT DETECTION (Layer 5)
# =============================================================================

def detect_conflicts(extracted: Dict[str, Any]) -> List[str]:
    """
    Detect conflicting preferences.
    
    Args:
        extracted: Normalized extraction
        
    Returns:
        List of conflict warnings
    """
    conflicts = []
    
    # Credit conflicts
    credit = extracted.get("credit", {})
    if credit.get("maximize_credit") and credit.get("minimize_credit"):
        conflicts.append("CONFLICT: Cannot both maximize and minimize credit")
    
    if credit.get("maximize_credit") and credit.get("minimize_tafb"):
        conflicts.append("WARNING: Maximize credit and minimize TAFB often conflict")
    
    # Layover conflicts
    layovers = extracted.get("layovers", {})
    prefer = set(layovers.get("prefer_layover_city", []))
    avoid = set(layovers.get("avoid_layover_city", []))
    overlap = prefer & avoid
    if overlap:
        conflicts.append(f"CONFLICT: Cities in both prefer and avoid: {overlap}")
    
    # Pairing conflicts
    pairing = extracted.get("pairing", {})
    prefer_types = set(pairing.get("prefer_pairing_type", []))
    avoid_types = set(pairing.get("avoid_pairing_type", []))
    overlap = prefer_types & avoid_types
    if overlap:
        conflicts.append(f"CONFLICT: Pairing types in both prefer and avoid: {overlap}")
    
    # Turn + layover conflict
    if "Turn" in prefer_types and layovers.get("prefer_layover_city"):
        conflicts.append("CONFLICT: Turns (day trips) cannot have layovers")
    
    if pairing.get("max_pairing_length") == 1 and layovers.get("prefer_layover_city"):
        conflicts.append("CONFLICT: 1-day trips cannot have layovers")
    
    return conflicts


# =============================================================================
# CONFIDENCE SCORING (Layer 6)
# =============================================================================

def calculate_confidence(
    extracted: Dict[str, Any],
    validation_errors: List[str],
    conflicts: List[str]
) -> str:
    """
    Calculate confidence level of the extraction.
    
    Args:
        extracted: The extraction
        validation_errors: Any validation errors
        conflicts: Any conflicts detected
        
    Returns:
        "high", "medium", or "low"
    """
    if extracted.get("ambiguous"):
        return "low"
    
    if validation_errors:
        return "low"
    
    critical_conflicts = [c for c in conflicts if c.startswith("CONFLICT")]
    if critical_conflicts:
        return "medium"
    
    if conflicts:  # Just warnings
        return "medium"
    
    # Check if extraction is meaningful
    non_empty_categories = 0
    for category in ["days_off", "pairing", "times", "layovers", "credit", "work_blocks"]:
        if category in extracted and extracted[category]:
            non_empty_categories += 1
    
    if non_empty_categories == 0:
        return "low"
    
    return "high"


# =============================================================================
# MAIN PARSER FUNCTION
# =============================================================================

def parse_preferences(
    user_input: str,
    bid_month: str = DEFAULT_BID_MONTH,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> ParseResult:
    """
    Parse natural language preferences into structured format.
    
    This is the main entry point for the NL Preference Parser.
    
    Args:
        user_input: Natural language preference text from pilot
        bid_month: Current bid month (YYYY-MM format)
        model: Claude model to use
        api_key: Optional API key
        
    Returns:
        ParseResult with structured preferences and metadata
    """
    result = ParseResult(success=False, original_input=user_input)
    
    # Layer 1: Preprocessing
    cleaned_input = preprocess_input(user_input)
    
    if not cleaned_input:
        result.errors.append("Empty input")
        result.clarification_needed = "Please describe your scheduling preferences"
        return result
    
    # Layer 2: LLM Extraction
    try:
        extracted, tokens_used, cached = call_llm(
            cleaned_input,
            bid_month=bid_month,
            model=model,
            api_key=api_key
        )
        result.tokens_used = tokens_used
        result.cached = cached
        result.raw_extraction = extracted
    except Exception as e:
        result.errors.append(f"LLM call failed: {str(e)}")
        return result
    
    # Check for ambiguous response
    if extracted.get("ambiguous"):
        result.success = True
        result.clarification_needed = extracted.get("clarification_needed", "Please clarify your preferences")
        result.confidence = "low"
        return result
    
    # Layer 3: Validation
    is_valid, validation_errors = validate_extraction(extracted)
    result.errors.extend(validation_errors)
    
    if not is_valid:
        result.confidence = "low"
        return result
    
    # Layer 4: Normalization
    normalized = normalize_extraction(extracted)
    
    # Layer 5: Conflict Detection
    conflicts = detect_conflicts(normalized)
    result.warnings.extend(conflicts)
    
    # Layer 6: Confidence Scoring
    result.confidence = calculate_confidence(normalized, validation_errors, conflicts)
    
    # Convert to PilotPreferences object
    try:
        result.preferences = PilotPreferences.from_dict(normalized)
        result.preferences.original_input = user_input
        result.preferences.parse_confidence = result.confidence
        result.preferences.parse_warnings = result.warnings
        
        # Run schema validation
        prefs_valid, prefs_errors = result.preferences.validate()
        if not prefs_valid:
            result.warnings.extend(prefs_errors)
            if result.confidence == "high":
                result.confidence = "medium"
        
        result.success = True
        
    except Exception as e:
        result.errors.append(f"Failed to build preferences: {str(e)}")
        result.success = False
    
    return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("NL PREFERENCE PARSER TEST")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠ ANTHROPIC_API_KEY not set. Skipping live test.")
        print("Set it with: export ANTHROPIC_API_KEY=your_key")
        print("\nTesting preprocessing only...")
        
        test_inputs = [
            "  I want Fridays off!!!  ",
            "Hawaii trips – no redeyes",
            '"turns only" max pay',
        ]
        
        for inp in test_inputs:
            cleaned = preprocess_input(inp)
            print(f"\nInput: {repr(inp)}")
            print(f"Cleaned: {repr(cleaned)}")
    
    else:
        print("\n✓ API key found. Running live test...")
        
        test_cases = [
            "I want Fridays off and Hawaii trips",
            "No redeyes, commuter from Denver, home by 6pm",
            "maximize pay, short trips only",
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {test_input}")
            print("=" * 60)
            
            result = parse_preferences(test_input)
            
            print(f"\nSuccess: {result.success}")
            print(f"Confidence: {result.confidence}")
            print(f"Tokens Used: {result.tokens_used}")
            print(f"Cached: {result.cached}")
            
            if result.warnings:
                print(f"Warnings: {result.warnings}")
            
            if result.errors:
                print(f"Errors: {result.errors}")
            
            if result.preferences:
                print(f"\nExtracted Preferences:")
                print(result.preferences.summary())