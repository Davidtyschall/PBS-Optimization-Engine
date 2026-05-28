# src/calendar_resolver.py
# Extracts calendar start dates using regex instead of LLM
'''
Docstring for pbs-optimizer.src.calendar_resolver
Deterministic algorithm to extract calendar start dates from OCR text using regex.
Uses spatial awareness to find dates near "CALENDAR START" labels (line [-45])
Solution to: 
    Since Tesseract and other OCR tools struggle to keep the numbers (1–31) aligned with their headers 
    (MO, TU, etc.), we propose a Spatial-Regex Algorithm.
Leverages tokenization to map character positions to line/column coordinates.
Cross-checks comparing results against ops_count from sequence headers.
'''
import re


def extract_calendar_dates_from_text(sequence_text: str, ops_count: int) -> dict:
    """
    Extract calendar start dates from OCR text using regex.
    
    The calendar grid appears on the right side of the text,
    with day numbers (1-31) and dashes (−−) for empty days.
    
    Args:
        sequence_text: OCR text for one sequence (from SEQ to TTL)
        ops_count: Expected number of dates from "X OPS"
        
    Returns:
        {
            "dates": [7, 11, ...],
            "confidence": "high" | "medium" | "low",
            "ops_match": True | False,
            "message": str
        }
    """
    
    lines = sequence_text.split('\n')
    found_dates = set()
    
    # Find the calendar column position from the header line
    calendar_start_pos = None
    for line in lines:
        if 'MO TU WE TH FR SA SU' in line:
            calendar_start_pos = line.find('MO TU WE TH FR SA SU')
            break
    
    # Fallback if header not found in sequence text
    if calendar_start_pos is None:
        calendar_start_pos = 80  # Default position
    
    for line in lines:
        # Skip short lines
        if len(line) <= calendar_start_pos:
            continue
            
        # Extract only the calendar region (from header position to end)
        calendar_region = line[calendar_start_pos:]
        
        # Skip header line itself
        if 'MO TU WE TH FR SA SU' in calendar_region:
            continue
        
        # Split by whitespace and dashes (both regular - and −)
        tokens = re.split(r'[\s\-−]+', calendar_region)
        
        for token in tokens:
            # Must be exactly 1-2 digits representing days 1-31
            if re.match(r'^([1-9]|[12][0-9]|3[01])$', token):
                found_dates.add(int(token))
    
    # Sort the dates
    dates = sorted(list(found_dates))
    
    # Determine confidence
    ops_match = len(dates) == ops_count
    
    if ops_match:
        confidence = "high"
        message = f"Found {len(dates)} dates matching ops_count"
    elif len(dates) > 0:
        confidence = "medium"
        message = f"Found {len(dates)} dates, expected {ops_count}"
    else:
        confidence = "low"
        message = "No dates found in calendar region"
    
    return {
        "dates": dates,
        "confidence": confidence,
        "ops_match": ops_match,
        "message": message
    }


def extract_sequence_text(full_page_text: str, seq_num: int) -> str:
    """
    Extract the text block for a single sequence.
    
    Args:
        full_page_text: Full OCR text for the page
        seq_num: Sequence number to find (e.g., 217)
        
    Returns:
        Text from "SEQ XXX" to the next "SEQ" or end of relevant section
    """
    lines = full_page_text.split('\n')
    
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        # Find start of this sequence
        if f'SEQ {seq_num}' in line and start_idx is None:
            start_idx = i
        # Find end (next sequence or TTL line)
        elif start_idx is not None:
            if line.strip().startswith('SEQ ') or line.strip().startswith('TTL'):
                if line.strip().startswith('TTL'):
                    end_idx = i + 1  # Include TTL line
                else:
                    end_idx = i
                break
    
    if start_idx is None:
        return ""
    
    if end_idx is None:
        end_idx = len(lines)
    
    return '\n'.join(lines[start_idx:end_idx])


# Test function
if __name__ == "__main__":
    from ocr import extract_text_with_ocr
    
    # Load page 7
    text = extract_text_with_ocr('../data/raw/DFW_JAN.pdf', page_num=7)
    
    # Test sequences
    test_cases = [
        (217, 2),   # Expected: [7, 11]
        (218, 2),   # Expected: [8, 9]
        (224, 3),   # Expected: [1, 3, 4]
        (225, 4),   # Expected: [2, 3, 4, 5]
        (227, 12),  # Expected: [12-23]
        (228, 3),   # Expected: [1, 2, 3]
    ]
    
    print("Calendar Date Extraction Test")
    print("=" * 50)
    
    for seq_num, ops_count in test_cases:
        seq_text = extract_sequence_text(text, seq_num)
        result = extract_calendar_dates_from_text(seq_text, ops_count)
        
        print(f"\nSEQ {seq_num} ({ops_count} OPS):")
        print(f"  Dates: {result['dates']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Match: {result['ops_match']}")