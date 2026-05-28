'''
Docstring for pbs-optimizer.src.compare

Ground-truth comparison module. 

In ML/AI systems, you need to track whether your system is truly accurate, or just confident. 

Ground truth = the known correct answer 
Prediction = what the system produces (engine's output)

Through comparing these two points, we obtain
accuracy metrics in order to objectively measure if system works well. 
'''

# src/compare.py
# Ground truth comparison tool

import json
from pathlib import Path
from config import DEFAULT_BID_PACKET


def load_json(file_path: str) -> dict:
    """Load a JSON file and return as dict."""
    # Your code here
    with open(file_path, "r") as f:
        data = json.load(f) # parses JSON into Python dict
    return data


def compare_values(expected, actual, path: str) -> list[dict]:
    """
    Compare two values recursively.
    
    Args:
        expected: The ground truth value
        actual: The LLM output value
        path: Current location in the JSON (e.g., "sequences.0.seq_num")
    
    Returns:
        List of mismatch dicts: {"path": str, "expected": any, "actual": any}
    """
    # Code here
    
def compare_values(expected, actual, path: str) -> list[dict]:
    mismatches = []
    
    # Case 1: Both are dicts
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected.keys():
            new_path = f"{path}.{key}" if path else key
            if key not in actual:
                mismatches.append({
                    "path": new_path,
                    "expected": expected[key],
                    "actual": "MISSING"
                })
            else:
                mismatches.extend(compare_values(expected[key], actual[key], new_path))
    
    # Case 2: Both are lists
    elif isinstance(expected, list) and isinstance(actual, list):
        max_len = max(len(expected), len(actual))
        for index in range(max_len):
            new_path = f"{path}[{index}]"
            if index >= len(expected):
                mismatches.append({
                    "path": new_path,
                    "expected": "MISSING",
                    "actual": actual[index]
                })
            elif index >= len(actual):
                mismatches.append({
                    "path": new_path,
                    "expected": expected[index],
                    "actual": "MISSING"
                })
            else:
                mismatches.extend(compare_values(expected[index], actual[index], new_path))
    
    # Case 3: Simple values (or type mismatch)
    else:
        if expected != actual:
            mismatches.append({
                "path": path,
                "expected": expected,
                "actual": actual
            })
        # If expected != actual, append a mismatch
    return mismatches


def compare_bid_packets(expected: dict, actual: dict) -> dict:
    """
    Compare full bid packet output against ground truth.
    
    Returns:
        Dict with keys: mismatches, mismatch_count, errors_by_field
    """
    # Get all mismatches
    mismatches = compare_values(expected, actual, "")
    
    # Group errors by field type (e.g., "calendar_start_dates", "meal")
    errors_by_field = {}
    for m in mismatches:
        # Extract field name from path (last part)
        # "sequences[0].calendar_start_dates[1]" → "calendar_start_dates"
        path = m["path"]
        # Find the field name (part before any [index])
        parts = path.replace("]", "").replace("[", ".").split(".")
        field_name = parts[-1] if not parts[-1].isdigit() else parts[-2]
        
        if field_name not in errors_by_field:
            errors_by_field[field_name] = []
        errors_by_field[field_name].append(m)
    
    return {
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "errors_by_field": errors_by_field
    }


def print_report(comparison_result: dict):
    """Print formatted comparison report to terminal."""
    
    # Header
    print("=" * 60)
    print("GROUND TRUTH COMPARISON")
    print("=" * 60)
    
    # Check if no mismatches
    if comparison_result["mismatch_count"] == 0:
        print("\n✓ Perfect match! No differences found.")
        return
    
    # Print each mismatch
    print(f"\nMismatches ({comparison_result['mismatch_count']}):\n")
    for mismatch in comparison_result["mismatches"]:
        print(f"  ✗ {mismatch['path']}")
        print(f"      Expected: {mismatch['expected']}")
        print(f"      Actual:   {mismatch['actual']}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total mismatches: {comparison_result['mismatch_count']}")
    
    # Errors by field
    print("\nErrors by field:")
    for field_name, errors in comparison_result["errors_by_field"].items():
        print(f"  {field_name}: {len(errors)} error(s)")


if __name__ == "__main__":
    # File paths
    ground_truth_path = DEFAULT_BID_PACKET.parent.parent / "parsed" / "example_page_902.json"
    engine_output_path = DEFAULT_BID_PACKET.parent.parent / "parsed" / "engine_output.json"
    
    # Load both files
    expected = load_json(ground_truth_path)
    actual = load_json(engine_output_path)
    
    # Run comparison
    result = compare_bid_packets(expected, actual)
    
    # Print report
    print_report(result)