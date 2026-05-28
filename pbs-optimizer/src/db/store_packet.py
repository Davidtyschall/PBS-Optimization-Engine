"""
Store Packet Module
Stores parsed bid packet data into Supabase database.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .supabase_client import get_client
from .transformers import (
    time_string_to_minutes,
    clock_time_to_minutes,
    extract_layover_cities,
    map_division,
    calculate_file_hash,
    derive_pairing_length,
    is_transoceanic_trip,
    is_redeye_trip,
)


@dataclass
class StorageResult:
    """Result of storing a parsed packet."""
    success: bool
    packet_id: Optional[int] = None
    sequences_inserted: int = 0
    skipped_duplicate: bool = False
    error: Optional[str] = None


def get_or_create_packet(
    pdf_path: str,
    metadata: dict,
    page_count: int,
    confidence: str = "high",
    needs_review: bool = False,
    errors: list = None,
    warnings: list = None,
) -> tuple[int, bool]:
    """
    Get existing packet by file hash, or create new one.
    
    Returns:
        (packet_id, is_new) - ID and whether it was newly created
    """
    supabase = get_client()
    file_hash = calculate_file_hash(pdf_path)
    
    # Check for existing packet with same hash
    existing = supabase.table("bid_packets").select("id").eq("file_hash", file_hash).execute()
    
    if existing.data:
        return existing.data[0]["id"], False
    
    # Extract metadata fields
    bid_status = metadata.get("bid_status", {})
    
    # Create new packet
    packet_data = {
        "airline": "AA",
        "base": bid_status.get("base", ""),
        "equipment": bid_status.get("equipment_family", ""),
        "division": map_division(bid_status.get("division", "DOM")),
        "bid_month": metadata.get("effective_date", ""),
        "file_hash": file_hash,
        "filename": Path(pdf_path).name,
        "page_count": page_count,
        "parse_confidence": confidence,
        "needs_review": needs_review,
        "parse_errors": errors or [],
        "parse_warnings": warnings or [],
        "pages_parsed": 0,
        "parse_status": "in_progress",
    }
    
    result = supabase.table("bid_packets").insert(packet_data).execute()
    
    if result.data:
        return result.data[0]["id"], True
    else:
        raise Exception("Failed to create bid_packet")


def transform_sequence(sequence: dict, packet_id: int, source_page: int) -> dict:
    """
    Transform a parsed sequence into database format.
    """
    # Extract layover cities
    duty_periods = sequence.get("duty_periods", [])
    layover_cities = extract_layover_cities(duty_periods)
    
    # Get TTL section
    ttl = sequence.get("ttl", {})
    
    # Get report time from sequence
    report_time = sequence.get("sequence_report_time", {}).get("base", "0000")
    
    # Get release time from last duty period
    release_time = "0000"
    if duty_periods:
        last_dp = duty_periods[-1]
        release_time = last_dp.get("release_time", {}).get("base", "0000")
    
    # Build database record
    return {
        "packet_id": packet_id,
        "seq_num": sequence.get("seq_num"),
        "ops_count": sequence.get("ops_count", 1),
        "positions": ",".join(sequence.get("positions", [])),
        "calendar_start_dates": sequence.get("calendar_start_dates", []),
        "total_block_minutes": time_string_to_minutes(ttl.get("total_block_hhmm", "0.00")),
        "total_tafb_minutes": time_string_to_minutes(ttl.get("tafb_hhmm", "0.00")),
        "total_credit_minutes": time_string_to_minutes(ttl.get("total_tpay_hhmm", "0.00")),
        "layover_cities": layover_cities if layover_cities else None,
        "pairing_length": derive_pairing_length(duty_periods),
        "duty_periods": len(duty_periods),
        "report_time_minutes": clock_time_to_minutes(report_time),
        "release_time_minutes": clock_time_to_minutes(release_time),
        "is_redeye": is_redeye_trip(sequence),
        "is_transoceanic": is_transoceanic_trip(layover_cities),
        "pairing_types": None,  # Can derive later
        "regions": None,  # Can derive later
        "data_json": sequence,  # Store full raw data
        "source_page": source_page,
    }


def store_sequences(packet_id: int, sequences: list, source_page: int) -> int:
    """
    Store multiple sequences for a packet.
    
    Returns:
        Number of sequences inserted
    """
    supabase = get_client()
    
    if not sequences:
        return 0
    
    # Transform all sequences
    transformed = [
        transform_sequence(seq, packet_id, source_page)
        for seq in sequences
    ]
    
    # Batch insert
    result = supabase.table("sequences").insert(transformed).execute()
    
    return len(result.data) if result.data else 0


def update_packet_progress(packet_id: int, pages_parsed: int, status: str = "in_progress"):
    """
    Update packet parsing progress.
    """
    supabase = get_client()
    
    supabase.table("bid_packets").update({
        "pages_parsed": pages_parsed,
        "parse_status": status,
    }).eq("id", packet_id).execute()


def store_parsed_page(
    pdf_path: str,
    page_num: int,
    parsed_data: dict,
    page_count: int = 1,
    confidence: str = "high",
    needs_review: bool = False,
    errors: list = None,
    warnings: list = None,
) -> StorageResult:
    """
    Store a single parsed page into the database.
    
    Args:
        pdf_path: Path to the source PDF
        page_num: Page number that was parsed
        parsed_data: The parser output (page_metadata + sequences)
        page_count: Total pages in PDF
        confidence: Parse confidence level
        needs_review: Whether manual review is needed
        errors: List of parse errors
        warnings: List of parse warnings
    
    Returns:
        StorageResult with success status and details
    """
    try:
        # Validate input
        if "page_metadata" not in parsed_data:
            return StorageResult(
                success=False,
                error="Missing page_metadata in parsed data"
            )
        
        if "sequences" not in parsed_data or not parsed_data["sequences"]:
            return StorageResult(
                success=False,
                error="No sequences found in parsed data"
            )
        
        metadata = parsed_data["page_metadata"]
        sequences = parsed_data["sequences"]
        
        # Get or create packet
        packet_id, is_new = get_or_create_packet(
            pdf_path=pdf_path,
            metadata=metadata,
            page_count=page_count,
            confidence=confidence,
            needs_review=needs_review,
            errors=errors,
            warnings=warnings,
        )
        
        if not is_new:
            # Check if this page was already parsed
            supabase = get_client()
            existing = supabase.table("sequences")\
                .select("id")\
                .eq("packet_id", packet_id)\
                .eq("source_page", page_num)\
                .limit(1)\
                .execute()
            
            if existing.data:
                return StorageResult(
                    success=True,
                    packet_id=packet_id,
                    sequences_inserted=0,
                    skipped_duplicate=True,
                )
        
        # Store sequences
        inserted_count = store_sequences(packet_id, sequences, page_num)
        
        # Update progress
        supabase = get_client()
        current = supabase.table("bid_packets")\
            .select("pages_parsed")\
            .eq("id", packet_id)\
            .execute()
        
        current_pages = current.data[0]["pages_parsed"] if current.data else 0
        update_packet_progress(packet_id, current_pages + 1)
        
        return StorageResult(
            success=True,
            packet_id=packet_id,
            sequences_inserted=inserted_count,
            skipped_duplicate=False,
        )
        
    except Exception as e:
        return StorageResult(
            success=False,
            error=str(e)
        )


# CLI for testing
if __name__ == "__main__":
    import sys
    
    # Load test data
    # Revisit this later 
    test_json = Path(__file__).parent.parent.parent / "data" / "parsed" / "engine_output.json"
    test_pdf = Path(__file__).parent.parent.parent / "data" / "raw" / "DFW_JAN.pdf"
    
    if not test_json.exists():
        print(f"Test JSON not found: {test_json}")
        sys.exit(1)
    
    if not test_pdf.exists():
        print(f"Test PDF not found: {test_pdf}")
        sys.exit(1)
    
    with open(test_json) as f:
        parsed_data = json.load(f)
    
    print("=" * 60)
    print("STORE PACKET TEST")
    print("=" * 60)
    print(f"PDF: {test_pdf}")
    print(f"JSON: {test_json}")
    print(f"Sequences: {len(parsed_data.get('sequences', []))}")
    print()
    
    # Store the page
    result = store_parsed_page(
        pdf_path=str(test_pdf),
        page_num=7,  # We parsed page 7
        parsed_data=parsed_data,
        page_count=972,
        confidence="high",
    )
    
    print("RESULT:")
    print(f"  Success: {result.success}")
    print(f"  Packet ID: {result.packet_id}")
    print(f"  Sequences Inserted: {result.sequences_inserted}")
    print(f"  Skipped (duplicate): {result.skipped_duplicate}")
    if result.error:
        print(f"  Error: {result.error}")