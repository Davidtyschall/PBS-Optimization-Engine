"""
Packets API Routes
==================
Endpoints for managing bid packets.
"""

from fastapi import APIRouter, HTTPException
from src.api.models import ListPacketsResponse, BidPacketResponse
from src.db.supabase_client import get_client

router = APIRouter(prefix="/packets", tags=["Packets"])


@router.get("", response_model=ListPacketsResponse)
async def list_packets():
    """
    List all bid packets in the database.
    """
    try:
        supabase = get_client()
        
        result = supabase.table("bid_packets").select("*").execute()
        
        packets = []
        for row in result.data:
            packets.append(BidPacketResponse(
                id=row["id"],
                airline=row["airline"],
                base=row["base"],
                equipment=row["equipment"],
                division=row["division"],
                bid_month=str(row["bid_month"]),
                filename=row["filename"],
                page_count=row["page_count"],
                pages_parsed=row["pages_parsed"] or 0,
                parse_status=row["parse_status"] or "unknown",
                parse_confidence=row.get("parse_confidence")
            ))
        
        return ListPacketsResponse(
            success=True,
            packets=packets,
            total=len(packets)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{packet_id}", response_model=BidPacketResponse)
async def get_packet(packet_id: int):
    """
    Get a specific bid packet by ID.
    """
    try:
        supabase = get_client()
        
        result = supabase.table("bid_packets").select("*").eq("id", packet_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Packet {packet_id} not found")
        
        row = result.data[0]
        
        return BidPacketResponse(
            id=row["id"],
            airline=row["airline"],
            base=row["base"],
            equipment=row["equipment"],
            division=row["division"],
            bid_month=str(row["bid_month"]),
            filename=row["filename"],
            page_count=row["page_count"],
            pages_parsed=row["pages_parsed"] or 0,
            parse_status=row["parse_status"] or "unknown",
            parse_confidence=row.get("parse_confidence")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))