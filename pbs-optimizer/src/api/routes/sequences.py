"""
Sequences API Routes
====================
Endpoints for managing and scoring sequences.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.api.models import (
    ListSequencesResponse,
    SequenceResponse,
    ScoreSequencesRequest,
    ScoreSequencesResponse,
    SequenceScoreResponse,
    CategoryScoreResponse,
)
from src.db.supabase_client import get_client
from src.scoring.engine import score_sequences
from src.preferences.schema import PilotPreferences

router = APIRouter(prefix="/sequences", tags=["Sequences"])


@router.get("", response_model=ListSequencesResponse)
async def list_sequences(
    packet_id: Optional[int] = Query(None, description="Filter by packet ID")
):
    """
    List sequences, optionally filtered by packet ID.
    """
    try:
        supabase = get_client()
        
        query = supabase.table("sequences").select("*")
        
        if packet_id is not None:
            query = query.eq("packet_id", packet_id)
        
        result = query.execute()
        
        sequences = []
        for row in result.data:
            sequences.append(SequenceResponse(
                id=row["id"],
                packet_id=row["packet_id"],
                seq_num=row["seq_num"],
                ops_count=row["ops_count"] or 1,
                positions=row["positions"] or "",
                calendar_start_dates=row["calendar_start_dates"] or [],
                total_block_minutes=row["total_block_minutes"] or 0,
                total_tafb_minutes=row["total_tafb_minutes"] or 0,
                total_credit_minutes=row["total_credit_minutes"] or 0,
                layover_cities=row["layover_cities"],
                pairing_length=row["pairing_length"] or 1,
                duty_periods=row["duty_periods"] or 1,
                report_time_minutes=row["report_time_minutes"] or 480,
                release_time_minutes=row["release_time_minutes"] or 1020,
                is_redeye=row["is_redeye"] or False,
                is_transoceanic=row["is_transoceanic"] or False,
                source_page=row.get("source_page")
            ))
        
        return ListSequencesResponse(
            success=True,
            sequences=sequences,
            total=len(sequences),
            packet_id=packet_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score", response_model=ScoreSequencesResponse)
async def score_sequences_endpoint(request: ScoreSequencesRequest):
    """
    Score sequences against pilot preferences.
    
    Returns ranked list with explanations for each score.
    """
    try:
        supabase = get_client()
        
        # Get sequences for the packet
        result = supabase.table("sequences").select("*").eq("packet_id", request.packet_id).execute()
        
        if not result.data:
            return ScoreSequencesResponse(
                success=False,
                error=f"No sequences found for packet {request.packet_id}"
            )
        
        # Convert to list of dicts
        sequences = []
        for row in result.data:
            sequences.append({
                "id": row["id"],
                "seq_num": row["seq_num"],
                "calendar_start_dates": row["calendar_start_dates"] or [],
                "layover_cities": row["layover_cities"],
                "pairing_length": row["pairing_length"] or 1,
                "is_redeye": row["is_redeye"] or False,
                "is_transoceanic": row["is_transoceanic"] or False,
                "report_time_minutes": row["report_time_minutes"] or 480,
                "release_time_minutes": row["release_time_minutes"] or 1020,
                "total_credit_minutes": row["total_credit_minutes"] or 0,
                "total_tafb_minutes": row["total_tafb_minutes"] or 0,
            })
        
        # Build preferences object
        prefs = PilotPreferences.from_dict(request.preferences)
        
        # Score sequences
        scoring_result = score_sequences(
            prefs=prefs,
            sequences=sequences,
            bid_year=request.bid_year,
            bid_month=request.bid_month
        )
        
        # Convert to response format
        ranked = []
        for seq_score in scoring_result.ranked_sequences:
            breakdown = {}
            for cat_name, cat_score in seq_score.breakdown.items():
                breakdown[cat_name] = CategoryScoreResponse(
                    score=cat_score.score,
                    max_score=cat_score.max_score,
                    weight=cat_score.weight,
                    weighted_score=cat_score.weighted_score,
                    factors=cat_score.factors
                )
            
            ranked.append(SequenceScoreResponse(
                sequence_id=seq_score.sequence_id,
                seq_num=seq_score.seq_num,
                rank=seq_score.rank,
                final_score=seq_score.final_score,
                disqualified=seq_score.disqualified,
                disqualification_reason=seq_score.disqualification_reason,
                breakdown=breakdown,
                explanation=seq_score.explanation,
                calendar_start_dates=seq_score.calendar_start_dates,
                layover_cities=seq_score.layover_cities or [],
                pairing_length=seq_score.pairing_length,
                total_credit_minutes=seq_score.total_credit_minutes,
                total_tafb_minutes=seq_score.total_tafb_minutes
            ))
        
        return ScoreSequencesResponse(
            success=True,
            ranked_sequences=ranked,
            total_sequences=scoring_result.total_sequences,
            disqualified_count=scoring_result.disqualified_count,
            preferences_summary=scoring_result.preferences_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))