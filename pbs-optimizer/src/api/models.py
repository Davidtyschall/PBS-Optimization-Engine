"""
API Request/Response Models
===========================
Pydantic models for API validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# =============================================================================
# PREFERENCES
# =============================================================================

class ParsePreferencesRequest(BaseModel):
    """Request to parse natural language preferences."""
    text: str = Field(..., description="Natural language preference text", min_length=1)
    bid_month: str = Field(default="2026-01", description="Bid month in YYYY-MM format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I want Fridays off and Hawaii trips, no redeyes",
                "bid_month": "2026-01"
            }
        }


class ParsePreferencesResponse(BaseModel):
    """Response from parsing preferences."""
    success: bool
    confidence: str = Field(description="high, medium, or low")
    preferences: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    errors: List[str] = []
    clarification_needed: Optional[str] = None
    tokens_used: int = 0
    cached: bool = False


# =============================================================================
# SCORING
# =============================================================================

class ScoreSequencesRequest(BaseModel):
    """Request to score sequences against preferences."""
    preferences: Dict[str, Any] = Field(..., description="Structured preferences object")
    packet_id: int = Field(..., description="Bid packet ID to score sequences from")
    bid_year: int = Field(default=2026)
    bid_month: int = Field(default=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "preferences": {
                    "days_off": {"days_off": ["Friday"]},
                    "layovers": {"prefer_layover_city": ["HNL", "OGG"]}
                },
                "packet_id": 1
            }
        }


class CategoryScoreResponse(BaseModel):
    """Score breakdown for a single category."""
    score: float
    max_score: float
    weight: float
    weighted_score: float
    factors: List[str]


class SequenceScoreResponse(BaseModel):
    """Scoring result for a single sequence."""
    sequence_id: int
    seq_num: int
    rank: int
    final_score: float
    disqualified: bool
    disqualification_reason: Optional[str] = None
    breakdown: Dict[str, CategoryScoreResponse]
    explanation: str
    
    # Sequence details
    calendar_start_dates: List[int]
    layover_cities: List[str]
    pairing_length: int
    total_credit_minutes: int
    total_tafb_minutes: int


class ScoreSequencesResponse(BaseModel):
    """Response from scoring sequences."""
    success: bool
    ranked_sequences: List[SequenceScoreResponse] = []
    total_sequences: int = 0
    disqualified_count: int = 0
    preferences_summary: str = ""
    error: Optional[str] = None


# =============================================================================
# PACKETS
# =============================================================================

class BidPacketResponse(BaseModel):
    """Bid packet information."""
    id: int
    airline: str
    base: str
    equipment: str
    division: str
    bid_month: str
    filename: str
    page_count: int
    pages_parsed: int
    parse_status: str
    parse_confidence: Optional[str] = None


class ListPacketsResponse(BaseModel):
    """Response listing all bid packets."""
    success: bool
    packets: List[BidPacketResponse] = []
    total: int = 0
    error: Optional[str] = None


# =============================================================================
# SEQUENCES
# =============================================================================

class SequenceResponse(BaseModel):
    """Sequence information."""
    id: int
    packet_id: int
    seq_num: int
    ops_count: int
    positions: str
    calendar_start_dates: List[int]
    total_block_minutes: int
    total_tafb_minutes: int
    total_credit_minutes: int
    layover_cities: Optional[List[str]] = None
    pairing_length: int
    duty_periods: int
    report_time_minutes: int
    release_time_minutes: int
    is_redeye: bool
    is_transoceanic: bool
    source_page: Optional[int] = None


class ListSequencesResponse(BaseModel):
    """Response listing sequences."""
    success: bool
    sequences: List[SequenceResponse] = []
    total: int = 0
    packet_id: Optional[int] = None
    error: Optional[str] = None