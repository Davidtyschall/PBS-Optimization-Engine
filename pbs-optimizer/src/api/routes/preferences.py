"""
Preferences API Routes
======================
Endpoints for parsing natural language preferences.
"""

from fastapi import APIRouter, HTTPException
from src.api.models import ParsePreferencesRequest, ParsePreferencesResponse
from src.preferences.parser import parse_preferences

router = APIRouter(prefix="/preferences", tags=["Preferences"])


@router.post("/parse", response_model=ParsePreferencesResponse)
async def parse_preferences_endpoint(request: ParsePreferencesRequest):
    """
    Parse natural language preferences into structured format.
    
    Takes free-form text like "I want Fridays off and Hawaii trips"
    and returns structured preferences that can be used for scoring.
    """
    try:
        result = parse_preferences(
            user_input=request.text,
            bid_month=request.bid_month
        )
        
        return ParsePreferencesResponse(
            success=result.success,
            confidence=result.confidence,
            preferences=result.preferences.to_dict() if result.preferences else None,
            warnings=result.warnings,
            errors=result.errors,
            clarification_needed=result.clarification_needed,
            tokens_used=result.tokens_used,
            cached=result.cached
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))