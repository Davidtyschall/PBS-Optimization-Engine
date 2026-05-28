"""
Prompt Template for NL Preference Parser
========================================
Builds the complete prompt for Claude API to extract pilot preferences.
"""

from .examples import get_examples_for_prompt

# =============================================================================
# SYSTEM INSTRUCTIONS
# =============================================================================

SYSTEM_INSTRUCTIONS = """You are a preference extraction system for airline pilot scheduling (PBS - Preferential Bidding System).

Your job is to convert natural language preferences into structured JSON that matches the schema below.

RULES:
1. Output ONLY valid JSON - no explanations, no markdown, no code blocks
2. Only use fields and values defined in the schema
3. Expand synonyms (e.g., "Hawaii" → ["HNL", "OGG", "LIH", "KOA"])
4. Expand related types (e.g., "redeyes" → ["Redeye", "Trailing Redeye"])
5. Convert times to minutes from midnight (e.g., "6pm" → 1080)
6. Use current bid month for specific dates (e.g., January 2026)
7. If input is ambiguous or empty, return: {"ambiguous": true, "clarification_needed": "question"}
8. If preferences conflict, still extract them - validation happens later

TIME CONVERSION REFERENCE:
- 6am = 360, 7am = 420, 8am = 480, 9am = 540, 10am = 600
- 12pm = 720, 1pm = 780, 2pm = 840, 3pm = 900, 4pm = 960
- 5pm = 1020, 6pm = 1080, 7pm = 1140, 8pm = 1200, 9pm = 1260"""


# =============================================================================
# COMPRESSED SCHEMA DEFINITION
# =============================================================================

SCHEMA_DEFINITION = """
PREFERENCE SCHEMA:

days_off:
  days_off: List[Day] - Recurring days off. Day = Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday
  days_off_strength: "must_have"|"strong"|"preferred"|"weak" - How important (default: "preferred")
  must_off_dates: List[str] - Specific dates YYYY-MM-DD that MUST be off
  prefer_off_dates: List[str] - Specific dates YYYY-MM-DD preferred off
  maximize_total_days_off: bool - Want maximum days off
  maximize_block_days_off: bool - Want consecutive days off
  maximize_weekends_off: bool - Want full weekends (Sat+Sun) off
  min_consecutive_days_off: int (1-14) - Minimum days off in a row

pairing:
  prefer_pairing_length: List[int] - Preferred trip lengths in days [1,2,3,4,5]
  min_pairing_length: int (1-6) - Minimum trip length
  max_pairing_length: int (1-6) - Maximum trip length
  prefer_pairing_type: List[Type] - Preferred types
  avoid_pairing_type: List[Type] - Types to avoid
  Type = Early|Morning|Midday|Evening|Late|Redeye|Trailing Redeye|ODAN|Split Duty|Transcon|Transoceanic|Turn|Charter
  max_duty_periods: int (1-5) - Max duty periods per trip
  max_legs_per_duty: int (1-8) - Max flight legs per duty period
  max_tafb_credit_ratio: float (1.0-5.0) - Max time-away/credit ratio (lower = more efficient)

times:
  report_after_minutes: int (0-1439) - Don't report before this time
  report_before_minutes: int (0-1439) - Don't report after this time
  release_before_minutes: int (0-1439) - Must be released by this time
  release_after_minutes: int (0-1439) - Don't release before this time
  is_commuter: bool - Pilot commutes to base
  commuter_base: str - 3-letter airport code where pilot lives (e.g., "DEN")

layovers:
  prefer_layover_city: List[str] - Preferred layover airports (3-letter codes)
  avoid_layover_city: List[str] - Airports to avoid
  prefer_region: List[Region] - Preferred regions
  avoid_region: List[Region] - Regions to avoid
  Region = Northeast|Southeast|North Central|South Central|Northwest|Southwest|Hawaii|Alaska|Canada|Mexico|Caribbean|Central America|South America|Europe|Africa|Middle East|Asia|Australia/NZ
  min_layover_hours: float (8-96) - Minimum layover duration
  max_layover_hours: float (8-96) - Maximum layover duration

credit:
  target_credit_min_minutes: int (3000-6600) - Minimum monthly credit target (50-110 hrs)
  target_credit_max_minutes: int (3000-6600) - Maximum monthly credit target
  maximize_credit: bool - Maximize pay/credit
  minimize_credit: bool - Minimize credit (senior pilots wanting time off)
  minimize_tafb: bool - Minimize time away from base
  prefer_efficient_flying: bool - High credit-to-TAFB ratio

work_blocks:
  min_work_block_days: int (1-7) - Minimum consecutive work days
  max_work_block_days: int (1-7) - Maximum consecutive work days
  prefer_work_block_days: List[int] - Preferred work block lengths
  min_days_off_between_blocks: int - Minimum rest between work blocks

reserve:
  reserve_type_preference: "Long Call"|"Short Call"|"Either" - Reserve type if not lineholder
  avoid_reserve: bool - Prefer lineholder over reserve

misc:
  prefer_deadheads: bool - Like deadhead legs
  avoid_deadheads: bool - Avoid deadhead legs
  buddy_bid_employee_numbers: List[str] - Up to 3 employee numbers to fly with

CITY/REGION EXPANSIONS:
- Hawaii → ["HNL", "OGG", "LIH", "KOA"]
- Japan/Tokyo → ["NRT", "HND"]
- Korea/Seoul → ["ICN"]
- London → ["LHR", "LGW"]
- Paris → ["CDG", "ORY"]
- New York/NYC → ["JFK", "LGA", "EWR"]
- weekends → ["Saturday", "Sunday"]
- short trips → max_pairing_length: 2
- turns/day trips → max_pairing_length: 1, prefer_pairing_type: ["Turn"]
- long trips → prefer_pairing_length: [4, 5]
"""


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(user_input: str, bid_month: str = "2026-01") -> tuple[str, str]:
    """
    Build the complete prompt for the LLM.
    
    Args:
        user_input: The pilot's natural language preferences
        bid_month: Current bid month in YYYY-MM format
        
    Returns:
        (system_message, user_message) tuple for API call
    """
    # System message (will be cached)
    system_message = f"""{SYSTEM_INSTRUCTIONS}

CURRENT BID MONTH: {bid_month}

{SCHEMA_DEFINITION}

EXAMPLES:
{get_examples_for_prompt()}"""
    
    # User message (dynamic, not cached)
    user_message = f"""Extract preferences from this input:

"{user_input}"

Output only valid JSON matching the schema. No explanations."""
    
    return system_message, user_message


def get_system_message(bid_month: str = "2026-01") -> str:
    """
    Get just the system message (for caching).
    
    Args:
        bid_month: Current bid month
        
    Returns:
        Complete system message string
    """
    system_message, _ = build_prompt("", bid_month)
    return system_message


def get_user_message(user_input: str) -> str:
    """
    Get just the user message (dynamic part).
    
    Args:
        user_input: Pilot's natural language preferences
        
    Returns:
        User message string
    """
    return f"""Extract preferences from this input:

"{user_input}"

Output only valid JSON matching the schema. No explanations."""


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (1 token ≈ 4 characters for English).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def get_prompt_stats(bid_month: str = "2026-01") -> dict:
    """
    Get statistics about the prompt size.
    
    Returns:
        Dict with token estimates and cost projections
    """
    system_msg = get_system_message(bid_month)
    
    system_tokens = estimate_tokens(system_msg)
    
    return {
        "system_message_chars": len(system_msg),
        "system_message_tokens_est": system_tokens,
        "user_message_tokens_est": 50,  # Typical user input
        "output_tokens_est": 200,  # Typical output
        "total_tokens_est": system_tokens + 50 + 200,
        "cost_per_request_haiku": round((system_tokens + 50) * 0.0000008 + 200 * 0.000004, 5),
        "cost_per_request_haiku_cached": round(50 * 0.0000008 + 200 * 0.000004, 5),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT TEMPLATE TEST")
    print("=" * 60)
    
    # Get stats
    stats = get_prompt_stats()
    print("\nPrompt Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show example prompt
    print("\n" + "=" * 60)
    print("EXAMPLE PROMPT BUILD")
    print("=" * 60)
    
    system_msg, user_msg = build_prompt("I want Fridays off and Hawaii trips")
    
    print(f"\nSystem Message Length: {len(system_msg)} chars")
    print(f"User Message Length: {len(user_msg)} chars")
    
    print("\n--- USER MESSAGE ---")
    print(user_msg)
    
    print("\n--- SYSTEM MESSAGE (first 500 chars) ---")
    print(system_msg[:500] + "...")