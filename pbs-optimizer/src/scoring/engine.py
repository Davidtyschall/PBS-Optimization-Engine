"""
Scoring Engine
==============
Ranks sequences against pilot preferences.

MVP Coverage: 25 core preference fields
TODO: Expand to all 71 fields in V2
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from src.preferences.schema import (
    PilotPreferences,
    AIRPORT_TO_REGION,
    Region,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_WEIGHTS = {
    "days_off": 1.0,
    "times": 0.8,
    "layovers": 0.6,
    "pairing": 0.5,
    "credit": 0.4,
}


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class CategoryScore:
    """Score for a single category."""
    score: float  # 0-100
    max_score: float  # Usually 100
    weight: float
    weighted_score: float
    factors: List[str] = field(default_factory=list)  # What contributed


@dataclass
class SequenceScore:
    """Complete scoring result for a sequence."""
    sequence_id: int
    seq_num: int
    final_score: float  # 0-100
    rank: int = 0
    breakdown: Dict[str, CategoryScore] = field(default_factory=dict)
    disqualified: bool = False
    disqualification_reason: Optional[str] = None
    explanation: str = ""
    
    # Sequence details for display
    calendar_start_dates: List[int] = field(default_factory=list)
    layover_cities: List[str] = field(default_factory=list)
    pairing_length: int = 0
    total_credit_minutes: int = 0
    total_tafb_minutes: int = 0


@dataclass
class ScoringResult:
    """Complete result of scoring all sequences."""
    success: bool
    ranked_sequences: List[SequenceScore] = field(default_factory=list)
    total_sequences: int = 0
    disqualified_count: int = 0
    preferences_summary: str = ""
    error: Optional[str] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def get_day_of_week(year: int, month: int, day: int) -> str:
    """Get day of week name for a date."""
    try:
        date = datetime(year, month, day)
        return date.strftime("%A")  # "Monday", "Tuesday", etc.
    except ValueError:
        return ""


def get_dates_from_calendar_days(
    calendar_days: List[int],
    bid_year: int = 2026,
    bid_month: int = 1
) -> List[Dict[str, Any]]:
    """
    Convert calendar day numbers to full date info.
    
    Args:
        calendar_days: List of day numbers [7, 11, 15]
        bid_year: Year of bid period
        bid_month: Month of bid period
        
    Returns:
        List of {day: int, date: str, day_of_week: str}
    """
    results = []
    for day in calendar_days:
        day_of_week = get_day_of_week(bid_year, bid_month, day)
        results.append({
            "day": day,
            "date": f"{bid_year}-{bid_month:02d}-{day:02d}",
            "day_of_week": day_of_week
        })
    return results


def get_region_for_city(city_code: str) -> Optional[str]:
    """Get region name for an airport code."""
    region = AIRPORT_TO_REGION.get(city_code.upper())
    return region.value if region else None


# =============================================================================
# CATEGORY SCORING FUNCTIONS
# =============================================================================

def score_days_off(
    prefs: PilotPreferences,
    sequence: Dict[str, Any],
    bid_year: int = 2026,
    bid_month: int = 1
) -> CategoryScore:
    """
    Score sequence against days-off preferences.
    
    Checks:
    - must_off_dates (hard disqualifier)
    - prefer_off_dates (soft penalty)
    - days_off (recurring days like "Friday")
    - maximize_weekends_off
    """
    score = 100.0
    factors = []
    
    days_off_prefs = prefs.days_off
    calendar_days = sequence.get("calendar_start_dates", [])
    pairing_length = sequence.get("pairing_length", 1)
    
    # Build list of all days the sequence occupies
    occupied_dates = []
    for start_day in calendar_days:
        for offset in range(pairing_length):
            occupied_day = start_day + offset
            # Handle month overflow simply (good enough for MVP)
            if occupied_day <= 31:
                occupied_dates.append({
                    "day": occupied_day,
                    "date": f"{bid_year}-{bid_month:02d}-{occupied_day:02d}",
                    "day_of_week": get_day_of_week(bid_year, bid_month, occupied_day)
                })
    
    # Check must_off_dates (HARD CONSTRAINT)
    for must_off in days_off_prefs.must_off_dates:
        for occ in occupied_dates:
            if occ["date"] == must_off:
                factors.append(f"DISQUALIFIED: Conflicts with must-off date {must_off}")
                return CategoryScore(
                    score=0,
                    max_score=100,
                    weight=DEFAULT_WEIGHTS["days_off"],
                    weighted_score=0,
                    factors=factors
                )
    
    # Check prefer_off_dates (soft penalty)
    for prefer_off in days_off_prefs.prefer_off_dates:
        for occ in occupied_dates:
            if occ["date"] == prefer_off:
                score -= 20
                factors.append(f"-20: Conflicts with preferred off date {prefer_off}")
    
    # Check recurring days off
    strength = days_off_prefs.days_off_strength
    penalty_map = {
        "must_have": 100,  # Effectively disqualifies
        "strong": 40,
        "preferred": 25,
        "weak": 10
    }
    penalty = penalty_map.get(strength, 25)
    
    for day_off in days_off_prefs.days_off:
        for occ in occupied_dates:
            if occ["day_of_week"] == day_off:
                score -= penalty
                factors.append(f"-{penalty}: Works on {day_off} ({occ['date']})")
                break  # Only penalize once per day type
    
    # Bonus for maximize_weekends_off
    if days_off_prefs.maximize_weekends_off:
        weekend_conflicts = 0
        for occ in occupied_dates:
            if occ["day_of_week"] in ["Saturday", "Sunday"]:
                weekend_conflicts += 1
        if weekend_conflicts == 0:
            score += 10
            factors.append("+10: No weekend conflicts")
        else:
            score -= weekend_conflicts * 10
            factors.append(f"-{weekend_conflicts * 10}: {weekend_conflicts} weekend day(s) worked")
    
    score = clamp(score, 0, 100)
    
    return CategoryScore(
        score=score,
        max_score=100,
        weight=DEFAULT_WEIGHTS["days_off"],
        weighted_score=score * DEFAULT_WEIGHTS["days_off"],
        factors=factors if factors else ["No days-off conflicts"]
    )


def score_times(
    prefs: PilotPreferences,
    sequence: Dict[str, Any]
) -> CategoryScore:
    """
    Score sequence against time preferences.
    
    Checks:
    - report_after_minutes
    - report_before_minutes
    - release_before_minutes
    - release_after_minutes
    - is_commuter (applies stricter defaults)
    """
    score = 100.0
    factors = []
    
    times_prefs = prefs.times
    report_time = sequence.get("report_time_minutes", 480)  # Default 8am
    release_time = sequence.get("release_time_minutes", 1020)  # Default 5pm
    
    # Apply commuter defaults if commuter but no explicit times set
    report_after = times_prefs.report_after_minutes
    release_before = times_prefs.release_before_minutes
    
    if times_prefs.is_commuter:
        if report_after is None:
            report_after = 540  # 9am default for commuters
        if release_before is None:
            release_before = 1140  # 7pm default for commuters
    
    # Check report_after_minutes
    if report_after is not None:
        if report_time < report_after:
            diff = report_after - report_time
            penalty = min(40, diff // 15 * 10)  # 10 points per 15 min early
            score -= penalty
            factors.append(f"-{penalty}: Reports too early ({_minutes_to_time(report_time)} < {_minutes_to_time(report_after)})")
    
    # Check report_before_minutes
    if times_prefs.report_before_minutes is not None:
        if report_time > times_prefs.report_before_minutes:
            diff = report_time - times_prefs.report_before_minutes
            penalty = min(30, diff // 15 * 10)
            score -= penalty
            factors.append(f"-{penalty}: Reports too late ({_minutes_to_time(report_time)} > {_minutes_to_time(times_prefs.report_before_minutes)})")
    
    # Check release_before_minutes
    if release_before is not None:
        if release_time > release_before:
            diff = release_time - release_before
            penalty = min(40, diff // 15 * 10)
            score -= penalty
            factors.append(f"-{penalty}: Releases too late ({_minutes_to_time(release_time)} > {_minutes_to_time(release_before)})")
    
    # Check release_after_minutes
    if times_prefs.release_after_minutes is not None:
        if release_time < times_prefs.release_after_minutes:
            diff = times_prefs.release_after_minutes - release_time
            penalty = min(30, diff // 15 * 10)
            score -= penalty
            factors.append(f"-{penalty}: Releases too early ({_minutes_to_time(release_time)} < {_minutes_to_time(times_prefs.release_after_minutes)})")
    
    score = clamp(score, 0, 100)
    
    return CategoryScore(
        score=score,
        max_score=100,
        weight=DEFAULT_WEIGHTS["times"],
        weighted_score=score * DEFAULT_WEIGHTS["times"],
        factors=factors if factors else ["Times within constraints"]
    )


def score_layovers(
    prefs: PilotPreferences,
    sequence: Dict[str, Any]
) -> CategoryScore:
    """
    Score sequence against layover preferences.
    
    Checks:
    - prefer_layover_city
    - avoid_layover_city
    - prefer_region
    - avoid_region
    """
    score = 50.0  # Neutral starting point
    factors = []
    
    layover_prefs = prefs.layovers
    layover_cities = sequence.get("layover_cities") or []
    
    # No layovers - neutral unless they wanted layovers
    if not layover_cities:
        if layover_prefs.prefer_layover_city or layover_prefs.prefer_region:
            score = 30
            factors.append("-20: No layovers (wanted specific destinations)")
        else:
            factors.append("No layovers (neutral)")
        
        return CategoryScore(
            score=score,
            max_score=100,
            weight=DEFAULT_WEIGHTS["layovers"],
            weighted_score=score * DEFAULT_WEIGHTS["layovers"],
            factors=factors
        )
    
    # Check each layover city
    for city in layover_cities:
        city_upper = city.upper()
        
        # Preferred cities
        if city_upper in [c.upper() for c in layover_prefs.prefer_layover_city]:
            score += 25
            factors.append(f"+25: Layover in preferred city {city_upper}")
        
        # Avoided cities
        if city_upper in [c.upper() for c in layover_prefs.avoid_layover_city]:
            score -= 40
            factors.append(f"-40: Layover in avoided city {city_upper}")
        
        # Region preferences
        city_region = get_region_for_city(city_upper)
        if city_region:
            if city_region in layover_prefs.prefer_region:
                score += 15
                factors.append(f"+15: Layover in preferred region {city_region}")
            if city_region in layover_prefs.avoid_region:
                score -= 30
                factors.append(f"-30: Layover in avoided region {city_region}")
    
    score = clamp(score, 0, 100)
    
    return CategoryScore(
        score=score,
        max_score=100,
        weight=DEFAULT_WEIGHTS["layovers"],
        weighted_score=score * DEFAULT_WEIGHTS["layovers"],
        factors=factors
    )


def score_pairing(
    prefs: PilotPreferences,
    sequence: Dict[str, Any]
) -> CategoryScore:
    """
    Score sequence against pairing preferences.
    
    Checks:
    - prefer_pairing_length
    - min_pairing_length
    - max_pairing_length
    - prefer_pairing_type
    - avoid_pairing_type
    """
    score = 50.0  # Neutral
    factors = []
    
    pairing_prefs = prefs.pairing
    pairing_length = sequence.get("pairing_length", 1)
    is_redeye = sequence.get("is_redeye", False)
    is_transoceanic = sequence.get("is_transoceanic", False)
    
    # Pairing length preferences
    if pairing_prefs.prefer_pairing_length:
        if pairing_length in pairing_prefs.prefer_pairing_length:
            score += 20
            factors.append(f"+20: Preferred trip length ({pairing_length} days)")
        else:
            score -= 10
            factors.append(f"-10: Not preferred length ({pairing_length} days, wanted {pairing_prefs.prefer_pairing_length})")
    
    # Min pairing length
    if pairing_prefs.min_pairing_length is not None:
        if pairing_length < pairing_prefs.min_pairing_length:
            score -= 25
            factors.append(f"-25: Too short ({pairing_length} < {pairing_prefs.min_pairing_length} days)")
    
    # Max pairing length
    if pairing_prefs.max_pairing_length is not None:
        if pairing_length > pairing_prefs.max_pairing_length:
            score -= 30
            factors.append(f"-30: Too long ({pairing_length} > {pairing_prefs.max_pairing_length} days)")
    
    # Avoid pairing types
    avoided_types = [t.lower() for t in pairing_prefs.avoid_pairing_type]
    
    if is_redeye and ("redeye" in avoided_types or "trailing redeye" in avoided_types):
        score -= 40
        factors.append("-40: Is a redeye (avoided)")
    
    if is_transoceanic and "transoceanic" in avoided_types:
        score -= 30
        factors.append("-30: Is transoceanic (avoided)")
    
    # Prefer pairing types
    preferred_types = [t.lower() for t in pairing_prefs.prefer_pairing_type]
    
    if is_transoceanic and "transoceanic" in preferred_types:
        score += 20
        factors.append("+20: Is transoceanic (preferred)")
    
    if pairing_length == 1 and "turn" in preferred_types:
        score += 20
        factors.append("+20: Is a turn/day trip (preferred)")
    
    score = clamp(score, 0, 100)
    
    return CategoryScore(
        score=score,
        max_score=100,
        weight=DEFAULT_WEIGHTS["pairing"],
        weighted_score=score * DEFAULT_WEIGHTS["pairing"],
        factors=factors if factors else ["Pairing characteristics neutral"]
    )


def score_credit(
    prefs: PilotPreferences,
    sequence: Dict[str, Any],
    all_sequences: List[Dict[str, Any]] = None
) -> CategoryScore:
    """
    Score sequence against credit preferences.
    
    Checks:
    - maximize_credit
    - minimize_credit
    - minimize_tafb
    - prefer_efficient_flying
    - target_credit_min/max
    """
    score = 50.0  # Neutral
    factors = []
    
    credit_prefs = prefs.credit
    credit_minutes = sequence.get("total_credit_minutes", 0)
    tafb_minutes = sequence.get("total_tafb_minutes", 0)
    
    # Credit targets
    if credit_prefs.target_credit_min_minutes is not None:
        if credit_minutes >= credit_prefs.target_credit_min_minutes:
            score += 15
            factors.append(f"+15: Credit meets minimum target")
        else:
            score -= 15
            factors.append(f"-15: Credit below minimum target")
    
    if credit_prefs.target_credit_max_minutes is not None:
        if credit_minutes <= credit_prefs.target_credit_max_minutes:
            score += 10
            factors.append(f"+10: Credit within maximum target")
        else:
            score -= 20
            factors.append(f"-20: Credit exceeds maximum target")
    
    # Efficiency (TAFB/Credit ratio)
    if credit_prefs.prefer_efficient_flying and credit_minutes > 0:
        ratio = tafb_minutes / credit_minutes
        if ratio < 2.0:
            score += 25
            factors.append(f"+25: Very efficient (TAFB/Credit ratio: {ratio:.2f})")
        elif ratio < 2.5:
            score += 15
            factors.append(f"+15: Efficient (TAFB/Credit ratio: {ratio:.2f})")
        elif ratio < 3.0:
            score += 5
            factors.append(f"+5: Moderately efficient (TAFB/Credit ratio: {ratio:.2f})")
        else:
            score -= 10
            factors.append(f"-10: Inefficient (TAFB/Credit ratio: {ratio:.2f})")
    
    # Minimize TAFB
    if credit_prefs.minimize_tafb and tafb_minutes > 0:
        # Lower TAFB is better - use relative scoring
        # For MVP, use simple thresholds
        hours = tafb_minutes / 60
        if hours < 20:
            score += 20
            factors.append(f"+20: Low TAFB ({hours:.1f} hours)")
        elif hours < 40:
            score += 10
            factors.append(f"+10: Moderate TAFB ({hours:.1f} hours)")
        else:
            score -= 10
            factors.append(f"-10: High TAFB ({hours:.1f} hours)")
    
    # Maximize/Minimize credit flags
    # These are better handled with relative scoring across all sequences
    # For MVP, use simple bonuses
    if credit_prefs.maximize_credit and credit_minutes > 0:
        hours = credit_minutes / 60
        if hours > 20:
            score += 15
            factors.append(f"+15: High credit ({hours:.1f} hours)")
        elif hours > 10:
            score += 5
            factors.append(f"+5: Moderate credit ({hours:.1f} hours)")
    
    if credit_prefs.minimize_credit and credit_minutes > 0:
        hours = credit_minutes / 60
        if hours < 10:
            score += 15
            factors.append(f"+15: Low credit ({hours:.1f} hours)")
        elif hours < 15:
            score += 5
            factors.append(f"+5: Moderate-low credit ({hours:.1f} hours)")
    
    score = clamp(score, 0, 100)
    
    return CategoryScore(
        score=score,
        max_score=100,
        weight=DEFAULT_WEIGHTS["credit"],
        weighted_score=score * DEFAULT_WEIGHTS["credit"],
        factors=factors if factors else ["Credit characteristics neutral"]
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _minutes_to_time(minutes: int) -> str:
    """Convert minutes from midnight to readable time."""
    hours = minutes // 60
    mins = minutes % 60
    period = "AM" if hours < 12 else "PM"
    display_hour = hours % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{mins:02d} {period}"


def generate_explanation(score_result: SequenceScore) -> str:
    """Generate human-readable explanation of score."""
    lines = []
    
    if score_result.disqualified:
        return f"DISQUALIFIED: {score_result.disqualification_reason}"
    
    lines.append(f"Score: {score_result.final_score:.1f}/100")
    lines.append("")
    
    # Top factors (positive)
    positive_factors = []
    negative_factors = []
    
    for category, cat_score in score_result.breakdown.items():
        for factor in cat_score.factors:
            if factor.startswith("+"):
                positive_factors.append(factor)
            elif factor.startswith("-"):
                negative_factors.append(factor)
    
    if positive_factors:
        lines.append("Strengths:")
        for f in positive_factors[:3]:  # Top 3
            lines.append(f"  {f}")
    
    if negative_factors:
        lines.append("Weaknesses:")
        for f in negative_factors[:3]:  # Top 3
            lines.append(f"  {f}")
    
    return "\n".join(lines)


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def score_sequence(
    prefs: PilotPreferences,
    sequence: Dict[str, Any],
    bid_year: int = 2026,
    bid_month: int = 1
) -> SequenceScore:
    """
    Score a single sequence against preferences.
    
    Args:
        prefs: Pilot preferences
        sequence: Sequence data from database
        bid_year: Year of bid period
        bid_month: Month of bid period
        
    Returns:
        SequenceScore with breakdown
    """
    result = SequenceScore(
        sequence_id=sequence.get("id", 0),
        seq_num=sequence.get("seq_num", 0),
        final_score=0,
        calendar_start_dates=sequence.get("calendar_start_dates", []),
        layover_cities=sequence.get("layover_cities") or [],
        pairing_length=sequence.get("pairing_length", 1),
        total_credit_minutes=sequence.get("total_credit_minutes", 0),
        total_tafb_minutes=sequence.get("total_tafb_minutes", 0),
    )
    
# Get weights from preferences (or use defaults)
    weights = {
        "days_off": prefs.priority_weights.get("days_off", 1.0),
        "times": prefs.priority_weights.get("times", 0.8),
        "layovers": prefs.priority_weights.get("layovers", 0.6),
        "pairing": prefs.priority_weights.get("pairing", 0.5),
        "credit": prefs.priority_weights.get("credit", 0.4),
    }
    
    # Score each category
    days_off_score = score_days_off(prefs, sequence, bid_year, bid_month)
    days_off_score.weight = weights["days_off"]
    days_off_score.weighted_score = days_off_score.score * weights["days_off"]
    
    times_score = score_times(prefs, sequence)
    times_score.weight = weights["times"]
    times_score.weighted_score = times_score.score * weights["times"]
    
    layovers_score = score_layovers(prefs, sequence)
    layovers_score.weight = weights["layovers"]
    layovers_score.weighted_score = layovers_score.score * weights["layovers"]
    
    pairing_score = score_pairing(prefs, sequence)
    pairing_score.weight = weights["pairing"]
    pairing_score.weighted_score = pairing_score.score * weights["pairing"]
    
    credit_score = score_credit(prefs, sequence)
    credit_score.weight = weights["credit"]
    credit_score.weighted_score = credit_score.score * weights["credit"]
    
    result.breakdown = {
        "days_off": days_off_score,
        "times": times_score,
        "layovers": layovers_score,
        "pairing": pairing_score,
        "credit": credit_score,
    }
    
    # Check for disqualification
    if days_off_score.score == 0 and any("DISQUALIFIED" in f for f in days_off_score.factors):
        result.disqualified = True
        result.disqualification_reason = next(
            f for f in days_off_score.factors if "DISQUALIFIED" in f
        )
        result.final_score = 0
        result.explanation = generate_explanation(result)
        return result
    
    # Calculate weighted total
    total_weighted = sum(cat.weighted_score for cat in result.breakdown.values())
    max_possible = sum(cat.weight * 100 for cat in result.breakdown.values())
    
    # Normalize to 0-100
    if max_possible > 0:
        result.final_score = (total_weighted / max_possible) * 100
    else:
        result.final_score = 0
    
    result.explanation = generate_explanation(result)
    
    return result


def score_sequences(
    prefs: PilotPreferences,
    sequences: List[Dict[str, Any]],
    bid_year: int = 2026,
    bid_month: int = 1
) -> ScoringResult:
    """
    Score and rank all sequences against preferences.
    
    Args:
        prefs: Pilot preferences
        sequences: List of sequences from database
        bid_year: Year of bid period
        bid_month: Month of bid period
        
    Returns:
        ScoringResult with ranked sequences
    """
    if not sequences:
        return ScoringResult(
            success=False,
            error="No sequences provided"
        )
    
    # Score each sequence
    scored = []
    disqualified_count = 0
    
    for seq in sequences:
        score_result = score_sequence(prefs, seq, bid_year, bid_month)
        scored.append(score_result)
        if score_result.disqualified:
            disqualified_count += 1
    
    # Sort by score descending (disqualified at the end)
    scored.sort(key=lambda x: (not x.disqualified, x.final_score), reverse=True)
    
    # Assign ranks
    for i, s in enumerate(scored, 1):
        s.rank = i
    
    return ScoringResult(
        success=True,
        ranked_sequences=scored,
        total_sequences=len(sequences),
        disqualified_count=disqualified_count,
        preferences_summary=prefs.summary()
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from src.preferences.schema import PilotPreferences, DaysOffPreferences, LayoverPreferences, PairingPreferences, TimePreferences, CreditPreferences
    
    print("=" * 60)
    print("SCORING ENGINE TEST")
    print("=" * 60)
    
    # Create test preferences
    prefs = PilotPreferences(
        days_off=DaysOffPreferences(
            days_off=["Friday"],
            days_off_strength="preferred"
        ),
        layovers=LayoverPreferences(
            prefer_layover_city=["HNL", "OGG", "LIH", "KOA"]
        ),
        pairing=PairingPreferences(
            avoid_pairing_type=["Redeye", "Trailing Redeye"],
            max_pairing_length=3
        ),
        times=TimePreferences(
            release_before_minutes=1080  # 6pm
        )
    )
    
    print("\nPreferences:")
    print(prefs.summary())
    
    # Create test sequences (mimicking database format)
    test_sequences = [
        {
            "id": 1,
            "seq_num": 224,
            "calendar_start_dates": [1, 3, 4],  # Wed, Fri, Sat
            "layover_cities": ["OGG"],
            "pairing_length": 2,
            "is_redeye": False,
            "is_transoceanic": False,
            "report_time_minutes": 480,  # 8am
            "release_time_minutes": 1020,  # 5pm
            "total_credit_minutes": 345,
            "total_tafb_minutes": 665,
        },
        {
            "id": 2,
            "seq_num": 217,
            "calendar_start_dates": [7, 11],  # Tue, Sat
            "layover_cities": None,  # Day trip
            "pairing_length": 1,
            "is_redeye": False,
            "is_transoceanic": False,
            "report_time_minutes": 480,
            "release_time_minutes": 960,  # 4pm
            "total_credit_minutes": 345,
            "total_tafb_minutes": 345,
        },
        {
            "id": 3,
            "seq_num": 225,
            "calendar_start_dates": [12, 13],  # Fri, Sat
            "layover_cities": ["ICN"],  # Seoul
            "pairing_length": 2,
            "is_redeye": True,  # Redeye!
            "is_transoceanic": True,
            "report_time_minutes": 1200,  # 8pm
            "release_time_minutes": 600,  # 10am next day
            "total_credit_minutes": 922,
            "total_tafb_minutes": 1715,
        },
    ]
    
    # Score sequences
    result = score_sequences(prefs, test_sequences)
    
    print("\n" + "=" * 60)
    print("RANKING RESULTS")
    print("=" * 60)
    
    for seq_score in result.ranked_sequences:
        print(f"\n#{seq_score.rank} - SEQ {seq_score.seq_num}")
        print(f"   Score: {seq_score.final_score:.1f}/100")
        print(f"   Layovers: {seq_score.layover_cities or 'None'}")
        print(f"   Length: {seq_score.pairing_length} days")
        
        if seq_score.disqualified:
            print(f"   DISQUALIFIED: {seq_score.disqualification_reason}")
        else:
            print("   Breakdown:")
            for cat, cat_score in seq_score.breakdown.items():
                print(f"      {cat}: {cat_score.score:.0f}/100 (weighted: {cat_score.weighted_score:.1f})")
    
    print(f"\nTotal: {result.total_sequences} sequences")
    print(f"Disqualified: {result.disqualified_count}")