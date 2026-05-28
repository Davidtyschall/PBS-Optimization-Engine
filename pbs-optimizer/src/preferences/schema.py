'''
Docstring for pbs-optimizer.src.preferences.schema
single source of truth for preference types
contains all validation logic in one place
heavily documented, and includes tests
'''

"""
PBS Preference Schema
=====================
Complete, production-ready schema for pilot scheduling preferences.

This module defines:
- All valid preference types with strict typing
- Validation functions for all data types
- Conflict detection logic
- Serialization/deserialization
- Synonym mappings for NL parsing

Version: 1.0.0
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple, Set
from enum import Enum
from datetime import datetime
import re
import json


# ============================================================================
# ENUMS - Valid Values
# ============================================================================

class DayOfWeek(str, Enum):
    """Days of the week."""
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"


class PairingType(str, Enum):
    """Types of pairings/trips."""
    EARLY = "Early"                      # Report 0400-0659
    MORNING = "Morning"                  # Report 0700-1159
    MIDDAY = "Midday"                    # Report 1000-1559
    EVENING = "Evening"                  # Report ≥1500, release <2300
    LATE = "Late"                        # Report ≥1500, release 2300-0659
    REDEYE = "Redeye"                    # Report <0330, arrive ≥0400
    TRAILING_REDEYE = "Trailing Redeye"  # Final duty is redeye
    ODAN = "ODAN"                        # On Duty All Night
    SPLIT_DUTY = "Split Duty"            # Duty spans 0100-0500 with mid-rest
    TRANSCON = "Transcon"                # Single leg >4.5hrs, ≥3hr timezone change
    TRANSOCEANIC = "Transoceanic"        # Crosses ocean/continents
    DEEP_SOUTH = "Deep South"            # South of equator
    ROCKET = "Rocket"                    # Transoceanic + redeye + <18hr layover
    SHUTTLE = "Shuttle"                  # DCA/LGA/BOS only
    CHARTER = "Charter"                  # 9000-series flights
    TURN = "Turn"                        # Same-day return (1-day trip)


class Region(str, Enum):
    """Geographic regions for layover preferences."""
    # US Domestic
    NORTHEAST = "Northeast"
    SOUTHEAST = "Southeast"
    NORTH_CENTRAL = "North Central"
    SOUTH_CENTRAL = "South Central"
    NORTHWEST = "Northwest"
    SOUTHWEST = "Southwest"
    
    # US Special
    HAWAII = "Hawaii"
    ALASKA = "Alaska"
    
    # International
    CANADA = "Canada"
    MEXICO = "Mexico"
    CARIBBEAN = "Caribbean"
    CENTRAL_AMERICA = "Central America"
    SOUTH_AMERICA = "South America"
    EUROPE = "Europe"
    AFRICA = "Africa"
    MIDDLE_EAST = "Middle East"
    ASIA = "Asia"
    AUSTRALIA_NZ = "Australia/NZ"


class ReserveType(str, Enum):
    """Reserve duty types."""
    LONG_CALL = "Long Call"
    SHORT_CALL = "Short Call"
    EITHER = "Either"


class PreferenceStrength(str, Enum):
    """How important a preference is."""
    MUST_HAVE = "must_have"    # Hard constraint - won't accept without
    STRONG = "strong"          # High priority
    PREFERRED = "preferred"    # Nice to have
    WEAK = "weak"              # If possible, low priority


class AwardedType(str, Enum):
    """Types of awarded schedules."""
    LINEHOLDER = "Lineholder"
    RESERVE_LC = "Reserve Long Call"
    RESERVE_SC = "Reserve Short Call"
    HYBRID = "Hybrid"


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

VALIDATION_RANGES = {
    "pairing_length": {"min": 1, "max": 6},
    "duty_periods": {"min": 1, "max": 5},
    "legs_per_duty": {"min": 1, "max": 8},
    "tafb_credit_ratio": {"min": 1.0, "max": 5.0},
    "credit_minutes": {"min": 3000, "max": 6600},  # 50-110 hours
    "layover_hours": {"min": 8, "max": 96},
    "time_minutes": {"min": 0, "max": 1439},  # 00:00-23:59
    "work_block_days": {"min": 1, "max": 7},
    "consecutive_days_off": {"min": 1, "max": 14},
    "buddy_bid_count": {"min": 0, "max": 3},
    "reserve_block_days": {"min": 1, "max": 6},
    "fly_through_days": {"min": 0, "max": 5},
}

# Date pattern: YYYY-MM-DD
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# Airport code pattern: 3 uppercase letters
AIRPORT_PATTERN = re.compile(r'^[A-Z]{3}$')


# ============================================================================
# AIRPORT AND REGION DATA
# ============================================================================

VALID_AIRPORT_CODES: Set[str] = {
    # US Major Hubs
    "DFW", "ORD", "LAX", "JFK", "MIA", "ATL", "SFO", "SEA", "DEN", "PHX",
    "LGA", "EWR", "BOS", "IAD", "DCA", "CLT", "PHL", "IAH", "LAS", "MCO",
    
    # US Regional
    "AUS", "SAT", "MSY", "SAN", "PDX", "SLC", "MSP", "DTW", "BNA", "RDU",
    "IND", "CMH", "PIT", "STL", "MCI", "OKC", "TUL", "ABQ", "ELP", "TUS",
    "BWI", "FLL", "TPA", "RSW", "JAX", "PBI", "SJC", "OAK", "SMF", "BUR",
    "SNA", "ONT", "PSP", "COS", "BOI", "GEG", "RNO", "TUS", "ALB", "BDL",
    "BUF", "ROC", "SYR", "PVD", "MHT", "PWM", "ORF", "RIC", "CHS", "SAV",
    "MYR", "GSP", "AVL", "TYS", "BHM", "HSV", "MOB", "PNS", "VPS", "ECP",
    "SRQ", "PIE", "DAB", "MLB", "SDF", "CVG", "DAY", "LEX", "CLE", "CAK",
    "MKE", "MDW", "GRR", "LAN", "FNT", "DSM", "OMA", "ICT", "LIT", "XNA",
    "MEM", "SGF", "CID", "MLI", "PIA", "FAR", "FSD", "RAP", "BIS", "GFK",
    
    # Hawaii
    "HNL", "OGG", "LIH", "KOA", "ITO",
    
    # Alaska
    "ANC", "FAI", "JNU", "KTN", "SIT",
    
    # Canada
    "YYZ", "YVR", "YUL", "YYC", "YEG", "YOW", "YWG", "YHZ", "YQB",
    
    # Mexico
    "MEX", "GDL", "CUN", "SJD", "PVR", "MZT", "ZIH", "ACA", "MTY", "TIJ",
    
    # Caribbean
    "SJU", "STT", "STX", "SXM", "AUA", "CUR", "BDA", "NAS", "FPO", "MBJ",
    "KIN", "PUJ", "SDQ", "HAV", "GCM", "PLS", "EIS", "ANU", "BGI", "UVF",
    
    # Central America
    "GUA", "SAL", "TGU", "MGA", "SJO", "LIR", "PTY", "BZE",
    
    # South America
    "GRU", "GIG", "BSB", "CNF", "SSA", "REC", "FOR", "POA", "CWB",
    "EZE", "AEP", "COR", "MDZ", "SCL", "LIM", "CUZ", "BOG", "MDE", "CTG",
    "CLO", "UIO", "GYE", "CCS", "MAR", "LPB", "VVI", "ASU", "MVD",
    
    # Europe
    "LHR", "LGW", "STN", "MAN", "EDI", "GLA", "BHX", "BRS", "NCL",
    "CDG", "ORY", "LYS", "NCE", "MRS", "TLS", "BOD",
    "FRA", "MUC", "DUS", "TXL", "HAM", "STR", "CGN", "HAJ",
    "AMS", "BRU", "LUX",
    "MAD", "BCN", "PMI", "AGP", "VLC", "SVQ", "BIO", "IBZ",
    "FCO", "MXP", "LIN", "VCE", "NAP", "FLR", "BGY", "PSA",
    "ZRH", "GVA", "BSL",
    "VIE", "SZG",
    "LIS", "OPO", "FAO",
    "DUB", "SNN", "ORK",
    "CPH", "OSL", "ARN", "GOT", "HEL",
    "WAW", "KRK", "PRG", "BUD", "OTP", "SOF", "ATH", "SKG",
    "IST", "SAW", "ESB", "AYT", "ADB",
    "SVO", "DME", "LED",
    "KEF",
    
    # Middle East
    "DXB", "AUH", "DOH", "BAH", "KWI", "MCT", "AMM", "BEY", "TLV", "CAI",
    "JED", "RUH", "DMM",
    
    # Africa
    "JNB", "CPT", "DUR", "NBO", "ADD", "LOS", "ABV", "ACC", "DSS", "CMN",
    "RAK", "TUN", "ALG", "CAI", "LXR", "HRG", "SSH", "MRU", "TNR",
    
    # Asia
    "NRT", "HND", "KIX", "NGO", "FUK", "CTS", "OKA",
    "ICN", "GMP", "PUS",
    "PVG", "PEK", "PKX", "CAN", "SZX", "CTU", "XIY", "HGH", "NKG", "WUH",
    "HKG", "TPE", "KHH",
    "MNL", "CEB",
    "SIN", "KUL", "PEN", "BKK", "DMK", "HKT", "CNX", "SGN", "HAN", "DAD",
    "CGK", "DPS",
    "DEL", "BOM", "MAA", "BLR", "HYD", "CCU", "COK", "AMD",
    "CMB", "MLE", "KTM", "DAC",
    
    # Australia/NZ
    "SYD", "MEL", "BNE", "PER", "ADL", "CBR", "OOL", "CNS", "HBA",
    "AKL", "WLG", "CHC", "ZQN",
    
    # Pacific
    "PPT", "NOU", "NAN", "APW", "GUM", "SPN",
}

REGION_TO_AIRPORTS: Dict[Region, List[str]] = {
    Region.NORTHEAST: [
        "JFK", "LGA", "EWR", "BOS", "PHL", "DCA", "IAD", "BWI", "BDL", "PVD",
        "ALB", "BUF", "ROC", "SYR", "MHT", "PWM"
    ],
    Region.SOUTHEAST: [
        "ATL", "MIA", "FLL", "TPA", "MCO", "CLT", "RDU", "JAX", "PBI", "RSW",
        "SAV", "CHS", "MYR", "ORF", "RIC", "BHM", "HSV", "MOB", "PNS", "VPS"
    ],
    Region.NORTH_CENTRAL: [
        "ORD", "MDW", "DTW", "MSP", "MKE", "CLE", "CVG", "CMH", "IND", "SDF",
        "GRR", "DSM", "OMA", "MCI", "STL", "FSD", "FAR", "BIS"
    ],
    Region.SOUTH_CENTRAL: [
        "DFW", "IAH", "AUS", "SAT", "MSY", "OKC", "TUL", "LIT", "MEM", "BNA",
        "XNA", "SHV", "ELP"
    ],
    Region.NORTHWEST: [
        "SEA", "PDX", "SLC", "BOI", "GEG", "FAI"
    ],
    Region.SOUTHWEST: [
        "LAX", "SFO", "SAN", "LAS", "PHX", "DEN", "ABQ", "TUS", "SJC", "OAK",
        "SMF", "BUR", "SNA", "ONT", "PSP", "RNO", "COS"
    ],
    Region.HAWAII: ["HNL", "OGG", "LIH", "KOA", "ITO"],
    Region.ALASKA: ["ANC", "FAI", "JNU", "KTN", "SIT"],
    Region.CANADA: ["YYZ", "YVR", "YUL", "YYC", "YEG", "YOW", "YWG", "YHZ", "YQB"],
    Region.MEXICO: ["MEX", "GDL", "CUN", "SJD", "PVR", "MZT", "ZIH", "ACA", "MTY"],
    Region.CARIBBEAN: [
        "SJU", "STT", "STX", "SXM", "AUA", "CUR", "NAS", "MBJ", "KIN", "PUJ",
        "SDQ", "GCM", "PLS", "ANU", "BGI", "UVF"
    ],
    Region.CENTRAL_AMERICA: ["GUA", "SAL", "TGU", "MGA", "SJO", "LIR", "PTY", "BZE"],
    Region.SOUTH_AMERICA: [
        "GRU", "GIG", "EZE", "SCL", "LIM", "BOG", "MDE", "CCS", "UIO", "CUZ"
    ],
    Region.EUROPE: [
        "LHR", "CDG", "FRA", "AMS", "MAD", "FCO", "MUC", "ZRH", "DUB", "BCN",
        "LIS", "VIE", "CPH", "OSL", "ARN", "HEL", "PRG", "BUD", "ATH", "IST"
    ],
    Region.MIDDLE_EAST: ["DXB", "DOH", "AUH", "TLV", "AMM", "BAH", "KWI", "JED", "RUH"],
    Region.AFRICA: ["JNB", "CPT", "NBO", "ADD", "LOS", "CMN", "CAI", "MRU"],
    Region.ASIA: [
        "NRT", "HND", "ICN", "PVG", "PEK", "HKG", "SIN", "BKK", "KUL", "MNL",
        "CGK", "DEL", "BOM", "TPE"
    ],
    Region.AUSTRALIA_NZ: ["SYD", "MEL", "BNE", "PER", "AKL", "WLG", "CHC"],
}

# Reverse lookup: airport -> region
AIRPORT_TO_REGION: Dict[str, Region] = {}
for region, airports in REGION_TO_AIRPORTS.items():
    for airport in airports:
        AIRPORT_TO_REGION[airport] = region


# ============================================================================
# SYNONYM MAPPINGS (for NL parsing)
# ============================================================================

DAY_SYNONYMS: Dict[str, List[DayOfWeek]] = {
    # Single days
    "monday": [DayOfWeek.MONDAY],
    "mon": [DayOfWeek.MONDAY],
    "tuesday": [DayOfWeek.TUESDAY],
    "tue": [DayOfWeek.TUESDAY],
    "tues": [DayOfWeek.TUESDAY],
    "wednesday": [DayOfWeek.WEDNESDAY],
    "wed": [DayOfWeek.WEDNESDAY],
    "thursday": [DayOfWeek.THURSDAY],
    "thu": [DayOfWeek.THURSDAY],
    "thur": [DayOfWeek.THURSDAY],
    "thurs": [DayOfWeek.THURSDAY],
    "friday": [DayOfWeek.FRIDAY],
    "fri": [DayOfWeek.FRIDAY],
    "saturday": [DayOfWeek.SATURDAY],
    "sat": [DayOfWeek.SATURDAY],
    "sunday": [DayOfWeek.SUNDAY],
    "sun": [DayOfWeek.SUNDAY],
    
    # Groups
    "weekends": [DayOfWeek.SATURDAY, DayOfWeek.SUNDAY],
    "weekend": [DayOfWeek.SATURDAY, DayOfWeek.SUNDAY],
    "weekdays": [DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY,
                 DayOfWeek.THURSDAY, DayOfWeek.FRIDAY],
    "weekday": [DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY,
                DayOfWeek.THURSDAY, DayOfWeek.FRIDAY],
}

PAIRING_TYPE_SYNONYMS: Dict[str, List[PairingType]] = {
    # Redeye variations
    "redeye": [PairingType.REDEYE],
    "red-eye": [PairingType.REDEYE],
    "red eye": [PairingType.REDEYE],
    "redeyes": [PairingType.REDEYE, PairingType.TRAILING_REDEYE],
    "red-eyes": [PairingType.REDEYE, PairingType.TRAILING_REDEYE],
    "red eyes": [PairingType.REDEYE, PairingType.TRAILING_REDEYE],
    "all redeyes": [PairingType.REDEYE, PairingType.TRAILING_REDEYE],
    "trailing redeye": [PairingType.TRAILING_REDEYE],
    
    # Time-based
    "early": [PairingType.EARLY],
    "early morning": [PairingType.EARLY],
    "morning": [PairingType.MORNING],
    "midday": [PairingType.MIDDAY],
    "afternoon": [PairingType.MIDDAY],
    "evening": [PairingType.EVENING],
    "late": [PairingType.LATE],
    "night": [PairingType.LATE, PairingType.REDEYE],
    "overnight": [PairingType.ODAN, PairingType.REDEYE],
    "all-nighter": [PairingType.ODAN],
    "all nighter": [PairingType.ODAN],
    
    # Route-based
    "transcon": [PairingType.TRANSCON],
    "transcontinental": [PairingType.TRANSCON],
    "transoceanic": [PairingType.TRANSOCEANIC],
    "transatlantic": [PairingType.TRANSOCEANIC],
    "transpacific": [PairingType.TRANSOCEANIC],
    "international": [PairingType.TRANSOCEANIC],
    "deep south": [PairingType.DEEP_SOUTH],
    "southern hemisphere": [PairingType.DEEP_SOUTH],
    
    # Special types
    "rocket": [PairingType.ROCKET],
    "shuttle": [PairingType.SHUTTLE],
    "charter": [PairingType.CHARTER],
    "turn": [PairingType.TURN],
    "turns": [PairingType.TURN],
    "day trip": [PairingType.TURN],
    "day trips": [PairingType.TURN],
    "split duty": [PairingType.SPLIT_DUTY],
    "odan": [PairingType.ODAN],
}

CITY_SYNONYMS: Dict[str, List[str]] = {
    # Hawaii
    "hawaii": ["HNL", "OGG", "LIH", "KOA"],
    "honolulu": ["HNL"],
    "maui": ["OGG"],
    "kauai": ["LIH"],
    "kona": ["KOA"],
    "big island": ["KOA", "ITO"],
    
    # Japan
    "japan": ["NRT", "HND"],
    "tokyo": ["NRT", "HND"],
    "narita": ["NRT"],
    "haneda": ["HND"],
    "osaka": ["KIX"],
    
    # Korea
    "korea": ["ICN"],
    "seoul": ["ICN"],
    "incheon": ["ICN"],
    
    # China
    "china": ["PVG", "PEK", "CAN"],
    "shanghai": ["PVG"],
    "beijing": ["PEK"],
    "hong kong": ["HKG"],
    
    # Europe
    "london": ["LHR", "LGW"],
    "heathrow": ["LHR"],
    "gatwick": ["LGW"],
    "paris": ["CDG", "ORY"],
    "frankfurt": ["FRA"],
    "amsterdam": ["AMS"],
    "madrid": ["MAD"],
    "barcelona": ["BCN"],
    "rome": ["FCO"],
    "milan": ["MXP"],
    "munich": ["MUC"],
    "zurich": ["ZRH"],
    "dublin": ["DUB"],
    "lisbon": ["LIS"],
    
    # Middle East
    "dubai": ["DXB"],
    "doha": ["DOH"],
    "abu dhabi": ["AUH"],
    "tel aviv": ["TLV"],
    
    # Caribbean
    "san juan": ["SJU"],
    "puerto rico": ["SJU"],
    "st thomas": ["STT"],
    "virgin islands": ["STT", "STX"],
    "nassau": ["NAS"],
    "bahamas": ["NAS"],
    "jamaica": ["MBJ", "KIN"],
    "montego bay": ["MBJ"],
    "punta cana": ["PUJ"],
    "aruba": ["AUA"],
    "curacao": ["CUR"],
    "st maarten": ["SXM"],
    "turks": ["PLS"],
    "turks and caicos": ["PLS"],
    "cayman": ["GCM"],
    "grand cayman": ["GCM"],
    
    # Mexico
    "mexico city": ["MEX"],
    "cancun": ["CUN"],
    "cabo": ["SJD"],
    "los cabos": ["SJD"],
    "puerto vallarta": ["PVR"],
    
    # South America
    "sao paulo": ["GRU"],
    "rio": ["GIG"],
    "rio de janeiro": ["GIG"],
    "buenos aires": ["EZE"],
    "santiago": ["SCL"],
    "lima": ["LIM"],
    "bogota": ["BOG"],
    
    # US Cities
    "new york": ["JFK", "LGA", "EWR"],
    "nyc": ["JFK", "LGA", "EWR"],
    "la": ["LAX"],
    "los angeles": ["LAX"],
    "chicago": ["ORD", "MDW"],
    "san francisco": ["SFO"],
    "sf": ["SFO"],
    "boston": ["BOS"],
    "miami": ["MIA"],
    "vegas": ["LAS"],
    "las vegas": ["LAS"],
    "seattle": ["SEA"],
    "denver": ["DEN"],
    "phoenix": ["PHX"],
    "atlanta": ["ATL"],
    "dallas": ["DFW"],
    "houston": ["IAH"],
    "dc": ["DCA", "IAD"],
    "washington": ["DCA", "IAD"],
}

REGION_SYNONYMS: Dict[str, List[Region]] = {
    "hawaii": [Region.HAWAII],
    "hawaiian": [Region.HAWAII],
    "islands": [Region.HAWAII, Region.CARIBBEAN],
    "alaska": [Region.ALASKA],
    "europe": [Region.EUROPE],
    "european": [Region.EUROPE],
    "asia": [Region.ASIA],
    "asian": [Region.ASIA],
    "pacific": [Region.ASIA, Region.AUSTRALIA_NZ, Region.HAWAII],
    "caribbean": [Region.CARIBBEAN],
    "island": [Region.CARIBBEAN, Region.HAWAII],
    "mexico": [Region.MEXICO],
    "mexican": [Region.MEXICO],
    "canada": [Region.CANADA],
    "canadian": [Region.CANADA],
    "south america": [Region.SOUTH_AMERICA],
    "latin america": [Region.SOUTH_AMERICA, Region.CENTRAL_AMERICA, Region.MEXICO],
    "central america": [Region.CENTRAL_AMERICA],
    "middle east": [Region.MIDDLE_EAST],
    "africa": [Region.AFRICA],
    "australia": [Region.AUSTRALIA_NZ],
    "new zealand": [Region.AUSTRALIA_NZ],
    "domestic": [Region.NORTHEAST, Region.SOUTHEAST, Region.NORTH_CENTRAL,
                 Region.SOUTH_CENTRAL, Region.NORTHWEST, Region.SOUTHWEST],
    "international": [Region.EUROPE, Region.ASIA, Region.SOUTH_AMERICA,
                      Region.MIDDLE_EAST, Region.AFRICA, Region.AUSTRALIA_NZ],
}

PAIRING_LENGTH_SYNONYMS: Dict[str, List[int]] = {
    "turn": [1],
    "turns": [1],
    "day trip": [1],
    "day trips": [1],
    "one day": [1],
    "1 day": [1],
    "short": [1, 2],
    "short trip": [1, 2],
    "short trips": [1, 2],
    "two day": [2],
    "2 day": [2],
    "overnight": [2],
    "overnights": [2, 3],
    "three day": [3],
    "3 day": [3],
    "medium": [2, 3],
    "four day": [4],
    "4 day": [4],
    "long": [4, 5],
    "long trip": [4, 5],
    "long trips": [4, 5],
    "five day": [5],
    "5 day": [5],
    "extended": [4, 5, 6],
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_date(date_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validate date string is YYYY-MM-DD format and represents a real date.
    
    Returns:
        (is_valid, error_message or None)
    """
    if not DATE_PATTERN.match(date_str):
        return False, f"Invalid date format '{date_str}'. Expected YYYY-MM-DD"
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True, None
    except ValueError as e:
        return False, f"Invalid date '{date_str}': {str(e)}"


def validate_airport_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate airport code format and existence.
    
    Returns:
        (is_valid, error_message or None)
    """
    code_upper = code.upper().strip()
    if not AIRPORT_PATTERN.match(code_upper):
        return False, f"Invalid airport code format '{code}'. Expected 3 letters"
    if code_upper not in VALID_AIRPORT_CODES:
        # Warn but don't fail - might be a valid code we don't have
        return True, f"Warning: Airport code '{code_upper}' not in known list"
    return True, None


def validate_airport_codes(codes: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate list of airport codes.
    
    Returns:
        (valid_codes, invalid_codes, warnings)
    """
    valid = []
    invalid = []
    warnings = []
    
    for code in codes:
        is_valid, message = validate_airport_code(code)
        if is_valid:
            valid.append(code.upper().strip())
            if message:
                warnings.append(message)
        else:
            invalid.append(code)
    
    return valid, invalid, warnings


def validate_range(
    value: int | float,
    field_name: str,
    range_key: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate numeric value is within allowed range.
    
    Returns:
        (is_valid, error_message or None)
    """
    if range_key not in VALIDATION_RANGES:
        return True, None
    
    range_def = VALIDATION_RANGES[range_key]
    min_val = range_def.get("min")
    max_val = range_def.get("max")
    
    if min_val is not None and value < min_val:
        return False, f"{field_name} ({value}) is below minimum ({min_val})"
    if max_val is not None and value > max_val:
        return False, f"{field_name} ({value}) exceeds maximum ({max_val})"
    
    return True, None


def parse_time_to_minutes(time_str: str) -> Tuple[Optional[int], bool, Optional[str]]:
    """
    Parse time string to minutes from midnight.
    
    Handles formats:
    - "6pm", "6 pm", "6:00pm", "6:00 pm"
    - "0600", "06:00"
    - "6" (ambiguous)
    
    Returns:
        (minutes, is_ambiguous, error_message)
    """
    if time_str is None:
        return None, False, "No time provided"
    
    time_str = time_str.lower().strip()
    
    # Try 24-hour format first: "0600", "06:00", "1430"
    match_24 = re.match(r'^(\d{2}):?(\d{2})$', time_str)
    if match_24:
        hour = int(match_24.group(1))
        minute = int(match_24.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour * 60 + minute, False, None
    
    # Try 12-hour format: "6pm", "6:30 am", "12:00pm"
    match_12 = re.match(r'^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?$', time_str)
    if match_12:
        hour = int(match_12.group(1))
        minute = int(match_12.group(2) or 0)
        meridiem = match_12.group(3)
        
        if meridiem:
            meridiem = meridiem.replace(".", "").strip()
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            return hour * 60 + minute, False, None
        else:
            # No AM/PM - ambiguous
            # Use heuristic: in scheduling context, small numbers likely mean PM
            if hour >= 1 and hour <= 6:
                # "home by 6" likely means 6pm
                assumed_hour = hour + 12
                return assumed_hour * 60 + minute, True, f"Assumed {hour}pm (ambiguous)"
            elif hour >= 7 and hour <= 11:
                # Could be AM or PM - truly ambiguous
                return hour * 60 + minute, True, f"Ambiguous: could be {hour}am or {hour}pm"
            else:
                return hour * 60 + minute, False, None
    
    # Try word format: "noon", "midnight"
    if time_str in ["noon", "12 noon", "12noon"]:
        return 720, False, None
    if time_str in ["midnight", "12 midnight"]:
        return 0, False, None
    
    return None, False, f"Could not parse time: {time_str}"


def minutes_to_time_str(minutes: int) -> str:
    """Convert minutes from midnight to readable time string."""
    if minutes is None:
        return "N/A"
    hours = minutes // 60
    mins = minutes % 60
    period = "AM" if hours < 12 else "PM"
    display_hour = hours % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{mins:02d} {period}"


# ============================================================================
# PREFERENCE DATACLASSES
# ============================================================================

@dataclass
class DaysOffPreferences:
    """Preferences related to days off."""
    
    # Specific days of week
    days_off: List[str] = field(default_factory=list)  # Store as strings for JSON
    days_off_strength: str = "preferred"
    
    # Specific dates (YYYY-MM-DD format)
    must_off_dates: List[str] = field(default_factory=list)
    prefer_off_dates: List[str] = field(default_factory=list)
    
    # Optimization flags
    maximize_total_days_off: bool = False
    maximize_block_days_off: bool = False
    maximize_weekends_off: bool = False
    maximize_weekend_days_off: bool = False
    
    # Block preferences
    min_consecutive_days_off: Optional[int] = None
    prefer_days_off_start_day: Optional[str] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        
        # Validate days of week
        valid_days = {d.value for d in DayOfWeek}
        for day in self.days_off:
            if day not in valid_days:
                errors.append(f"Invalid day of week: {day}")
        
        # Validate strength
        valid_strengths = {s.value for s in PreferenceStrength}
        if self.days_off_strength not in valid_strengths:
            errors.append(f"Invalid preference strength: {self.days_off_strength}")
        
        # Validate dates
        for date in self.must_off_dates + self.prefer_off_dates:
            is_valid, error = validate_date(date)
            if not is_valid:
                errors.append(error)
        
        # Validate consecutive days off range
        if self.min_consecutive_days_off is not None:
            is_valid, error = validate_range(
                self.min_consecutive_days_off,
                "min_consecutive_days_off",
                "consecutive_days_off"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate start day
        if self.prefer_days_off_start_day is not None:
            if self.prefer_days_off_start_day not in valid_days:
                errors.append(f"Invalid start day: {self.prefer_days_off_start_day}")
        
        return len(errors) == 0, errors


@dataclass
class PairingPreferences:
    """Preferences related to trip/pairing characteristics."""
    
    # Length
    prefer_pairing_length: List[int] = field(default_factory=list)
    min_pairing_length: Optional[int] = None
    max_pairing_length: Optional[int] = None
    
    # Type
    prefer_pairing_type: List[str] = field(default_factory=list)
    avoid_pairing_type: List[str] = field(default_factory=list)
    
    # Duty periods
    max_duty_periods: Optional[int] = None
    max_legs_per_duty: Optional[int] = None
    
    # Efficiency
    max_tafb_credit_ratio: Optional[float] = None
    min_credit_per_duty_minutes: Optional[int] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        
        # Validate pairing lengths
        for length in self.prefer_pairing_length:
            is_valid, error = validate_range(length, "pairing_length", "pairing_length")
            if not is_valid:
                errors.append(error)
        
        if self.min_pairing_length is not None:
            is_valid, error = validate_range(
                self.min_pairing_length, "min_pairing_length", "pairing_length"
            )
            if not is_valid:
                errors.append(error)
        
        if self.max_pairing_length is not None:
            is_valid, error = validate_range(
                self.max_pairing_length, "max_pairing_length", "pairing_length"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate min <= max
        if (self.min_pairing_length is not None and
            self.max_pairing_length is not None and
            self.min_pairing_length > self.max_pairing_length):
            errors.append(
                f"min_pairing_length ({self.min_pairing_length}) > "
                f"max_pairing_length ({self.max_pairing_length})"
            )
        
        # Validate pairing types
        valid_types = {t.value for t in PairingType}
        for pt in self.prefer_pairing_type + self.avoid_pairing_type:
            if pt not in valid_types:
                errors.append(f"Invalid pairing type: {pt}")
        
        # Check for conflicts
        overlap = set(self.prefer_pairing_type) & set(self.avoid_pairing_type)
        if overlap:
            errors.append(f"Pairing types in both prefer and avoid: {overlap}")
        
        # Validate duty periods
        if self.max_duty_periods is not None:
            is_valid, error = validate_range(
                self.max_duty_periods, "max_duty_periods", "duty_periods"
            )
            if not is_valid:
                errors.append(error)
        
        if self.max_legs_per_duty is not None:
            is_valid, error = validate_range(
                self.max_legs_per_duty, "max_legs_per_duty", "legs_per_duty"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate TAFB ratio
        if self.max_tafb_credit_ratio is not None:
            is_valid, error = validate_range(
                self.max_tafb_credit_ratio, "max_tafb_credit_ratio", "tafb_credit_ratio"
            )
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors


@dataclass
class TimePreferences:
    """Preferences related to report/release times."""
    
    # Report time constraints (minutes from midnight)
    report_after_minutes: Optional[int] = None
    report_before_minutes: Optional[int] = None
    
    # Release time constraints
    release_after_minutes: Optional[int] = None
    release_before_minutes: Optional[int] = None
    
    # Mid-pairing constraints
    mid_pairing_report_after_minutes: Optional[int] = None
    mid_pairing_release_before_minutes: Optional[int] = None
    
    # Commuter info
    is_commuter: bool = False
    commuter_base: Optional[str] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        
        # Validate all time fields
        time_fields = [
            ("report_after_minutes", self.report_after_minutes),
            ("report_before_minutes", self.report_before_minutes),
            ("release_after_minutes", self.release_after_minutes),
            ("release_before_minutes", self.release_before_minutes),
            ("mid_pairing_report_after_minutes", self.mid_pairing_report_after_minutes),
            ("mid_pairing_release_before_minutes", self.mid_pairing_release_before_minutes),
        ]
        
        for field_name, value in time_fields:
            if value is not None:
                is_valid, error = validate_range(value, field_name, "time_minutes")
                if not is_valid:
                    errors.append(error)
        
        # Validate report after < report before
        if (self.report_after_minutes is not None and
            self.report_before_minutes is not None and
            self.report_after_minutes >= self.report_before_minutes):
            errors.append(
                f"report_after ({minutes_to_time_str(self.report_after_minutes)}) "
                f"must be before report_before ({minutes_to_time_str(self.report_before_minutes)})"
            )
        
        # Validate release after < release before
        if (self.release_after_minutes is not None and
            self.release_before_minutes is not None and
            self.release_after_minutes >= self.release_before_minutes):
            errors.append(
                f"release_after ({minutes_to_time_str(self.release_after_minutes)}) "
                f"must be before release_before ({minutes_to_time_str(self.release_before_minutes)})"
            )
        
        # Validate commuter base
        if self.is_commuter and self.commuter_base:
            is_valid, error = validate_airport_code(self.commuter_base)
            if not is_valid:
                errors.append(error)
        
        # Warn if commuter but no base
        if self.is_commuter and not self.commuter_base:
            errors.append("is_commuter is True but no commuter_base specified")
        
        return len(errors) == 0, errors


@dataclass
class LayoverPreferences:
    """Preferences related to layover cities and regions."""
    
    # City preferences (airport codes)
    prefer_layover_city: List[str] = field(default_factory=list)
    avoid_layover_city: List[str] = field(default_factory=list)
    
    # Region preferences
    prefer_region: List[str] = field(default_factory=list)
    avoid_region: List[str] = field(default_factory=list)
    
    # Layover duration
    min_layover_hours: Optional[float] = None
    max_layover_hours: Optional[float] = None
    
    # Landing preferences
    prefer_landing_city: List[str] = field(default_factory=list)
    avoid_landing_city: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        warnings = []
        
        # Validate airport codes
        all_cities = (
            self.prefer_layover_city +
            self.avoid_layover_city +
            self.prefer_landing_city +
            self.avoid_landing_city
        )
        valid, invalid, code_warnings = validate_airport_codes(all_cities)
        if invalid:
            errors.append(f"Invalid airport codes: {invalid}")
        warnings.extend(code_warnings)
        
        # Check for city conflicts
        layover_overlap = set(self.prefer_layover_city) & set(self.avoid_layover_city)
        if layover_overlap:
            errors.append(f"Cities in both prefer and avoid layover: {layover_overlap}")
        
        landing_overlap = set(self.prefer_landing_city) & set(self.avoid_landing_city)
        if landing_overlap:
            errors.append(f"Cities in both prefer and avoid landing: {landing_overlap}")
        
        # Validate regions
        valid_regions = {r.value for r in Region}
        for region in self.prefer_region + self.avoid_region:
            if region not in valid_regions:
                errors.append(f"Invalid region: {region}")
        
        # Check for region conflicts
        region_overlap = set(self.prefer_region) & set(self.avoid_region)
        if region_overlap:
            errors.append(f"Regions in both prefer and avoid: {region_overlap}")
        
        # Validate layover hours
        if self.min_layover_hours is not None:
            is_valid, error = validate_range(
                self.min_layover_hours, "min_layover_hours", "layover_hours"
            )
            if not is_valid:
                errors.append(error)
        
        if self.max_layover_hours is not None:
            is_valid, error = validate_range(
                self.max_layover_hours, "max_layover_hours", "layover_hours"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate min <= max
        if (self.min_layover_hours is not None and
            self.max_layover_hours is not None and
            self.min_layover_hours > self.max_layover_hours):
            errors.append(
                f"min_layover_hours ({self.min_layover_hours}) > "
                f"max_layover_hours ({self.max_layover_hours})"
            )
        
        return len(errors) == 0, errors


@dataclass
class CreditPreferences:
    """Preferences related to pay/credit hours."""
    
    # Credit targets (in minutes)
    target_credit_min_minutes: Optional[int] = None
    target_credit_max_minutes: Optional[int] = None
    
    # Optimization flags
    maximize_credit: bool = False
    minimize_credit: bool = False
    minimize_tafb: bool = False
    prefer_efficient_flying: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        
        # Check for maximize/minimize conflict
        if self.maximize_credit and self.minimize_credit:
            errors.append("Cannot both maximize and minimize credit")
        
        # Validate credit ranges
        if self.target_credit_min_minutes is not None:
            is_valid, error = validate_range(
                self.target_credit_min_minutes,
                "target_credit_min_minutes",
                "credit_minutes"
            )
            if not is_valid:
                errors.append(error)
        
        if self.target_credit_max_minutes is not None:
            is_valid, error = validate_range(
                self.target_credit_max_minutes,
                "target_credit_max_minutes",
                "credit_minutes"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate min <= max
        if (self.target_credit_min_minutes is not None and
            self.target_credit_max_minutes is not None and
            self.target_credit_min_minutes > self.target_credit_max_minutes):
            errors.append(
                f"target_credit_min_minutes ({self.target_credit_min_minutes}) > "
                f"target_credit_max_minutes ({self.target_credit_max_minutes})"
            )
        
        # Warn about conflicting optimizations
        if self.maximize_credit and self.minimize_tafb:
            errors.append(
                "Warning: maximize_credit and minimize_tafb are often conflicting goals"
            )
        
        return len(errors) == 0, errors


@dataclass
class WorkBlockPreferences:
    """Preferences related to work block structure."""
    
    # Block length
    min_work_block_days: Optional[int] = None
    max_work_block_days: Optional[int] = None
    prefer_work_block_days: List[int] = field(default_factory=list)
    
    # Days off between blocks
    min_days_off_between_blocks: Optional[int] = None
    
    # Start day
    prefer_work_block_start_day: Optional[str] = None
    
    # Pairing mix
    prefer_pairing_mix: Optional[str] = None
    
    # Waivers
    allow_reduced_post_block_rest: bool = False
    allow_co_terminal_mix: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        
        # Validate work block days
        if self.min_work_block_days is not None:
            is_valid, error = validate_range(
                self.min_work_block_days, "min_work_block_days", "work_block_days"
            )
            if not is_valid:
                errors.append(error)
        
        if self.max_work_block_days is not None:
            is_valid, error = validate_range(
                self.max_work_block_days, "max_work_block_days", "work_block_days"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate min <= max
        if (self.min_work_block_days is not None and
            self.max_work_block_days is not None and
            self.min_work_block_days > self.max_work_block_days):
            errors.append(
                f"min_work_block_days ({self.min_work_block_days}) > "
                f"max_work_block_days ({self.max_work_block_days})"
            )
        
        for days in self.prefer_work_block_days:
            is_valid, error = validate_range(days, "prefer_work_block_days", "work_block_days")
            if not is_valid:
                errors.append(error)
        
        # Validate start day
        if self.prefer_work_block_start_day is not None:
            valid_days = {d.value for d in DayOfWeek}
            if self.prefer_work_block_start_day not in valid_days:
                errors.append(f"Invalid start day: {self.prefer_work_block_start_day}")
        
        return len(errors) == 0, errors


@dataclass
class ReservePreferences:
    """Preferences related to reserve duty."""
    
    # Reserve type preference
    reserve_type_preference: str = "Either"
    avoid_reserve: bool = False
    
    # Reserve block structure
    min_reserve_block_days: Optional[int] = None
    max_reserve_block_days: Optional[int] = None
    prefer_fly_through_days: Optional[int] = None
    
    # Reserve days off
    reserve_must_off_dates: List[str] = field(default_factory=list)
    reserve_prefer_off_dates: List[str] = field(default_factory=list)
    
    # Waivers
    waive_reserve_block_of_4_days_off: bool = False
    allow_single_reserve_day_off: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        
        # Validate reserve type
        valid_types = {t.value for t in ReserveType}
        if self.reserve_type_preference not in valid_types:
            errors.append(f"Invalid reserve type: {self.reserve_type_preference}")
        
        # Validate reserve block days
        if self.min_reserve_block_days is not None:
            is_valid, error = validate_range(
                self.min_reserve_block_days,
                "min_reserve_block_days",
                "reserve_block_days"
            )
            if not is_valid:
                errors.append(error)
        
        if self.max_reserve_block_days is not None:
            is_valid, error = validate_range(
                self.max_reserve_block_days,
                "max_reserve_block_days",
                "reserve_block_days"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate fly through days
        if self.prefer_fly_through_days is not None:
            is_valid, error = validate_range(
                self.prefer_fly_through_days,
                "prefer_fly_through_days",
                "fly_through_days"
            )
            if not is_valid:
                errors.append(error)
        
        # Validate dates
        for date in self.reserve_must_off_dates + self.reserve_prefer_off_dates:
            is_valid, error = validate_date(date)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors


@dataclass
class MiscPreferences:
    """Miscellaneous preferences."""
    
    # Deadheads
    prefer_deadheads: bool = False
    avoid_deadheads: bool = False
    prefer_deadhead_first_leg: bool = False
    prefer_deadhead_last_leg: bool = False
    
    # Buddy bidding (FO only, up to 3 CAs)
    buddy_bid_employee_numbers: List[str] = field(default_factory=list)
    
    # One-leg duty periods
    prefer_one_leg_first_duty: bool = False
    prefer_one_leg_last_duty: bool = False
    
    # Rest preferences
    prefer_extended_domicile_rest: bool = False
    allow_reduced_domicile_rest: bool = False
    
    # Waivers
    allow_double_up: bool = False
    allow_same_day_pairing: bool = False
    allow_rest_at_outstation: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this preference category."""
        errors = []
        
        # Check deadhead conflict
        if self.prefer_deadheads and self.avoid_deadheads:
            errors.append("Cannot both prefer and avoid deadheads")
        
        # Validate buddy bid count
        if len(self.buddy_bid_employee_numbers) > VALIDATION_RANGES["buddy_bid_count"]["max"]:
            errors.append(
                f"Maximum {VALIDATION_RANGES['buddy_bid_count']['max']} buddy bids allowed, "
                f"got {len(self.buddy_bid_employee_numbers)}"
            )
        
        return len(errors) == 0, errors


# ============================================================================
# MASTER PREFERENCE OBJECT
# ============================================================================

@dataclass
class PilotPreferences:
    """Complete pilot preference profile."""
    
    # Metadata
    pilot_id: Optional[str] = None
    bid_month: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    # All preference categories
    days_off: DaysOffPreferences = field(default_factory=DaysOffPreferences)
    pairing: PairingPreferences = field(default_factory=PairingPreferences)
    times: TimePreferences = field(default_factory=TimePreferences)
    layovers: LayoverPreferences = field(default_factory=LayoverPreferences)
    credit: CreditPreferences = field(default_factory=CreditPreferences)
    work_blocks: WorkBlockPreferences = field(default_factory=WorkBlockPreferences)
    reserve: ReservePreferences = field(default_factory=ReservePreferences)
    misc: MiscPreferences = field(default_factory=MiscPreferences)
    
    # Priority weights (for scoring engine)
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "days_off": 1.0,
        "times": 0.8,
        "layovers": 0.6,
        "pairing": 0.5,
        "credit": 0.4,
        "work_blocks": 0.3,
        "reserve": 0.2,
        "misc": 0.1,
    })
    
    # Parser metadata
    original_input: Optional[str] = None
    parse_confidence: Optional[str] = None
    parse_warnings: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate all preferences.
        
        Returns:
            (is_valid, list_of_errors)
        """
        all_errors = []
        
        # Validate each category
        categories = [
            ("days_off", self.days_off),
            ("pairing", self.pairing),
            ("times", self.times),
            ("layovers", self.layovers),
            ("credit", self.credit),
            ("work_blocks", self.work_blocks),
            ("reserve", self.reserve),
            ("misc", self.misc),
        ]
        
        for name, category in categories:
            is_valid, errors = category.validate()
            for error in errors:
                all_errors.append(f"[{name}] {error}")
        
        # Cross-category validations
        
        # 1. Commuter time constraints
        if self.times.is_commuter:
            if self.times.report_after_minutes is None:
                all_errors.append(
                    "[cross-validation] Commuter but no report_after_minutes set. "
                    "Commuters typically need late report times."
                )
            if self.times.release_before_minutes is None:
                all_errors.append(
                    "[cross-validation] Commuter but no release_before_minutes set. "
                    "Commuters typically need early release times."
                )
        
        # 2. Turn preference vs layover preference conflict
        if (PairingType.TURN.value in self.pairing.prefer_pairing_type and
            (self.layovers.prefer_layover_city or self.layovers.prefer_region)):
            all_errors.append(
                "[cross-validation] Prefer turns (day trips) conflicts with "
                "layover city/region preferences. Turns have no layovers."
            )
        
        # 3. Max pairing length 1 vs layover preferences
        if (self.pairing.max_pairing_length == 1 and
            (self.layovers.prefer_layover_city or self.layovers.prefer_region)):
            all_errors.append(
                "[cross-validation] max_pairing_length=1 (day trips only) conflicts with "
                "layover preferences. 1-day trips have no layovers."
            )
        
        # 4. Transoceanic preference vs short trip preference
        if (PairingType.TRANSOCEANIC.value in self.pairing.prefer_pairing_type and
            self.pairing.max_pairing_length is not None and
            self.pairing.max_pairing_length < 2):
            all_errors.append(
                "[cross-validation] Prefer transoceanic trips conflicts with "
                f"max_pairing_length={self.pairing.max_pairing_length}. "
                "Transoceanic trips require minimum 2 days."
            )
        
        # 5. Validate priority weights
        expected_categories = {"days_off", "times", "layovers", "pairing",
                             "credit", "work_blocks", "reserve", "misc"}
        actual_categories = set(self.priority_weights.keys())
        if actual_categories != expected_categories:
            missing = expected_categories - actual_categories
            extra = actual_categories - expected_categories
            if missing:
                all_errors.append(f"[priority_weights] Missing categories: {missing}")
            if extra:
                all_errors.append(f"[priority_weights] Unknown categories: {extra}")
        
        # Validate weight values
        for category, weight in self.priority_weights.items():
            if not isinstance(weight, (int, float)):
                all_errors.append(
                    f"[priority_weights] Invalid weight type for {category}: {type(weight)}"
                )
            elif weight < 0 or weight > 1:
                all_errors.append(
                    f"[priority_weights] Weight for {category} ({weight}) "
                    "should be between 0 and 1"
                )
        
        return len(all_errors) == 0, all_errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, handling nested dataclasses.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        def convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            return obj
        
        return convert(self)
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PilotPreferences':
        """
        Create PilotPreferences from dictionary.
        
        Args:
            data: Dictionary with preference data
            
        Returns:
            PilotPreferences instance
        """
        # Create nested dataclass instances
        days_off_data = data.get("days_off", {})
        pairing_data = data.get("pairing", {})
        times_data = data.get("times", {})
        layovers_data = data.get("layovers", {})
        credit_data = data.get("credit", {})
        work_blocks_data = data.get("work_blocks", {})
        reserve_data = data.get("reserve", {})
        misc_data = data.get("misc", {})
        
        return cls(
            pilot_id=data.get("pilot_id"),
            bid_month=data.get("bid_month"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            days_off=DaysOffPreferences(**days_off_data) if days_off_data else DaysOffPreferences(),
            pairing=PairingPreferences(**pairing_data) if pairing_data else PairingPreferences(),
            times=TimePreferences(**times_data) if times_data else TimePreferences(),
            layovers=LayoverPreferences(**layovers_data) if layovers_data else LayoverPreferences(),
            credit=CreditPreferences(**credit_data) if credit_data else CreditPreferences(),
            work_blocks=WorkBlockPreferences(**work_blocks_data) if work_blocks_data else WorkBlockPreferences(),
            reserve=ReservePreferences(**reserve_data) if reserve_data else ReservePreferences(),
            misc=MiscPreferences(**misc_data) if misc_data else MiscPreferences(),
            priority_weights=data.get("priority_weights", {
                "days_off": 1.0,
                "times": 0.8,
                "layovers": 0.6,
                "pairing": 0.5,
                "credit": 0.4,
                "work_blocks": 0.3,
                "reserve": 0.2,
                "misc": 0.1,
            }),
            original_input=data.get("original_input"),
            parse_confidence=data.get("parse_confidence"),
            parse_warnings=data.get("parse_warnings", []),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PilotPreferences':
        """
        Create PilotPreferences from JSON string.
        
        Args:
            json_str: JSON string with preference data
            
        Returns:
            PilotPreferences instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_active_preferences(self) -> Dict[str, Any]:
        """
        Get only the preferences that have been set (non-default values).
        
        Returns:
            Dictionary of active preferences
        """
        active = {}
        full_dict = self.to_dict()
        
        # Check each category
        for category_name in ["days_off", "pairing", "times", "layovers",
                             "credit", "work_blocks", "reserve", "misc"]:
            category_data = full_dict.get(category_name, {})
            active_in_category = {}
            
            for key, value in category_data.items():
                # Skip None, empty lists, False booleans (defaults)
                if value is None:
                    continue
                if isinstance(value, list) and len(value) == 0:
                    continue
                if isinstance(value, bool) and value is False:
                    continue
                if isinstance(value, str) and value in ["preferred", "Either"]:
                    continue
                active_in_category[key] = value
            
            if active_in_category:
                active[category_name] = active_in_category
        
        return active
    
    def summary(self) -> str:
        """
        Generate human-readable summary of active preferences.
        
        Returns:
            Summary string
        """
        active = self.get_active_preferences()
        if not active:
            return "No preferences set."
        
        lines = []
        
        # Days off
        if "days_off" in active:
            d = active["days_off"]
            if d.get("days_off"):
                lines.append(f"Days off: {', '.join(d['days_off'])}")
            if d.get("must_off_dates"):
                lines.append(f"Must have off: {', '.join(d['must_off_dates'])}")
            if d.get("maximize_weekends_off"):
                lines.append("Maximize weekends off")
            if d.get("maximize_block_days_off"):
                lines.append("Maximize consecutive days off")
        
        # Pairing
        if "pairing" in active:
            p = active["pairing"]
            if p.get("prefer_pairing_length"):
                lines.append(f"Prefer {p['prefer_pairing_length']}-day trips")
            if p.get("max_pairing_length"):
                lines.append(f"Max trip length: {p['max_pairing_length']} days")
            if p.get("avoid_pairing_type"):
                lines.append(f"Avoid: {', '.join(p['avoid_pairing_type'])}")
            if p.get("prefer_pairing_type"):
                lines.append(f"Prefer: {', '.join(p['prefer_pairing_type'])}")
        
        # Times
        if "times" in active:
            t = active["times"]
            if t.get("is_commuter"):
                base = t.get("commuter_base", "unknown")
                lines.append(f"Commuter from {base}")
            if t.get("report_after_minutes"):
                lines.append(f"Report after: {minutes_to_time_str(t['report_after_minutes'])}")
            if t.get("release_before_minutes"):
                lines.append(f"Release before: {minutes_to_time_str(t['release_before_minutes'])}")
        
        # Layovers
        if "layovers" in active:
            l = active["layovers"]
            if l.get("prefer_layover_city"):
                lines.append(f"Prefer layovers: {', '.join(l['prefer_layover_city'])}")
            if l.get("avoid_layover_city"):
                lines.append(f"Avoid layovers: {', '.join(l['avoid_layover_city'])}")
            if l.get("prefer_region"):
                lines.append(f"Prefer regions: {', '.join(l['prefer_region'])}")
        
        # Credit
        if "credit" in active:
            c = active["credit"]
            if c.get("maximize_credit"):
                lines.append("Maximize credit/pay")
            if c.get("minimize_credit"):
                lines.append("Minimize credit (senior pilot)")
            if c.get("minimize_tafb"):
                lines.append("Minimize time away from base")
        
        return "\n".join(lines) if lines else "No significant preferences set."


# ============================================================================
# TEST SUITE
# ============================================================================

def run_tests():
    """Run validation tests on the schema."""
    print("=" * 60)
    print("PREFERENCE SCHEMA TEST SUITE")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Valid preferences
    print("\nTest 1: Valid basic preferences")
    prefs = PilotPreferences(
        days_off=DaysOffPreferences(
            days_off=["Friday", "Saturday"],
            must_off_dates=["2026-02-14"],
        ),
        layovers=LayoverPreferences(
            prefer_layover_city=["HNL", "OGG"],
        ),
        pairing=PairingPreferences(
            avoid_pairing_type=["Redeye", "Trailing Redeye"],
            max_pairing_length=3,
        ),
        times=TimePreferences(
            release_before_minutes=1080,  # 6pm
        ),
    )
    is_valid, errors = prefs.validate()
    if is_valid:
        print("  ✓ PASSED")
        tests_passed += 1
    else:
        print(f"  ✗ FAILED: {errors}")
        tests_failed += 1
    
    # Test 2: Invalid day of week
    print("\nTest 2: Invalid day of week")
    prefs = PilotPreferences(
        days_off=DaysOffPreferences(
            days_off=["Fri"],  # Should be "Friday"
        ),
    )
    is_valid, errors = prefs.validate()
    if not is_valid and any("Invalid day" in e for e in errors):
        print("  ✓ PASSED (correctly detected invalid day)")
        tests_passed += 1
    else:
        print(f"  ✗ FAILED: Should have caught invalid day")
        tests_failed += 1
    
    # Test 3: Conflicting credit preferences
    print("\nTest 3: Conflicting credit preferences")
    prefs = PilotPreferences(
        credit=CreditPreferences(
            maximize_credit=True,
            minimize_credit=True,
        ),
    )
    is_valid, errors = prefs.validate()
    if not is_valid and any("maximize and minimize" in e for e in errors):
        print("  ✓ PASSED (correctly detected conflict)")
        tests_passed += 1
    else:
        print(f"  ✗ FAILED: Should have caught credit conflict")
        tests_failed += 1
    
    # Test 4: Conflicting layover preferences
    print("\nTest 4: Same city in prefer and avoid")
    prefs = PilotPreferences(
        layovers=LayoverPreferences(
            prefer_layover_city=["HNL", "NRT"],
            avoid_layover_city=["HNL"],  # Conflict!
        ),
    )
    is_valid, errors = prefs.validate()
    if not is_valid and any("prefer and avoid" in e for e in errors):
        print("  ✓ PASSED (correctly detected conflict)")
        tests_passed += 1
    else:
        print(f"  ✗ FAILED: Should have caught layover conflict")
        tests_failed += 1
    
    # Test 5: Cross-validation - turns with layover preference
    print("\nTest 5: Turns (day trips) with layover preference")
    prefs = PilotPreferences(
        pairing=PairingPreferences(
            prefer_pairing_type=["Turn"],
        ),
        layovers=LayoverPreferences(
            prefer_layover_city=["HNL"],
        ),
    )
    is_valid, errors = prefs.validate()
    if not is_valid and any("Turns have no layovers" in e for e in errors):
        print("  ✓ PASSED (correctly detected cross-validation conflict)")
        tests_passed += 1
    else:
        print(f"  ✗ FAILED: Should have caught turn/layover conflict")
        tests_failed += 1
    
    # Test 6: Invalid date format
    print("\nTest 6: Invalid date format")
    prefs = PilotPreferences(
        days_off=DaysOffPreferences(
            must_off_dates=["Feb 14, 2026"],  # Wrong format
        ),
    )
    is_valid, errors = prefs.validate()
    if not is_valid and any("Invalid date format" in e for e in errors):
        print("  ✓ PASSED (correctly detected invalid date)")
        tests_passed += 1
    else:
        print(f"  ✗ FAILED: Should have caught invalid date format")
        tests_failed += 1
    
    # Test 7: Time parsing
    print("\nTest 7: Time parsing")
    test_cases = [
        ("6pm", 1080, False),
        ("6:30pm", 1110, False),
        ("0600", 360, False),
        ("06:00", 360, False),
        ("noon", 720, False),
        ("6", None, True),  # Ambiguous
    ]
    all_passed = True
    for time_str, expected_minutes, expected_ambiguous in test_cases:
        minutes, ambiguous, error = parse_time_to_minutes(time_str)
        if expected_minutes is not None and minutes != expected_minutes:
            print(f"  ✗ '{time_str}' parsed to {minutes}, expected {expected_minutes}")
            all_passed = False
        if ambiguous != expected_ambiguous:
            print(f"  ✗ '{time_str}' ambiguity={ambiguous}, expected {expected_ambiguous}")
            all_passed = False
    if all_passed:
        print("  ✓ PASSED (all time formats parsed correctly)")
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 8: Serialization round-trip
    print("\nTest 8: JSON serialization round-trip")
    prefs = PilotPreferences(
        pilot_id="TEST123",
        days_off=DaysOffPreferences(
            days_off=["Friday"],
            maximize_weekends_off=True,
        ),
        layovers=LayoverPreferences(
            prefer_region=["Hawaii"],
        ),
    )
    json_str = prefs.to_json()
    restored = PilotPreferences.from_json(json_str)
    if (restored.pilot_id == "TEST123" and
        restored.days_off.days_off == ["Friday"] and
        restored.days_off.maximize_weekends_off is True and
        "Hawaii" in restored.layovers.prefer_region):
        print("  ✓ PASSED")
        tests_passed += 1
    else:
        print("  ✗ FAILED: Round-trip lost data")
        tests_failed += 1
    
    # Test 9: Summary generation
    print("\nTest 9: Summary generation")
    prefs = PilotPreferences(
        days_off=DaysOffPreferences(
            days_off=["Friday", "Saturday"],
        ),
        pairing=PairingPreferences(
            avoid_pairing_type=["Redeye"],
        ),
        times=TimePreferences(
            is_commuter=True,
            commuter_base="DEN",
        ),
    )
    summary = prefs.summary()
    if ("Friday" in summary and
        "Redeye" in summary and
        "DEN" in summary):
        print("  ✓ PASSED")
        print(f"  Summary:\n{summary}")
        tests_passed += 1
    else:
        print("  ✗ FAILED: Summary missing expected content")
        tests_failed += 1
    
    # Test 10: Commuter without times warning
    print("\nTest 10: Commuter without time constraints")
    prefs = PilotPreferences(
        times=TimePreferences(
            is_commuter=True,
            commuter_base="DEN",
            # No report_after or release_before set
        ),
    )
    is_valid, errors = prefs.validate()
    if any("Commuter but no report_after" in e for e in errors):
        print("  ✓ PASSED (correctly warned about commuter times)")
        tests_passed += 1
    else:
        print(f"  ✗ FAILED: Should have warned about commuter times")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)