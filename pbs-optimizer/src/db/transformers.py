"""
Data Transformation Utilities
Converts parser output to database-ready format
"""

import hashlib
from pathlib import Path
from typing import List, Optional


def time_string_to_minutes(time_str: str) -> int: # function declaration
    """
    Convert 'H.MM' or 'HH.MM' or 'HHH.MM' format to total minutes.
    
    Examples:
        '5.45' -> 345 (5 hours, 45 minutes)
        '28.35' -> 1715 (28 hours, 35 minutes)
        '104.35' -> 6275 (104 hours, 35 minutes)
    """
    if not time_str or time_str == "0.00": # constraints 
        return 0
    
    try:
        parts = time_str.split(".") # split string by "."
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        return hours * 60 + minutes
    except (ValueError, IndexError):
        return 0


def clock_time_to_minutes(clock_str: str) -> int:
    """
    Convert 'HHMM' format to minutes from midnight.
    
    Examples:
        '0800' -> 480 (8 * 60)
        '1905' -> 1145 (19 * 60 + 5)
        '0030' -> 30 (0 * 60 + 30)
    """
    if not clock_str or len(clock_str) != 4:
        return 0
    
    try:
        hours = int(clock_str[:2]) # slices for the first two characters (hours)
        minutes = int(clock_str[2:]) # slices for the last two characters (minutes)
        return hours * 60 + minutes
    except ValueError:
        return 0


def extract_layover_cities(duty_periods: List[dict]) -> List[str]:
    """
    Extract layover city codes from duty periods.
    
    Returns list of city codes, e.g., ['OGG'], ['ICN'], ['NRT']
    """
    cities = [] # initialize empty list 
    for dp in duty_periods: # loop through each duty period (dictionary)
        layover = dp.get("layover") # retieve layover info (key)
        if layover and layover.get("city_code"):
            cities.append(layover["city_code"])
    return cities


def map_division(division: str) -> str:
    """
    Map parser division format to database format.
    
    'INTL' -> 'INT'
    'DOM' -> 'DOM'
    """
    mapping = {
        "INTL": "INT",
        "INT": "INT",
        "DOM": "DOM",
        "DOMESTIC": "DOM"
    }
    return mapping.get(division.upper(), "DOM")


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    """
    sha256_hash = hashlib.sha256() # creates the sha256 hash object, initialization
    with open(file_path, "rb") as f: # opens the file in binary read mode 
        for byte_block in iter(lambda: f.read(4096), b""): # reads in 4KB chunks and loops until EOF
            sha256_hash.update(byte_block) # feed each chunk into the hash object 
    return sha256_hash.hexdigest() # return the hexadecimal representation of the hash


def derive_pairing_length(duty_periods: List[dict]) -> int:
    """
    Calculate pairing length in days.
    
    Count of duty periods = number of calendar days for the trip.
    """
    return len(duty_periods) if duty_periods else 1


def is_transoceanic_trip(layover_cities: List[str]) -> bool:
    """
    Determine if trip is transoceanic based on layover cities.
    """
    TRANSOCEANIC_CITIES = {
        # Asia
        "NRT", "HND", "ICN", "HKG", "PVG", "PEK", "SIN", "BKK", "DEL", "BOM",
        # Europe  
        "LHR", "CDG", "FRA", "AMS", "MAD", "FCO", "MUC", "ZRH", "DUB",
        # Australia/Pacific
        "SYD", "MEL", "AKL",
        # South America (deep south)
        "GRU", "EZE", "SCL", "BOG", "LIM",
        # Africa
        "JNB", "CPT",
        # Middle East
        "DXB", "DOH", "AUH"
    }
    return any(city in TRANSOCEANIC_CITIES for city in layover_cities)


def is_redeye_trip(sequence: dict) -> bool:
    """
    Determine if trip contains a redeye segment.
    
    Redeye: departs late night, arrives early morning.
    """
    for dp in sequence.get("duty_periods", []):
        release = dp.get("release_time", {}).get("base", "")
        if release:
            release_mins = clock_time_to_minutes(release)
            # Release between midnight and 8am suggests redeye
            if 0 <= release_mins <= 480:
                return True
    return False


# Test if run directly
if __name__ == "__main__":
    # Test cases
    print("Testing transformers...")
    
    assert time_string_to_minutes("5.45") == 345
    assert time_string_to_minutes("28.35") == 1715
    assert time_string_to_minutes("0.00") == 0
    print("✓ time_string_to_minutes")
    
    assert clock_time_to_minutes("0800") == 480
    assert clock_time_to_minutes("1905") == 1145
    assert clock_time_to_minutes("0030") == 30
    print("✓ clock_time_to_minutes")
    
    assert map_division("INTL") == "INT"
    assert map_division("DOM") == "DOM"
    print("✓ map_division")
    
    assert is_transoceanic_trip(["NRT"]) == True
    assert is_transoceanic_trip(["LAS"]) == False
    print("✓ is_transoceanic_trip")
    
    print("\nAll transformer tests passed!")