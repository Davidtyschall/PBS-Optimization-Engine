# src/llm/client.py
# LLM client for parsing bid packets

import os # reads environment variable to authenticate with OpenAI API
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def create_client():
    """Create and return an OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def parse_full_page(text: str) -> dict:
    """
    Use LLM to parse entire bid packet page into structured JSON.
    
    Args:
        text: Raw OCR text from bid packet
        
    Returns:
        Complete parsed page as dictionary matching BidPacketPage schema
    """
    client = create_client()
    
    prompt = f"""Parse this airline bid packet page into structured JSON.

    CRITICAL RULES:
    1. Extract ALL sequences (look for "SEQ XXX" patterns)
    2. Extract ALL flight legs within each sequence
    3. Group legs into duty_periods (duty period 1, duty period 2, etc.)
    4. Include layover information (hotel names, transport)
    5. Include calendar_start_dates from the right side of each sequence

    DUTY PERIOD SEPARATION (VERY IMPORTANT):
    - A LAYOVER marks the END of one duty period and the START of the next
    - If a sequence has a layover (hotel info), it has AT LEAST 2 duty periods
    - Duty Period 1: flights BEFORE the layover, ends with RLS time before hotel
    - Duty Period 2: flights AFTER the layover, starts with RPT time after hotel
    - Each duty period has its OWN release_time and duty_totals
    - Do NOT merge all flights into a single duty period

    MEAL CODES:
    - Look for single letters B, L, D, or M before the arrival station
    - B = Breakfast, L = Lunch, D = Dinner, M = Meal
    - These appear in the "ML" column between departure and arrival

    GROUND TIME BETWEEN FLIGHTS (VERY IMPORTANT):
    - On a flight line, AFTER the block time, look for an additional time value like "3.50X" or "2.56X"
    - This indicates ground time between flights and MUST be a separate leg entry
    - Example line: "1  1/1 83  320  DFW 0900/0900  B LAS 1010/1210   3.10          3.50X"
    - This contains TWO items to extract:
        1. FLIGHT: DFW to LAS, block_hhmm "3.10"
        2. GROUND_BETWEEN_LEGS: ground_hhmm "3.50", equipment_change true (because of "X")
    - The "X" suffix means equipment_change: true (pilot changes aircraft)
    - Without "X" suffix, equipment_change: false
    - Create the GROUND_BETWEEN_LEGS leg AFTER the flight leg in the legs array

    DEADHEAD LEGS:
    - If the block_time column shows "AA" or another 2-letter carrier code instead of a time like "3.10", this is a DEADHEAD
    - Set leg_type to "DEADHEAD"
    - The carrier code (e.g., "AA") goes in deadhead_carrier_code
    - The credit time goes in deadhead_credit_hhmm
    - Flight numbers ending in "D" or "p" often indicate deadheads

    CALENDAR START DATES:
    - Look at the calendar grid on the right side (MO TU WE TH FR SA SU)
    - Find the day numbers (1-31) that are NOT dashes
    - These are the dates this sequence starts
    - Must have EXACTLY ops_count number of dates

    Return JSON matching this EXACT structure:

    {{
    "page_metadata": {{
        "page_number": <int from PAGE XXX at bottom>,
        "issued_date": "<YYYY-MM-DD from ISSUED line>",
        "effective_date": "<YYYY-MM-DD from EFF line>",
        "bid_status": {{
        "base": "<3-letter code like DFW>",
        "equipment_family": "<like 777>",
        "division": "<like INTL>"
        }},
        "bidding_period": {{
        "start_md": "<MM/DD>",
        "end_md": "<MM/DD>",
        "length_days": <int>
        }}
    }},
    "sequences": [
        {{
        "seq_num": <int>,
        "ops_count": <int from "X OPS">,
        "positions": ["CA", "FO"],
        "sequence_label": "<like KOREAN OPERATION or null>",
        "sequence_report_time": {{"local": "HHMM", "base": "HHMM"}},
        "calendar_start_dates": [<list of day numbers from calendar grid>],
        "duty_periods": [
            {{
            "duty_index": <1, 2, 3, etc.>,
            "date_span_md": {{"start": "M/D", "end": "M/D"}},
            "legs": [
                {{
                "leg_type": "FLIGHT",
                "equipment_code": <int like 83>,
                "flight_number": "<string>",
                "depart": {{"station": "XXX", "time_local": "HHMM", "time_base": "HHMM"}},
                "meal": "<B, L, D, M, or null>",
                "arrive": {{"station": "XXX", "time_local": "HHMM", "time_base": "HHMM"}},
                "block_hhmm": "<H.MM>"
                }},
                {{
                "leg_type": "GROUND_BETWEEN_LEGS",
                "ground_hhmm": "<H.MM>",
                "equipment_change": <true or false>
                }},
                {{
                "leg_type": "FLIGHT",
                "equipment_code": <int>,
                "flight_number": "<string>",
                "depart": {{"station": "XXX", "time_local": "HHMM", "time_base": "HHMM"}},
                "meal": "<B, L, D, M, or null>",
                "arrive": {{"station": "XXX", "time_local": "HHMM", "time_base": "HHMM"}},
                "block_hhmm": "<H.MM>"
                }}
            ],
            "release_time": {{"local": "HHMM", "base": "HHMM"}},
            "duty_totals": {{
                "block_hhmm": "<H.MM>",
                "synthetic_hhmm": "<H.MM>",
                "tpay_hhmm": "<H.MM>",
                "duty_hhmm": "<H.MM>",
                "fdp_hhmm": "<H.MM>"
            }},
            "layover": {{
                "city_code": "XXX",
                "hotel_name": "<string>",
                "transport_provider": "<string>",
                "odl_hhmm": "<H.MM>"
            }} or null
            }}
        ],
        "ttl": {{
            "total_block_hhmm": "<H.MM>",
            "total_synthetic_hhmm": "<H.MM>",
            "total_tpay_hhmm": "<H.MM>",
            "tafb_hhmm": "<H.MM>"
        }}
        }}
    ]
    }}

    IMPORTANT:
    - ALL time fields (block_hhmm, duty_hhmm, fdp_hhmm, ground_hhmm, etc.) must be STRINGS like "5.45", never numbers
    - calendar_start_dates must contain EXACTLY ops_count number of dates
    - Sequences with layovers MUST have multiple duty_periods
    - When you see "X.XXX" after block time on a flight line, create a GROUND_BETWEEN_LEGS leg

    Return ONLY valid JSON. No explanation.

    Text:
    {text}
    """
    # set temperature to 0 for determinstic output 
    response = client.chat.completions.create(
        model="gpt-4o-mini", # model declaration
        max_tokens=8000,
        temperature = 0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content
    
    # Handle markdown code blocks
    if result.startswith("```"):
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
    
    try:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
        return json.loads(result), usage
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
        print(f"[ERROR] Raw response: {result[:1000]}")
        return {}, {"input_tokens": 0, "output_tokens": 0}  # Return tuple


# Keep old function for backward compatibility during transition
def parse_flight_legs_with_llm(text: str) -> list[dict]:
    """Legacy function - use parse_full_page instead."""
    result = parse_full_page(text)
    
    # Extract just flight legs for backward compatibility
    legs = []
    for seq in result.get("sequences", []):
        for dp in seq.get("duty_periods", []):
            for leg in dp.get("legs", []):
                if leg.get("leg_type") == "FLIGHT":
                    legs.append({
                        "duty_period": dp.get("duty_index"),
                        "equipment_code": leg.get("equipment_code"),
                        "flight_number": leg.get("flight_number"),
                        "depart_station": leg.get("depart", {}).get("station"),
                        "depart_time_local": leg.get("depart", {}).get("time_local"),
                        "depart_time_base": leg.get("depart", {}).get("time_base"),
                        "arrive_station": leg.get("arrive", {}).get("station"),
                        "arrive_time_local": leg.get("arrive", {}).get("time_local"),
                        "arrive_time_base": leg.get("arrive", {}).get("time_base"),
                        "block_time": leg.get("block_hhmm"),
                        "meal": leg.get("meal")
                    })
    return legs

'''
Important Considerations: 

Jump from high confidence -> low confidence = LLM non-determinism. 
Challenge of working with probabilistic models for structured data extraction in production. 
No code issue -> even with temperature of 0, the model still makes probablistic choices about tokens. 
Token bottleneck. 
Our architecture provides a strong defense mechanism -> engine catches imperfection (cross-checking).

Solutions: 

(A) Improve prompt
(B) Set temperature to 0 to make model as deterministic as possible. Add system instructions. 

Appeal: 

Engine is production-grade and ensures that the data is correct and informs if incorrect. 

Shield (Pydantic) + Radar (Regex Validation).

Solutions (Part 2):

1. Add retry on mistmatch logic. If regex count ≠ LLM count, retry automatically

'''