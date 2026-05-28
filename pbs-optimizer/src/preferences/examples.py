"""
Few-Shot Examples for NL Preference Parser
==========================================
These examples teach the LLM how to map natural language to our preference schema.
"""

FEW_SHOT_EXAMPLES = [
    # =========================================================================
    # SIMPLE SINGLE PREFERENCES
    # =========================================================================
    {
        "input": "I want Fridays off",
        "output": {
            "days_off": {
                "days_off": ["Friday"]
            }
        },
        "explanation": "Simple day-off request"
    },
    {
        "input": "No redeyes please",
        "output": {
            "pairing": {
                "avoid_pairing_type": ["Redeye", "Trailing Redeye"]
            }
        },
        "explanation": "Avoiding a pairing type - include trailing redeye as well"
    },
    {
        "input": "I'd like Hawaii layovers",
        "output": {
            "layovers": {
                "prefer_layover_city": ["HNL", "OGG", "LIH", "KOA"]
            }
        },
        "explanation": "Region expands to all Hawaii airports"
    },
    
    # =========================================================================
    # MULTIPLE PREFERENCES
    # =========================================================================
    {
        "input": "Weekends off, short trips only, and I want to maximize my pay",
        "output": {
            "days_off": {
                "days_off": ["Saturday", "Sunday"],
                "maximize_weekends_off": True
            },
            "pairing": {
                "max_pairing_length": 2
            },
            "credit": {
                "maximize_credit": True
            }
        },
        "explanation": "Multiple preferences across categories"
    },
    {
        "input": "Tokyo or Seoul layovers, 3-day trips, no early morning reports",
        "output": {
            "layovers": {
                "prefer_layover_city": ["NRT", "HND", "ICN"]
            },
            "pairing": {
                "prefer_pairing_length": [3]
            },
            "times": {
                "report_after_minutes": 480
            }
        },
        "explanation": "City names map to airport codes, early morning avoided via report time"
    },
    {
        "input": "I need the 15th and 16th off for my daughter's wedding, and prefer European layovers",
        "output": {
            "days_off": {
                "must_off_dates": ["2026-01-15", "2026-01-16"]
            },
            "layovers": {
                "prefer_region": ["Europe"]
            }
        },
        "explanation": "Specific dates as must-have, region preference"
    },
    
    # =========================================================================
    # COMMUTER SCENARIOS
    # =========================================================================
    {
        "input": "I commute from Denver so I need late reports and early releases",
        "output": {
            "times": {
                "is_commuter": True,
                "commuter_base": "DEN",
                "report_after_minutes": 600,
                "release_before_minutes": 1080
            }
        },
        "explanation": "Commuter implies time constraints - 10am report, 6pm release"
    },
    {
        "input": "I'm a commuter from Phoenix, no early trips",
        "output": {
            "times": {
                "is_commuter": True,
                "commuter_base": "PHX"
            },
            "pairing": {
                "avoid_pairing_type": ["Early"]
            }
        },
        "explanation": "Commuter base identified, early trips avoided"
    },
    
    # =========================================================================
    # SYNONYMS AND INFORMAL LANGUAGE
    # =========================================================================
    {
        "input": "give me the japan trips, hate redeyes, wanna be home by 6",
        "output": {
            "layovers": {
                "prefer_layover_city": ["NRT", "HND"]
            },
            "pairing": {
                "avoid_pairing_type": ["Redeye", "Trailing Redeye"]
            },
            "times": {
                "release_before_minutes": 1080
            }
        },
        "explanation": "Informal language parsed correctly, japan=NRT/HND, 6=6pm=1080min"
    },
    {
        "input": "turns only, max pay, no overnights",
        "output": {
            "pairing": {
                "prefer_pairing_type": ["Turn"],
                "max_pairing_length": 1
            },
            "credit": {
                "maximize_credit": True
            }
        },
        "explanation": "Turns = 1-day trips, no overnights reinforces this"
    },
    
    # =========================================================================
    # EDGE CASES
    # =========================================================================
    {
        "input": "I want efficient trips with good credit to TAFB ratio",
        "output": {
            "credit": {
                "prefer_efficient_flying": True,
                "minimize_tafb": True
            },
            "pairing": {
                "max_tafb_credit_ratio": 2.5
            }
        },
        "explanation": "Technical request about efficiency metrics"
    },
    {
        "input": "Maybe Fridays off if possible, definitely no redeyes",
        "output": {
            "days_off": {
                "days_off": ["Friday"],
                "days_off_strength": "weak"
            },
            "pairing": {
                "avoid_pairing_type": ["Redeye", "Trailing Redeye"]
            }
        },
        "explanation": "Hedging language (maybe, if possible) = weak strength"
    },
]


def get_examples_for_prompt() -> str:
    """
    Format examples for inclusion in the LLM prompt.
    
    Returns:
        Formatted string of examples
    """
    lines = []
    
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Input: \"{example['input']}\"")
        lines.append(f"Output: {_format_json(example['output'])}")
        lines.append("")
    
    return "\n".join(lines)


def _format_json(obj: dict, indent: int = 2) -> str:
    """Format dict as compact JSON string."""
    import json
    return json.dumps(obj, indent=indent)


# Test the examples
if __name__ == "__main__":
    print("=" * 60)
    print("FEW-SHOT EXAMPLES FOR NL PARSER")
    print("=" * 60)
    print(f"Total examples: {len(FEW_SHOT_EXAMPLES)}")
    print()
    print(get_examples_for_prompt())