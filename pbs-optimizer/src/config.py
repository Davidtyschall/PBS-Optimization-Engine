import os 
from pathlib import Path 

# Path Configs 

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"

# Default file paths 

DEFAULT_BID_PACKET = RAW_DIR / "DFW_JAN.pdf"

# Configure LLMS 

LLM_MODEL = "gpt-4o-mini" # Model name 
LLM_MAX_TOKENS = 8000

# ENSURE THAT THE DIRECTORIES EXIST 

PARSED_DIR.mkdir(parents=True, exist_ok=True)