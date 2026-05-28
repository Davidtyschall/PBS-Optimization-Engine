# David Schallipp
# 15 December 2025
# Parser Pipeline Module - OCR Exploration

import pymupdf
import os
import re # for regex transformations later
from llm.client import parse_flight_legs_with_llm
# import pydantic for schema validation 
# import logging ** very important for production code
# add unit testing logic 
# import hashlib for file integrity checks
# import caching library to limit lllm calls 
# batch processing 
# rate limiting per use (add logic as well)

# Moduralize the bid packet ingestion and input validation
# def validate_input(file_path): logic 

# add def parse_page(pdf_page): logic


def extract_text_with_ocr(file_path: str) -> str:
    '''
    Extract text from a PDF file.
    Falls back to OCR if PDF is image-based.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no text could be extracted
    '''
    
    # === DEBUG: Verify file exists ===
    print(f"Current working directory: {os.getcwd()}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # === Open the PDF ===
    doc = pymupdf.open(file_path)
    print(f"Number of pages: {len(doc)}")

    # === Get the first page ===
    page = doc[0]

    # === Try standard text extraction first ===
    text = page.get_text()
    print(f"Standard extraction - text length: {len(text)}")
    
    # === If empty -> try OCR ===
    if not text.strip():
        print("No text found. Attempting OCR...")
        # PyMuPDF OCR requires Tesseract installed: brew install tesseract
        # This extracts text from the image using OCR
        text = page.get_text("text", textpage=page.get_textpage_ocr())
    print(f"OCR extraction - text length: {len(text)}")
    # === Print the result ===
    print("\n" + "="*50)
    print("EXTRACTED TEXT:")
    print("="*50)
    print(text)
    
    if not text.strip():
        raise ValueError(f"Could not extract text from: {file_path}")
    return text
# Config 
FILE_PATH = "../data/raw/bid_packet_example.pdf"

  
# # === If empty, the PDF is image-based. Use OCR. ===
# if not text.strip():
#     print("No text found. Attempting OCR...")
    
#     # PyMuPDF OCR requires Tesseract installed: brew install tesseract
#     # This extracts text from the image using OCR
#     text = page.get_text("text", textpage=page.get_textpage_ocr())
    
#     print(f"OCR extraction - text length: {len(text)}")

# # === Print the result ===
# print("\n" + "="*50)
# print("EXTRACTED TEXT:")
# print("="*50)
# print(text)

# if not text found from OCR -> raise error + log it + handle logic below

# === Save output for transformation using python regex rules ===
# with open("data/parsed/ocr_output_page902.txt", "w") as f:
#     f.write(text)
# print("Saved to data/parsed/ocr_output_page902.txt")

# text = """SEQ 217
# 2 OPS
# POSN CA FO"""

# After OCR extraction
# Function 1
# === Apply regex transformations to extract sequence headers ===
def parse_seq_headers(text: str) -> list[dict]:
    '''
    Extract sequence headers from bid packet text.
    
    Returns:
        List of dicts with keys: seq_num, ops_count, positions
    '''
    pattern = r"SEQ\s+(\d+)\s+(\d+)\s+OPS\s+_*POSN\s+([A-Z]{2})\s+([A-Z]{2})"
    matches = re.findall(pattern, text)
    results = []
    for match in matches:
        results.append({
            "seq_num": int(match[0]),
            "ops_count": int(match[1]),
            "positions": [match[2], match[3]]
        })
    return results

# Function 2
# === Apply regex transformations to extract time-related data ===
def parse_rpt_times(text: str) -> list[dict]:
    '''
    Return dict with keys for RPT and RLS times.
    {"local": "0800", "base": "0800"}.
    '''
    
    rpt_pattern = r"RPT\s+(\d{4})/(\d{4})"
    rpt_matches = re.findall(rpt_pattern, text)
    results = []
    for match in rpt_matches:
        results.append({
            "local": match[0],
            "base": match[1]
        })
    return results

# Function 3
# === Apply regex transformations to extract release times  ===

def parse_rls_times(text: str) -> list[dict]:
    '''
    RLS = Release Time 
    Return dict with keys for RLS and RLS times.
    {"local": "1200", "base": "1200"}.
    Time a pilot is released from a sequence. 
    '''
    
    rls_pattern = r"RLS\s+(\d{4})/(\d{4})"
    rls_matches = re.findall(rls_pattern, text)
    results = []
    for match in rls_matches:
        results.append({
            "local": match[0],
            "base": match[1]
        })
    return results

# Function 4 
# === Apply regex transformations to extract TTL totals  ===

def parse_ttl_totals(text: str) -> list[dict]:
    '''
    Extract TTL (totals) from bid packet text.
    Normalizes OCR-corrupted values (e.g., "3110" -> "3.10").
    
    Returns:
        List of dicts with keys: total_block, total_synthetic, total_tpay, tafb
    '''
    # Helper function to fix OCR errors
    def normalize_time(value: str) -> str:
        value = value.replace(':', '.')
        if '.' not in value:
            value = value[:-2] + '.' + value[-2:]
        return value
    
    pattern = r"TTL\s+([\d.:]+)\s+([\d.:]+)\s+([\d.:]+)\s+([\d.:]+)"
    matches = re.findall(pattern, text)
    
    results = []
    for match in matches:
        results.append({
            "total_block": normalize_time(match[0]),
            "total_synthetic": normalize_time(match[1]),
            "total_tpay": normalize_time(match[2]),
            "tafb": normalize_time(match[3])
        })
    return results

# === Pipeline Orchestrator Function  ===

def parse_bid_packet(file_path: str) -> dict:
    '''
    Orchestrates the parsing of bid packet text.
    
    Returns:
        Parsed data as a dictionary.
        
    Args:
    file_path: Path to bid packet PDF file.
        
    Returns:
        Dict with keys: text_length, seq_headers, rpt_times, rls_times, ttl_totals
    '''
    
    text = extract_text_with_ocr(file_path)
    parsed_data = {
        "sequences": parse_seq_headers(text),
        "rpt_times": parse_rpt_times(text),
        "rls_times": parse_rls_times(text),
        "ttl_totals": parse_ttl_totals(text),
        "flight_legs": parse_flight_legs_with_llm(text)
    }
    return parsed_data

# === MAIN EXECUTION ===
'''
Allowf for testing the pipeline, and importing it 
from parser import pipeline 
'''

if __name__ == "__main__":
    # This only runs when you execute: python parser_pipeline.py
    # It does NOT run when you import this file elsewhere
    
    result = parse_bid_packet(FILE_PATH)
    
    print("=" * 50)
    print("PARSING RESULTS")
    print("=" * 50)
    print(f"Sequences found: {len(result['sequences'])}")
    print(f"RPT times found: {len(result['rpt_times'])}")
    print(f"RLS times found: {len(result['rls_times'])}")
    print(f"TTL totals found: {len(result['ttl_totals'])}")
    print()
    print("Sequences:")
    for seq in result['sequences']:
        print(f"  SEQ {seq['seq_num']}: {seq['ops_count']} OPS, positions {seq['positions']}")
        
    print(f"Flight legs found: {len(result['flight_legs'])}")
    print()
    print("Flight Legs:")
    for leg in result['flight_legs']:
        print(f"  {leg['depart_station']} -> {leg['arrive_station']}: Flight {leg['flight_number']}, Block: {leg['block_time']}")


# === Add LLM for complex/noisy patterns  ===

# === Assemble and validation phase  ===

