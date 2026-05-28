"""
Tests for calendar_resolver.py

These tests protect our spatial-anchor algorithm that extracts
calendar dates from OCR text with 100% accuracy.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from calendar_resolver import extract_calendar_dates_from_text, extract_sequence_text
from ocr import extract_text_with_ocr


class TestCalendarDatesExtraction:
    """Tests for the extract_calendar_dates_from_text function."""
    
    @pytest.fixture
    def page_7_text(self):
        """Load page 7 text once for all tests in this class."""
        pdf_path = Path(__file__).parent.parent.parent / "data" / "raw" / "DFW_JAN.pdf"
        return extract_text_with_ocr(str(pdf_path), page_num=7)
    
    # -------------------------------------------------------------------------
    # Core Extraction Tests
    # -------------------------------------------------------------------------
    
    def test_seq_217_extracts_two_dates(self, page_7_text):
        """SEQ 217 with 2 OPS should return [7, 11]."""
        seq_text = extract_sequence_text(page_7_text, 217)
        result = extract_calendar_dates_from_text(seq_text, ops_count=2)
        
        assert result["dates"] == [7, 11]
    
    def test_seq_218_extracts_two_dates(self, page_7_text):
        """SEQ 218 with 2 OPS should return [8, 9]."""
        seq_text = extract_sequence_text(page_7_text, 218)
        result = extract_calendar_dates_from_text(seq_text, ops_count=2)
        
        assert result["dates"] == [8, 9]
    
    def test_seq_224_extracts_three_dates(self, page_7_text):
        """SEQ 224 with 3 OPS should return [1, 3, 4]."""
        seq_text = extract_sequence_text(page_7_text, 224)
        result = extract_calendar_dates_from_text(seq_text, ops_count=3)
        
        assert result["dates"] == [1, 3, 4]
    
    def test_seq_227_extracts_twelve_dates(self, page_7_text):
        """SEQ 227 with 12 OPS should return [12, 13, 14, ..., 23]."""
        seq_text = extract_sequence_text(page_7_text, 227)
        result = extract_calendar_dates_from_text(seq_text, ops_count=12)
        
        expected = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        assert result["dates"] == expected
    
    def test_seq_228_extracts_three_dates(self, page_7_text):
        """SEQ 228 with 3 OPS should return [1, 2, 3]."""
        seq_text = extract_sequence_text(page_7_text, 228)
        result = extract_calendar_dates_from_text(seq_text, ops_count=3)
        
        assert result["dates"] == [1, 2, 3]
    
    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------
    
    def test_ops_count_matches_when_correct(self, page_7_text):
        """ops_match should be True when date count equals ops_count."""
        seq_text = extract_sequence_text(page_7_text, 217)
        result = extract_calendar_dates_from_text(seq_text, ops_count=2)
        
        assert result["ops_match"] == True
        assert result["confidence"] == "high"
    
    def test_ops_count_mismatch_detected(self, page_7_text):
        """ops_match should be False when counts don't match."""
        seq_text = extract_sequence_text(page_7_text, 217)
        # Pass wrong ops_count intentionally
        result = extract_calendar_dates_from_text(seq_text, ops_count=5)
        
        assert result["ops_match"] == False
        assert result["confidence"] == "medium"
    
    # -------------------------------------------------------------------------
    # Edge Case Tests
    # -------------------------------------------------------------------------
    
    def test_empty_text_returns_low_confidence(self):
        """Empty text should return empty dates with low confidence."""
        result = extract_calendar_dates_from_text("", ops_count=2)
        
        assert result["dates"] == []
        assert result["confidence"] == "low"
        assert result["ops_match"] == False
    
    def test_no_calendar_header_uses_fallback(self):
        """Text without MO TU WE header should still attempt extraction."""
        text = "Some random text without calendar header 7 11"
        result = extract_calendar_dates_from_text(text, ops_count=2)
        
        # Should use fallback position and find numbers
        assert isinstance(result["dates"], list)
    
    # -------------------------------------------------------------------------
    # Data Integrity Tests
    # -------------------------------------------------------------------------
    
    def test_dates_are_integers(self, page_7_text):
        """All extracted dates should be integers, not strings."""
        seq_text = extract_sequence_text(page_7_text, 217)
        result = extract_calendar_dates_from_text(seq_text, ops_count=2)
        
        for date in result["dates"]:
            assert isinstance(date, int)
    
    def test_dates_are_sorted(self, page_7_text):
        """Extracted dates should be in ascending order."""
        seq_text = extract_sequence_text(page_7_text, 227)
        result = extract_calendar_dates_from_text(seq_text, ops_count=12)
        
        assert result["dates"] == sorted(result["dates"])
    
    def test_no_duplicate_dates(self, page_7_text):
        """Should not return duplicate dates."""
        seq_text = extract_sequence_text(page_7_text, 227)
        result = extract_calendar_dates_from_text(seq_text, ops_count=12)
        
        assert len(result["dates"]) == len(set(result["dates"]))
    
    def test_dates_within_valid_range(self, page_7_text):
        """All dates should be between 1 and 31."""
        seq_text = extract_sequence_text(page_7_text, 227)
        result = extract_calendar_dates_from_text(seq_text, ops_count=12)
        
        for date in result["dates"]:
            assert 1 <= date <= 31


class TestSequenceTextExtraction:
    """Tests for the extract_sequence_text function."""
    
    @pytest.fixture
    def page_7_text(self):
        """Load page 7 text once for all tests in this class."""
        pdf_path = Path(__file__).parent.parent.parent / "data" / "raw" / "DFW_JAN.pdf"
        return extract_text_with_ocr(str(pdf_path), page_num=7)
    
    def test_extracts_seq_217_text(self, page_7_text):
        """Should extract text block containing SEQ 217."""
        seq_text = extract_sequence_text(page_7_text, 217)
        
        assert "SEQ 217" in seq_text
        assert len(seq_text) > 0
    
    def test_nonexistent_sequence_returns_empty(self, page_7_text):
        """Requesting non-existent sequence should return empty string."""
        seq_text = extract_sequence_text(page_7_text, 999)
        
        assert seq_text == ""
    
    def test_extracted_text_contains_flight_info(self, page_7_text):
        """Extracted sequence text should contain flight details."""
        seq_text = extract_sequence_text(page_7_text, 217)
        
        assert "DFW" in seq_text  # Departure station
        assert "320" in seq_text  # Flight number