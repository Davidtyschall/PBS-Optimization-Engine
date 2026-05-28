"""
Tests for engine.py

These tests validate the full parsing pipeline.
Note: These are integration tests that call the actual LLM.
They cost ~$0.003 per run and take 60-70 seconds.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import parse_bid_packet, ParseResult


class TestEngineBasicFunctionality:
    """Tests for basic engine operations."""
    
    @pytest.fixture(scope="class")
    def parsed_result(self):
        """
        Parse page 7 once and share across all tests in this class.
        
        scope="class" means this runs once per class, not once per test.
        This saves money and time since LLM calls are expensive.
        """
        pdf_path = Path(__file__).parent.parent.parent / "data" / "raw" / "DFW_JAN.pdf"
        return parse_bid_packet(str(pdf_path), page_num=7)
    
    # -------------------------------------------------------------------------
    # Success Tests
    # -------------------------------------------------------------------------
    
    def test_parse_returns_success(self, parsed_result):
        """Engine should successfully parse a valid page."""
        assert parsed_result.success == True
    
    def test_parse_returns_parse_result_type(self, parsed_result):
        """Engine should return a ParseResult object."""
        assert isinstance(parsed_result, ParseResult)
    
    def test_parse_returns_data(self, parsed_result):
        """Successful parse should include data."""
        assert parsed_result.data is not None
    
    # -------------------------------------------------------------------------
    # Sequence Tests
    # -------------------------------------------------------------------------
    
    def test_parse_finds_all_sequences(self, parsed_result):
        """Page 7 should contain 7 sequences."""
        sequences = parsed_result.data.get("sequences", [])
        assert len(sequences) == 7
    
    def test_sequences_have_seq_num(self, parsed_result):
        """Every sequence should have a seq_num."""
        for seq in parsed_result.data["sequences"]:
            assert "seq_num" in seq
            assert isinstance(seq["seq_num"], int)
    
    def test_sequences_have_ops_count(self, parsed_result):
        """Every sequence should have an ops_count."""
        for seq in parsed_result.data["sequences"]:
            assert "ops_count" in seq
            assert isinstance(seq["ops_count"], int)
            assert seq["ops_count"] > 0
    
    def test_sequences_have_calendar_dates(self, parsed_result):
        """Every sequence should have calendar_start_dates."""
        for seq in parsed_result.data["sequences"]:
            assert "calendar_start_dates" in seq
            assert isinstance(seq["calendar_start_dates"], list)
            assert len(seq["calendar_start_dates"]) > 0
    
    def test_sequences_have_duty_periods(self, parsed_result):
        """Every sequence should have at least one duty period."""
        for seq in parsed_result.data["sequences"]:
            assert "duty_periods" in seq
            assert len(seq["duty_periods"]) >= 1
    
    # -------------------------------------------------------------------------
    # Calendar Resolver Integration Tests
    # -------------------------------------------------------------------------
    
    def test_seq_217_calendar_dates_correct(self, parsed_result):
        """SEQ 217 should have calendar dates [7, 11] from resolver."""
        seq_217 = next(s for s in parsed_result.data["sequences"] if s["seq_num"] == 217)
        assert seq_217["calendar_start_dates"] == [7, 11]
    
    def test_seq_227_calendar_dates_correct(self, parsed_result):
        """SEQ 227 should have 12 calendar dates from resolver."""
        seq_227 = next(s for s in parsed_result.data["sequences"] if s["seq_num"] == 227)
        expected = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        assert seq_227["calendar_start_dates"] == expected
    
    # -------------------------------------------------------------------------
    # Metrics Tests
    # -------------------------------------------------------------------------
    
    def test_result_has_cost_metrics(self, parsed_result):
        """Result should include cost metrics."""
        assert parsed_result.cost is not None
        assert hasattr(parsed_result.cost, "input_tokens")
        assert hasattr(parsed_result.cost, "output_tokens")
        assert hasattr(parsed_result.cost, "estimated_cost_usd")
    
    def test_cost_metrics_are_positive(self, parsed_result):
        """Cost metrics should be non-zero for successful parse."""
        assert parsed_result.cost.input_tokens > 0
        assert parsed_result.cost.output_tokens > 0
        assert parsed_result.cost.estimated_cost_usd > 0
    
    def test_result_has_timing_metrics(self, parsed_result):
        """Result should include timing metrics."""
        assert parsed_result.timing is not None
        assert hasattr(parsed_result.timing, "ocr_seconds")
        assert hasattr(parsed_result.timing, "llm_seconds")
        assert hasattr(parsed_result.timing, "total_seconds")
    
    def test_timing_metrics_are_positive(self, parsed_result):
        """Timing metrics should be non-zero for successful parse."""
        assert parsed_result.timing.ocr_seconds >= 0
        assert parsed_result.timing.llm_seconds > 0
        assert parsed_result.timing.total_seconds > 0
    
    def test_confidence_is_valid(self, parsed_result):
        """Confidence should be high, medium, or low."""
        assert parsed_result.confidence in ["high", "medium", "low"]
    
    # -------------------------------------------------------------------------
    # Schema Structure Tests
    # -------------------------------------------------------------------------
    
    def test_data_has_page_metadata(self, parsed_result):
        """Parsed data should include page_metadata."""
        assert "page_metadata" in parsed_result.data
    
    def test_data_has_sequences(self, parsed_result):
        """Parsed data should include sequences."""
        assert "sequences" in parsed_result.data
    
    def test_page_metadata_has_required_fields(self, parsed_result):
        """Page metadata should have required fields."""
        metadata = parsed_result.data["page_metadata"]
        assert "page_number" in metadata
        assert "bid_status" in metadata
        assert "bidding_period" in metadata


class TestEngineErrorHandling:
    """Tests for engine error handling."""
    
    def test_nonexistent_file_returns_failure(self):
        """Non-existent file should return success=False."""
        result = parse_bid_packet("/nonexistent/file.pdf", page_num=0)
        
        assert result.success == False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower() or "File not found" in result.errors[0]
    
    def test_invalid_page_returns_failure(self):
        """Invalid page number should return success=False."""
        pdf_path = Path(__file__).parent.parent.parent / "data" / "raw" / "DFW_JAN.pdf"
        result = parse_bid_packet(str(pdf_path), page_num=99999)
        
        assert result.success == False
        assert len(result.errors) > 0